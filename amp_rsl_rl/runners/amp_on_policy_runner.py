# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import pathlib
import statistics
import time
from collections import deque
from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan, resolve_callable

import amp_rsl_rl
from amp_rsl_rl.algorithms import AMP_PPO  # noqa: F401  (registers class_name)
from amp_rsl_rl.networks import Discriminator, ActorCriticMoE  # noqa: F401
from amp_rsl_rl.utils import export_policy_as_onnx
from amp_rsl_rl.utils.registry import resolve_amp_callable

# Backward-compatible alias: class resolution now lives in rsl-rl.
resolve_class = resolve_callable


class AMPOnPolicyRunner(OnPolicyRunner):
    """AMP on-policy runner built on rsl-rl v5.5.0 :class:`OnPolicyRunner`.

    The runner delegates algorithm construction to
    :meth:`AMP_PPO.construct_algorithm` (which builds the actor, critic, storage,
    discriminator and expert-motion loader), reuses the base class multi-GPU
    setup, and keeps AMP-specific logging (style/task reward split, discriminator
    metrics) and ONNX export. The style reward is mixed into the environment
    reward inside :meth:`AMP_PPO.process_env_step`.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.style_weight = train_cfg.get("style_weight", 0.5)

        # Multi-GPU configuration (inherited helper); sets cfg["multi_gpu"].
        self._configure_multi_gpu()

        # Build the algorithm (actor + critic + storage + discriminator + amp data).
        obs = self.env.get_observations()
        alg_class = resolve_amp_callable(self.cfg["algorithm"]["class_name"])
        self.alg: AMP_PPO = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)
        self.discriminator = self.alg.discriminator

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Logging / checkpoint state
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__, amp_rsl_rl.__file__]

        # Optional custom exporter function (set via set_export_policy_fn).
        self._export_policy_fn: Callable | None = None

    # ------------------------------------------------------------------ logging

    def _init_writer(self) -> None:
        if self.log_dir is None or self.writer is not None:
            return
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_log_writer import NeptuneLogWriter

            self.writer = NeptuneLogWriter(
                log_dir=self.log_dir, project_name=self.cfg.get("neptune_project")
            )
            self.writer.store_config(self.env.cfg, self.cfg)
        elif self.logger_type == "wandb":
            from amp_rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.store_config(self.env.cfg, self.cfg)
        elif self.logger_type == "mlflow":
            from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

            self.writer = MLflowSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.store_config(self.env.cfg, self.cfg)
        elif self.logger_type == "tensorboard":
            self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise AssertionError(f"logger type '{self.logger_type}' not found")

    # ------------------------------------------------------------------ learn

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._init_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            self.alg.broadcast_parameters()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # Accumulate on-device to avoid a GPU->CPU sync every step.
            mean_style_reward_log = torch.zeros((), device=self.device)
            mean_task_reward_log = torch.zeros((), device=self.device)

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Style reward is mixed into ``rewards`` inside process_env_step.
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    mean_task_reward_log += self.alg.task_reward
                    mean_style_reward_log += self.alg.style_reward

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = torch.nonzero(dones, as_tuple=False)
                        if new_ids.numel() > 0:
                            env_indices = new_ids.view(-1)
                            rewbuffer.extend(cur_reward_sum[env_indices].cpu().tolist())
                            lenbuffer.extend(cur_episode_length[env_indices].cpu().tolist())
                            cur_reward_sum[env_indices] = 0
                            cur_episode_length[env_indices] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            # Single synchronization point for the whole rollout.
            mean_style_reward_log = mean_style_reward_log.item() / self.num_steps_per_env
            mean_task_reward_log = mean_task_reward_log.item() / self.num_steps_per_env

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)
            ep_infos.clear()

            if it == start_iter and self.log_dir is not None:
                git_file_paths = self._store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ("wandb", "neptune", "mlflow") and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
                save_onnx=True,
            )

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        loss_dict = locs["loss_dict"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std_value = self.alg.get_policy().output_std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # PPO + AMP losses (returned as a dict by AMP_PPO.update)
        self.writer.add_scalar("Loss/value_function", loss_dict["value"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", loss_dict["surrogate"], locs["it"])
        self.writer.add_scalar("Loss/entropy", loss_dict["entropy"], locs["it"])
        self.writer.add_scalar("Loss/amp_loss", loss_dict["amp"], locs["it"])
        self.writer.add_scalar("Loss/grad_pen_loss", loss_dict["grad_pen"], locs["it"])
        self.writer.add_scalar("Loss/policy_pred", loss_dict["policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", loss_dict["expert_pred"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_policy", loss_dict["accuracy_policy"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_expert", loss_dict["accuracy_expert"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/mean_kl_divergence", loss_dict["kl_divergence"], locs["it"])
        if "symmetry" in loss_dict:
            self.writer.add_scalar("Loss/symmetry", loss_dict["symmetry"], locs["it"])
        if "rnd" in loss_dict:
            self.writer.add_scalar("Loss/rnd", loss_dict["rnd"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std_value.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if self.log_dir and self.logger_type in ("wandb", "mlflow"):
            self.writer.add_video_files(self.log_dir, step=locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
            )
            self.writer.add_scalar("Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"])
            self.writer.add_scalar("Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"])
            # wandb/mlflow do not support non-integer x-axis logging.
            if self.logger_type not in ("wandb", "mlflow"):
                self.writer.add_scalar(
                    "Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{header.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s,"""
            f""" learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {loss_dict['value']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {loss_dict['surrogate']:.4f}\n"""
            f"""{'AMP loss:':>{pad}} {loss_dict['amp']:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std_value.item():.2f}\n"""
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        log_string += ep_string

        eta_seconds = (
            self.tot_time / (locs["it"] + 1) * (locs["num_learning_iterations"] - locs["it"])
        )
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)

    # ------------------------------------------------------------------ export

    def set_export_policy_fn(self, fn: Callable) -> None:
        """Set a custom function used to export the policy to ONNX."""
        self._export_policy_fn = fn

    # ------------------------------------------------------------- save / load

    def save(self, path: str, infos=None, save_onnx: bool = False) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)

        if self.logger_type in ("neptune", "wandb", "mlflow"):
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            onnx_folder = os.path.dirname(path)
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"
            actor = self.alg.get_policy()
            normalizer = getattr(
                actor, "actor_obs_normalizer", getattr(actor, "obs_normalizer", None)
            )
            export_fn = self._export_policy_fn or export_policy_as_onnx
            export_fn(
                actor,
                normalizer=normalizer,
                path=onnx_folder,
                filename=onnx_model_name,
            )
            if self.logger_type in ("neptune", "wandb", "mlflow"):
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        weights_only: bool = False,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        **kwargs,
    ) -> dict | None:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "discriminator": True,
                "optimizer": load_optimizer,
                "iteration": True,
                "rnd": True,
            }
        else:
            load_cfg.setdefault("optimizer", load_optimizer)

        loaded_dict = torch.load(
            path,
            map_location=(map_location if map_location is not None else self.device),
            weights_only=weights_only,
        )
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    # ------------------------------------------------------------------ modes

    def get_inference_policy(self, device=None) -> Callable:
        self.eval_mode()
        actor = self.alg.get_policy()
        if device is not None:
            actor.to(device)
        return lambda obs: actor(obs, stochastic_output=False)

    def train_mode(self) -> None:
        self.alg.train_mode()

    def eval_mode(self) -> None:
        self.alg.eval_mode()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    # ------------------------------------------------------------------ git

    @staticmethod
    def _store_code_state(log_dir: str, repos: list[str]) -> list[str]:
        """Store the git diff of the given repositories under ``log_dir/git``."""
        try:
            import git
        except ImportError:
            return []
        git_log_dir = os.path.join(log_dir, "git")
        os.makedirs(git_log_dir, exist_ok=True)
        file_paths: list[str] = []
        for repository_file_path in repos:
            try:
                repo = git.Repo(repository_file_path, search_parent_directories=True)
                tree = repo.head.commit.tree
                commit_hash = repo.head.commit.hexsha
            except Exception:
                print(f"Could not find git repository in {repository_file_path}. Skipping.")
                continue
            repo_name = pathlib.Path(repo.working_dir).name
            diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
            if os.path.isfile(diff_file_name):
                continue
            with open(diff_file_name, "x", encoding="utf-8") as f:
                f.write(
                    f"--- git commit ---\n{commit_hash}\n\n\n"
                    f"--- git status ---\n{repo.git.status()} \n\n\n"
                    f"--- git diff ---\n{repo.git.diff(tree)}"
                )
            file_paths.append(diff_file_name)
        return file_paths
