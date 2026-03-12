# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger

import amp_rsl_rl
from amp_rsl_rl.utils import AMPLoader
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator, ActorCriticMoE
from amp_rsl_rl.utils import export_policy_as_onnx


class AMPOnPolicyRunner:
    """
    AMPOnPolicyRunner is a high-level orchestrator that manages the training and evaluation
    of a policy using Adversarial Motion Priors (AMP) combined with on-policy reinforcement learning (PPO).

    It brings together multiple components:
    - Environment (`VecEnv`)
    - Policy (`ActorCritic`, `ActorCriticRecurrent`)
    - Discriminator (Discriminator)
    - Expert dataset (AMPLoader)
    - Reward combination (task + style)
    - Logging and checkpointing

    ---
    🔧 Configuration
    ----------------
    The class expects a `train_cfg` dictionary structured with keys:
    - "obs_groups": optional mapping describing which observation tensors belong to "policy" and "critic" inputs.
    - "policy": configuration for the policy network, including `"class_name"`
    - "algorithm": configuration for PPO/AMP_PPO, including `"class_name"`
    - "discriminator": configuration for the AMP discriminator
    - "dataset": dictionary forwarded to `AMPLoader`, containing at least:
        * "amp_data_path": folder with the `.npy` expert datasets
        * "datasets": mapping of dataset name -> sampling weight (floats)
        * "slow_down_factor": slowdown applied to real motion data to match sim dynamics
    - "num_steps_per_env": rollout horizon per environment
    - "save_interval": frequency (in iterations) for model checkpointing
    - "empirical_normalization": (deprecated) legacy flag mirrored to `policy.actor_obs_normalization`
    - "logger": one of "tensorboard", "wandb", or "neptune"

    ---
    📦 Dataset format
    ------------------
    The expert motion datasets loaded via `AMPLoader` must be `.npy` files with a dictionary containing:

    - `"joints_list"`: List[str] — ordered list of joint names
    - `"joint_positions"`: List[np.ndarray] — joint configurations per timestep (1D arrays)
    - `"root_position"`: List[np.ndarray] — base position in world coordinates
    - `"root_quaternion"`: List[np.ndarray] — base orientation in **`xyzw`** format (SciPy default)
    - `"fps"`: float — original dataset frame rate

    Internally:
    - Quaternions are interpolated via SLERP and converted to **`wxyz`** format before being used by the model (to match Isaac Gym convention).
    - Velocities are estimated with finite differences.
    - All data is converted to torch tensors and placed on the desired device.

    ---
    🎓 AMP Reward
    -------------
    During each training step, the runner collects AMP-specific observations and computes
    a discriminator-based "style reward" from the expert dataset. This is combined
    with the environment reward as:

        `reward = (1-<style_weight>) * task_reward + <style_weight> * style_reward`

    This can be later generalized into a weighted or learned reward mixing policy.

    ---
    🔁 Training loop
    ----------------
    The `learn()` method performs:
    - `rollout`: collects TensorDict observations via `self.alg.act()` and `env.step()`
    - `style_reward`: computed from discriminator via `predict_reward(...)`
    - `storage update`: via `process_env_step()` and `process_amp_step()`
    - `return computation`: via `compute_returns()` using the latest observation TensorDict
    - `update`: performs backpropagation with `self.alg.update()`
    - Logging via TensorBoard/WandB/Neptune

    ---
    💾 Saving and ONNX export
    --------------------------
    At each `save_interval`, the runner:
    - Saves the full state (`model`, `optimizer`, `discriminator`, etc.)
    - Optionally exports the policy as an ONNX model for deployment
    - Uploads checkpoints to logging services if enabled

    ---
    📤 Inference policy
    -------------------
    `get_inference_policy()` returns a callable that takes an observation and returns an action.
    If empirical normalization is enabled, observations are normalized before inference.

    ---
    🛠️ Additional tools
    -------------------
    - Git integration via `store_code_state()` to track code changes
    - Logging of learning statistics, reward breakdown, discriminator metrics
    - Compatible with multi-task setups via dataset weights

    ---
    📚 Notes
    --------
    - This runner assumes an AMP-compatible VecEnv, providing `observations["amp"]`
    - AMP uses both current and next state to train the discriminator
    - Logging behavior is separated from core logic (WandB, Neptune, TensorBoard)
    - The Discriminator and AMP_PPO must follow expected APIs

    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.actor_cfg = train_cfg["actor"]
        self.critic_cfg = train_cfg["critic"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.dataset_cfg = train_cfg["dataset"]
        self.device = device
        self.env = env
        self.style_weight = train_cfg.get("style_weight", 0.5)  # default to equal weighting if not specified

        observations = self.env.get_observations()
        default_sets = ["critic"]
        self.cfg["obs_groups"] = resolve_obs_groups(
            observations, self.cfg.get("obs_groups"), default_sets
        )
        self.actor_cfg.pop("cnn_cfg", None)  # remove cnn_cfg if present, since it's not used in MLPModel
        self.critic_cfg.pop("cnn_cfg", None)  # remove cnn_cfg if present, since it's not used in MLPModel
        actor_class = eval(self.actor_cfg.pop("class_name", "MLPModel"))
        critic_class = eval(self.critic_cfg.pop("class_name", "MLPModel"))
        
        actor: MLPModel = actor_class(
            observations,
            self.cfg["obs_groups"],
            "actor",
            self.env.num_actions,
            **self.actor_cfg,
        ).to(self.device)
        
        critic: MLPModel = critic_class(
            observations,
            self.cfg["obs_groups"],
            "critic",
            1,
            **self.critic_cfg,
        ).to(self.device)

        amp_joint_names = self.dataset_cfg.get("amp_joint_names", None)
        simulation_dt = self._resolve_simulation_dt()

        # Initialize all the ingredients required for AMP (discriminator, dataset loader)
        num_amp_obs = self._flatten_amp_obs(observations["amp"]).shape[1]
        amp_data = AMPLoader(
            device=self.device,
            dataset_path_root=self.dataset_cfg["amp_data_path"],
            datasets=self.dataset_cfg["datasets"],
            simulation_dt=simulation_dt,
            slow_down_factor=self.dataset_cfg["slow_down_factor"],
            expected_joint_names=amp_joint_names,
        )

        self.discriminator = Discriminator(
            input_dim=num_amp_obs
            * 2,  # the discriminator takes in the concatenation of the current and next observation
            hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
            reward_scale=self.discriminator_cfg["reward_scale"],
            device=self.device,
            loss_type=self.discriminator_cfg["loss_type"],
            empirical_normalization=self.discriminator_cfg["empirical_normalization"],
        ).to(self.device)

        # Initialize the PPO algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # AMP_PPO
        # This removes from alg_cfg fields that are not in AMP_PPO but are introduced in rsl_rl 2.2.3 PPO
        # normalize_advantage_per_mini_batch=False,
        # rnd_cfg: dict | None = None,
        # symmetry_cfg: dict | None = None,
        # multi_gpu_cfg: dict | None = None,
        for key in list(self.alg_cfg.keys()):
            if key not in AMP_PPO.__init__.__code__.co_varnames:
                self.alg_cfg.pop(key)

        self.alg: AMP_PPO = alg_class(
            actor=actor,
            critic=critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        # init storage and model
        obs_template = observations.clone().detach().to(self.device)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_template,
            (self.env.num_actions,),
        )


        # Log
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__, amp_rsl_rl.__file__]

    @staticmethod
    def _flatten_amp_obs(amp_obs):
        """Return AMP observations as a flat tensor regardless of source format."""
        if isinstance(amp_obs, torch.Tensor):
            return amp_obs
        if isinstance(amp_obs, dict):
            if "joint_pos" in amp_obs and "joint_vel" in amp_obs:
                return torch.cat([amp_obs["joint_pos"], amp_obs["joint_vel"]], dim=-1)
            keys = sorted(list(amp_obs.keys()))
            return torch.cat([amp_obs[key] for key in keys], dim=-1)
        if hasattr(amp_obs, "keys") and "joint_pos" in amp_obs and "joint_vel" in amp_obs:
            return torch.cat([amp_obs["joint_pos"], amp_obs["joint_vel"]], dim=-1)
        if hasattr(amp_obs, "keys"):
            keys = sorted(list(amp_obs.keys()))
            return torch.cat([amp_obs[key] for key in keys], dim=-1)
        raise TypeError(f"Unsupported AMP observation type: {type(amp_obs)}")

    def _resolve_simulation_dt(self) -> float:
        """Resolve effective simulation dt used by AMP loader across cfg variants."""
        sim_cfg = getattr(self.env.cfg, "sim", None)
        decimation = float(getattr(self.env.cfg, "decimation", 1))

        base_dt = None
        if sim_cfg is not None:
            base_dt = getattr(sim_cfg, "dt", None)
            if base_dt is None:
                mujoco_cfg = getattr(sim_cfg, "mujoco", None)
                if mujoco_cfg is not None:
                    base_dt = getattr(mujoco_cfg, "timestep", None)

        if base_dt is None:
            raise AttributeError(
                "Unable to resolve simulation dt from env config. Expected either "
                "`env.cfg.sim.dt` or `env.cfg.sim.mujoco.timestep`."
            )

        return float(base_dt) * decimation

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                from amp_rsl_rl.utils.wandb_utils import WandbSummaryWriter
                import wandb

                # Update the run name with a sequence number. This function is useful to
                # replicate the same behaviour of rsl-rl-lib before v2.3.0
                def update_run_name_with_sequence(prefix: str) -> None:
                    # Retrieve the current wandb run details (project and entity)
                    project = wandb.run.project
                    entity = wandb.run.entity

                    # Use wandb's API to list all runs in your project
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project}")

                    max_num = 0
                    # Iterate through runs to extract the numeric suffix after the prefix.
                    for run in runs:
                        if run.name.startswith(prefix):
                            # Extract the numeric part from the run name.
                            numeric_suffix = run.name[
                                len(prefix) :
                            ]  # e.g., from "prefix564", get "564"
                            try:
                                run_num = int(numeric_suffix)
                                if run_num > max_num:
                                    max_num = run_num
                            except ValueError:
                                continue

                    # Increment to get the new run number
                    new_num = max_num + 1
                    new_run_name = f"{prefix}{new_num}"

                    # Update the wandb run's name
                    wandb.run.name = new_run_name
                    print("Updated run name to:", wandb.run.name)

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                update_run_name_with_sequence(prefix=self.cfg["wandb_project"])

                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        amp_obs = self._flatten_amp_obs(obs["amp"]).clone()
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout

            mean_style_reward_log = 0
            mean_task_reward_log = 0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    self.alg.act_amp(amp_obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    next_amp_obs = self._flatten_amp_obs(obs["amp"]).clone()
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs
                    )

                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    rewards = (1 - self.style_weight) * rewards + self.style_weight * style_rewards

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.alg.process_amp_step(next_amp_obs)

                    amp_obs = next_amp_obs

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
                            lenbuffer.extend(
                                cur_episode_length[env_indices].cpu().tolist()
                            )
                            cur_reward_sum[env_indices] = 0
                            cur_episode_length[env_indices] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs)

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_kl_divergence,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = []
                try:
                    # Older rsl_rl API: static method taking (log_dir, repos)
                    git_file_paths = Logger._store_code_state(self.log_dir, self.git_status_repos)
                except TypeError:
                    # Newer rsl_rl API: instance method with no args
                    try:
                        logger_stub = Logger.__new__(Logger)
                        logger_stub.log_dir = self.log_dir
                        logger_stub.disable_logs = False
                        logger_stub.git_status_repos = self.git_status_repos
                        git_file_paths = Logger._store_code_state(logger_stub)
                    except Exception:
                        git_file_paths = []
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
            save_onnx=True,
        )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if getattr(self.alg.actor, "noise_std_type", "scalar") == "log":
            mean_std_value = torch.exp(self.alg.actor.log_std).mean()
        else:
            mean_std_value = self.alg.actor.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )

        # Adding logging due to AMP
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar(
            "Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"]
        )

        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar(
            "Loss/mean_kl_divergence", locs["mean_kl_divergence"], locs["it"]
        )
        self.writer.add_scalar(
            "Policy/mean_noise_std", mean_std_value.item(), locs["it"]
        )
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        if self.log_dir and self.logger_type == "wandb":
            self.writer.add_video_files(self.log_dir, step=locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"]
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std_value.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std_value.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # make the eta in H:M:S
        eta_seconds = (
            self.tot_time
            / (locs["it"] + 1)
            * (locs["num_learning_iterations"] - locs["it"])
        )

        # Convert seconds to H:M:S
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

    def save(self, path, infos=None, save_onnx=False):
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            # Save the model in ONNX format
            # extract the folder path
            onnx_folder = os.path.dirname(path)

            # extract the iteration number from the path. The path is expected to be in the format
            # model_{iteration}.pt
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"

            export_policy_as_onnx(
                self.alg.actor,
                normalizer=getattr(self.alg.actor, "actor_obs_normalizer", self.alg.actor.obs_normalizer),
                path=onnx_folder,
                filename=onnx_model_name,
            )

            if self.logger_type in ["neptune", "wandb"]:
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(
        self,
        path,
        load_optimizer=True,
        weights_only=False,
        load_cfg=None,
        strict=True,
        map_location=None,
        **kwargs,
    ):
        if load_cfg is None:
            load_cfg = {}

        do_load_actor = load_cfg.get("actor", True)
        do_load_critic = load_cfg.get("critic", True)
        do_load_discriminator = load_cfg.get("discriminator", True)
        do_load_optimizer = load_cfg.get("optimizer", load_optimizer)

        loaded_dict = torch.load(
            path,
            map_location=(map_location if map_location is not None else self.device),
            weights_only=weights_only,
        )

        if do_load_actor and "actor_state_dict" in loaded_dict:
            self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if do_load_critic and "critic_state_dict" in loaded_dict:
            self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if do_load_discriminator and "discriminator_state_dict" in loaded_dict:
            discriminator_state = loaded_dict["discriminator_state_dict"]
            self.alg.discriminator.load_state_dict(discriminator_state, strict=False)

        amp_normalizer_module = loaded_dict.get("amp_normalizer")
        if do_load_discriminator and amp_normalizer_module is not None and getattr(
            self.alg.discriminator, "empirical_normalization", False
        ):
            # Old checkpoints stored the empirical normalizer separately; hydrate it if present.
            self.alg.discriminator.amp_normalizer.load_state_dict(
                amp_normalizer_module.state_dict()
            )
        if do_load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor.to(device)
        return lambda obs: self.alg.actor(obs, stochastic_output=False)

    def train_mode(self):
        self.alg.actor.train()
        self.alg.critic.train()
        self.alg.discriminator.train()

    def eval_mode(self):
        self.alg.actor.eval()
        self.alg.critic.eval()
        self.alg.discriminator.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
