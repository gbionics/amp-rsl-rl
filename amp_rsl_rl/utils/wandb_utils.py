# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import warnings
from dataclasses import asdict

import wandb
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.utils.log_writer import LogWriter


class WandbSummaryWriter(SummaryWriter, LogWriter):
    """W&B summary writer compatible with the rsl-rl v5.5.0 ``LogWriter`` API.

    Mirrors scalars to both TensorBoard and Weights & Biases and forwards
    checkpoints, files and videos as artifacts. Project/entity/group/notes are
    read from ``cfg["wandb_kwargs"]`` (falling back to ``cfg["wandb_project"]``).
    """

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        SummaryWriter.__init__(self, log_dir, flush_secs)

        run_name = os.path.split(log_dir)[-1]

        wandb_kwargs: dict = cfg.get("wandb_kwargs", {})
        project = wandb_kwargs.get("project") or cfg.get("wandb_project")
        if project is None:
            raise KeyError(
                "Please specify 'wandb_project' or 'wandb_kwargs.project' in the runner config."
            )

        entity = wandb_kwargs.get("entity")
        if entity is None:
            warnings.warn("wandb entity not specified in the runner config.")
        group = wandb_kwargs.get("group")
        notes = wandb_kwargs.get("notes")

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            notes=notes,
        )
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }
        self.video_files: list[str] = []
        self.logged_videos: set[str] = set()

        self.update_run_name_with_sequence(prefix=project)

    # -------------------------------------------------------------- config

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload the training and environment configuration to W&B."""
        try:
            wandb.config.update({"train_cfg": train_cfg}, allow_val_change=True)
        except Exception:
            pass
        try:
            env_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
            wandb.config.update({"env_cfg": env_dict}, allow_val_change=True)
        except Exception:
            pass

    # -------------------------------------------------------------- scalars

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style
        )
        wandb.log({self.name_map.get(tag, tag): scalar_value}, step=global_step)

    # -------------------------------------------------------------- videos

    def add_video_files(self, log_dir: str, step: int) -> None:
        """Log new ``.mp4`` files found under *log_dir* to W&B."""
        if not os.path.exists(log_dir):
            return
        for root, _dirs, files in os.walk(log_dir):
            for video_file in files:
                if video_file.endswith(".mp4") and video_file not in self.video_files:
                    self.video_files.append(video_file)
                    video_path = os.path.join(root, video_file)
                    wandb.log({"Video": wandb.Video(video_path, format="mp4")}, step=step)

    def save_video(self, video, it: int) -> None:
        """Upload a single video artifact once per filename (LogWriter API)."""
        name = os.path.basename(str(video))
        if name not in self.logged_videos:
            wandb.log({"Video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(name)

    # -------------------------------------------------------------- files

    def save_model(self, model_path: str, it: int) -> None:
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))

    def stop(self) -> None:
        wandb.finish()

    # -------------------------------------------------------------- naming

    def update_run_name_with_sequence(self, prefix: str) -> None:
        """Rename the run to ``{prefix}{n}`` with an auto-incrementing suffix."""
        project = wandb.run.project
        entity = wandb.run.entity
        try:
            api = wandb.Api()
            runs = api.runs(f"{entity}/{project}")
        except Exception:
            return

        max_num = 0
        for run in runs:
            if run.name.startswith(prefix):
                numeric_suffix = run.name[len(prefix) :]
                try:
                    run_num = int(numeric_suffix)
                    max_num = max(max_num, run_num)
                except ValueError:
                    continue

        wandb.run.name = f"{prefix}{max_num + 1}"
        print("Updated run name to:", wandb.run.name)
