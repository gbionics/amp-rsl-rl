# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
import warnings
from dataclasses import asdict

from torch.utils.tensorboard import SummaryWriter

try:
    import mlflow
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "MLflow is required to log to MLflow. Install it with: pip install mlflow"
    ) from None

from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_DIRTY,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_DIFF,
)

try:
    import git as _git
except ImportError:
    _git = None


class MLflowSummaryWriter(SummaryWriter):
    """Summary writer for MLflow tracking.

    This class is a drop-in replacement for :class:`WandbSummaryWriter`.
    It inherits from TensorBoard's ``SummaryWriter`` so that local TB logs
    are still produced, while simultaneously forwarding metrics, configs,
    artifacts and videos to an MLflow Tracking server.

    Configuration is read from ``cfg["mlflow_kwargs"]`` with the following keys:

    - ``"experiment_name"`` (required): MLflow experiment name.
    - ``"tracking_uri"`` (optional): MLflow tracking URI. Falls back to the
      ``MLFLOW_TRACKING_URI`` environment variable or ``"./mlruns"``.
    - ``"run_name"`` (optional): explicit run name; defaults to the last
      component of *log_dir*.
    - ``"tags"`` (optional): dict of tags forwarded to ``mlflow.start_run``.
    - ``"notes"`` (optional): description string stored as the run description.
    """

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        mlflow_kwargs: dict = cfg.get("mlflow_kwargs", {})

        # --- experiment name (required) ---
        experiment_name = mlflow_kwargs.get("experiment_name")
        if experiment_name is None:
            raise KeyError("Please specify 'experiment_name' in cfg['mlflow_kwargs'].")

        # --- tracking URI ---
        tracking_uri = mlflow_kwargs.get(
            "tracking_uri",
            os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"),
        )
        mlflow.set_tracking_uri(tracking_uri)

        # --- set experiment (must be done before searching runs) ---
        mlflow.set_experiment(experiment_name)

        # --- run name with auto-increment sequence ---
        # Use run_name (if provided) as prefix, otherwise fall back to
        # experiment_name. The prefix is appended with an incrementing number
        # (e.g., "qdd-amp1", "qdd-amp2", ...), mirroring wandb behaviour.
        prefix = mlflow_kwargs.get("run_name", experiment_name)
        run_name = self._next_sequential_run_name(prefix)

        # --- tags / description ---
        tags = mlflow_kwargs.get("tags", {})
        notes = mlflow_kwargs.get("notes", "")

        # --- inject git info into tags ---
        git_tags = self._collect_git_tags()
        # User-supplied tags take precedence over auto-detected ones
        merged_tags = {**git_tags, **tags}

        # --- start (or resume) a run ---
        self._run = mlflow.start_run(
            run_name=run_name,
            tags=merged_tags,
            description=notes,
        )
        print(f"MLflow run started: {run_name}")

        # Store log dir as a run param
        mlflow.log_param("log_dir", log_dir)

        # Replicate the name_map from WandbSummaryWriter for metric-key
        # sanitisation (MLflow metrics may not contain certain characters).
        self.name_map: dict[str, str] = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        # Keep track of already-logged video files (same pattern as wandb writer)
        self.video_files: list[str] = []

    # ------------------------------------------------------------------
    # Run naming helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _next_sequential_run_name(prefix: str) -> str:
        """Generate the next sequential run name for the active experiment.

        Searches existing runs whose name starts with *prefix*, extracts
        numeric suffixes, and returns ``prefix{max+1}``.
        """
        experiment = mlflow.get_experiment_by_name(prefix)
        if experiment is None:
            # First run ever in this experiment
            return f"{prefix}1"

        try:
            runs_df = mlflow.search_runs(
                [experiment.experiment_id],
                output_format="pandas",
            )
        except Exception:
            return f"{prefix}1"

        if runs_df.empty or "tags.mlflow.runName" not in runs_df.columns:
            return f"{prefix}1"

        max_num = 0
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        for name in runs_df["tags.mlflow.runName"].dropna():
            m = pattern.match(name)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num

        return f"{prefix}{max_num + 1}"

    # ------------------------------------------------------------------
    # Git info helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_git_tags() -> dict[str, str]:
        """Auto-detect the git repo from CWD and return MLflow-native git tags.

        Returns an empty dict if gitpython is not installed or no repo is found.
        """
        if _git is None:
            warnings.warn(
                "gitpython is not installed — git info will not be logged to MLflow. "
                "Install it with: pip install gitpython"
            )
            return {}

        try:
            repo = _git.Repo(os.getcwd(), search_parent_directories=True)
        except Exception:
            warnings.warn(
                "Could not find a git repository from the current working directory. "
                "Git info will not be logged to MLflow."
            )
            return {}

        git_tags: dict[str, str] = {}

        # Commit SHA
        try:
            git_tags[MLFLOW_GIT_COMMIT] = repo.head.commit.hexsha
        except Exception:
            pass

        # Branch name (may fail on detached HEAD)
        try:
            git_tags[MLFLOW_GIT_BRANCH] = repo.active_branch.name
        except TypeError:
            git_tags[MLFLOW_GIT_BRANCH] = "DETACHED_HEAD"

        # Dirty status
        try:
            git_tags[MLFLOW_GIT_DIRTY] = str(repo.is_dirty())
        except Exception:
            pass

        # Remote URL (origin)
        try:
            if repo.remotes:
                git_tags[MLFLOW_GIT_REPO_URL] = repo.remotes.origin.url
        except Exception:
            pass

        # Diff (truncated to 5000 chars — MLflow tag value limit)
        try:
            diff = repo.git.diff(repo.head.commit.tree)
            max_tag_len = 5000
            if len(diff) > max_tag_len:
                diff = diff[:max_tag_len] + "\n... [truncated]"
            git_tags[MLFLOW_GIT_DIFF] = diff
        except Exception:
            pass

        return git_tags

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def store_config(
        self,
        env_cfg: dict | object,
        runner_cfg: dict,
        alg_cfg: dict,
        policy_cfg: dict,
    ) -> None:
        """Persist configuration dicts as MLflow params."""

        def _flatten(d: dict, prefix: str = "") -> dict:
            """Flatten a nested dict into dot-separated keys."""
            items: dict = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    items.update(_flatten(v, key))
                else:
                    items[key] = v
            return items

        def _safe_log_params(params: dict) -> None:
            """Log params while truncating values that exceed MLflow's limit."""
            max_len = 500
            for k, v in params.items():
                str_v = str(v)
                if len(str_v) > max_len:
                    str_v = str_v[:max_len] + "..."
                try:
                    mlflow.log_param(k, str_v)
                except mlflow.exceptions.MlflowException:
                    pass  # param already logged – ignore

        _safe_log_params(_flatten(runner_cfg, "runner_cfg"))
        _safe_log_params(_flatten(policy_cfg, "policy_cfg"))
        _safe_log_params(_flatten(alg_cfg, "alg_cfg"))

        try:
            env_dict = (
                env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
            )
            _safe_log_params(_flatten(env_dict, "env_cfg"))
        except Exception:
            warnings.warn("Could not log env_cfg to MLflow params.")

    def log_config(
        self,
        env_cfg: dict | object,
        runner_cfg: dict,
        alg_cfg: dict,
        policy_cfg: dict,
    ) -> None:
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------
    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        # Forward to TensorBoard
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        # Sanitise metric name for MLflow
        metric_name = self.name_map.get(tag, tag)
        mlflow.log_metric(metric_name, scalar_value, step=global_step)

    # ------------------------------------------------------------------
    # Video logging
    # ------------------------------------------------------------------
    def add_video_files(self, log_dir: str, step: int) -> None:
        """Log new ``.mp4`` video files found under *log_dir* as MLflow artifacts."""
        if not os.path.exists(log_dir):
            return

        for root, _dirs, files in os.walk(log_dir):
            for video_file in files:
                if video_file.endswith(".mp4") and video_file not in self.video_files:
                    self.video_files.append(video_file)
                    video_path = os.path.join(root, video_file)
                    mlflow.log_artifact(video_path, artifact_path="videos")

    # ------------------------------------------------------------------
    # Model / file saving
    # ------------------------------------------------------------------
    def save_model(self, model_path: str, iter: int) -> None:
        """Log a model checkpoint as an MLflow artifact."""
        mlflow.log_artifact(model_path, artifact_path="models")

    def save_file(self, path: str) -> None:
        """Log an arbitrary file as an MLflow artifact."""
        mlflow.log_artifact(path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
