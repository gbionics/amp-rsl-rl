# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`MLflowSummaryWriter`.

Run with::

    pytest tests/test_mlflow_summary_writer.py -v
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# We mock *mlflow* so that tests can run without an actual tracking server.
# ---------------------------------------------------------------------------

# Create a fake mlflow module with the minimum surface needed by the writer.
_fake_mlflow = MagicMock()
_fake_mlflow.exceptions = MagicMock()
_fake_mlflow.exceptions.MlflowException = Exception  # used in except clauses


@pytest.fixture(autouse=True)
def _patch_mlflow(monkeypatch):
    """Ensure all tests use a mocked mlflow so no server is required."""
    # Build a fake mlflow.utils.mlflow_tags module with the tag-name constants.
    # This is required because mlflow_utils.py has a module-level
    # `from mlflow.utils.mlflow_tags import ...` that runs on first import
    # while sys.modules["mlflow"] is already a MagicMock (not a real package).
    _fake_tags = ModuleType("mlflow.utils.mlflow_tags")
    _fake_tags.MLFLOW_GIT_BRANCH = "mlflow.gitBranchName"
    _fake_tags.MLFLOW_GIT_COMMIT = "mlflow.gitCommit"
    _fake_tags.MLFLOW_GIT_DIRTY = "mlflow.gitDirty"
    _fake_tags.MLFLOW_GIT_REPO_URL = "mlflow.gitRepoURL"
    _fake_tags.MLFLOW_GIT_DIFF = "mlflow.gitDiff"

    _fake_utils = ModuleType("mlflow.utils")

    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", _fake_mlflow.exceptions)
    monkeypatch.setitem(sys.modules, "mlflow.utils", _fake_utils)
    monkeypatch.setitem(sys.modules, "mlflow.utils.mlflow_tags", _fake_tags)
    # Setting git to None causes `import git` to raise ImportError, so
    # mlflow_utils sets _git = None and _collect_git_tags() returns {}.
    monkeypatch.setitem(sys.modules, "git", None)

    # Also evict the cached mlflow_utils module so the patched sys.modules
    # is used when it is re-imported inside each test.
    monkeypatch.delitem(sys.modules, "amp_rsl_rl.utils.mlflow_utils", raising=False)

    # Reset call tracking between tests
    _fake_mlflow.reset_mock()
    yield


@pytest.fixture()
def tmp_log_dir(tmp_path: Path):
    """Provide a temporary log directory that is cleaned up afterwards."""
    log_dir = tmp_path / "mlflow_test_logs"
    log_dir.mkdir()
    yield str(log_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cfg(extra: dict | None = None) -> dict:
    """Return a minimal valid cfg dict."""
    cfg: dict = {
        "mlflow_kwargs": {
            "experiment_name": "test_experiment",
            "tracking_uri": "./test_mlruns",
            "run_name": "test_run",
            "tags": {"env": "unit_test"},
            "notes": "Automated test run",
        },
    }
    if extra:
        cfg["mlflow_kwargs"].update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMLflowSummaryWriterInit:
    """Constructor / configuration tests."""

    def test_basic_init(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        cfg = _default_cfg()
        writer = MLflowSummaryWriter(log_dir=tmp_log_dir, flush_secs=10, cfg=cfg)

        # MLflow calls
        _fake_mlflow.set_tracking_uri.assert_called_once_with("./test_mlruns")
        _fake_mlflow.set_experiment.assert_called_once_with("test_experiment")
        _fake_mlflow.start_run.assert_called_once_with(
            run_name="test_run1",
            tags={"env": "unit_test"},
            description="Automated test run",
        )
        _fake_mlflow.log_param.assert_any_call("log_dir", tmp_log_dir)

        assert writer.video_files == []
        assert isinstance(writer.name_map, dict)

    def test_missing_experiment_name_raises(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        cfg: dict = {"mlflow_kwargs": {}}  # no experiment_name
        with pytest.raises(KeyError, match="experiment_name"):
            MLflowSummaryWriter(log_dir=tmp_log_dir, flush_secs=10, cfg=cfg)

    def test_defaults_from_env(self, tmp_log_dir: str, monkeypatch):
        """When tracking_uri is absent, falls back to env var."""
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://my-server:5000")
        cfg: dict = {
            "mlflow_kwargs": {
                "experiment_name": "exp",
            }
        }
        MLflowSummaryWriter(log_dir=tmp_log_dir, flush_secs=10, cfg=cfg)
        _fake_mlflow.set_tracking_uri.assert_called_once_with("http://my-server:5000")

    def test_run_name_defaults_to_log_dir_basename(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        cfg: dict = {
            "mlflow_kwargs": {
                "experiment_name": "exp",
            }
        }
        MLflowSummaryWriter(log_dir=tmp_log_dir, flush_secs=10, cfg=cfg)
        expected_prefix = os.path.split(tmp_log_dir)[-1]
        _fake_mlflow.start_run.assert_called_once()
        call_kwargs = _fake_mlflow.start_run.call_args
        assert call_kwargs.kwargs["run_name"].startswith(expected_prefix)


class TestAddScalar:
    def test_add_scalar_forwards_to_mlflow(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        writer.add_scalar("Loss/value_function", 0.42, global_step=10)

        _fake_mlflow.log_metric.assert_called_with("Loss/value_function", 0.42, step=10)

    def test_name_map_is_applied(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        writer.add_scalar("Train/mean_reward/time", 1.5, global_step=5)

        _fake_mlflow.log_metric.assert_called_with(
            "Train/mean_reward_time", 1.5, step=5
        )


class TestVideoLogging:
    def test_add_video_files_logs_artifacts(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        # Create a fake video file
        video_path = os.path.join(tmp_log_dir, "episode_0.mp4")
        Path(video_path).touch()

        writer.add_video_files(tmp_log_dir, step=100)

        _fake_mlflow.log_artifact.assert_called_once_with(
            video_path, artifact_path="videos"
        )
        assert "episode_0.mp4" in writer.video_files

    def test_duplicate_videos_are_skipped(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        video_path = os.path.join(tmp_log_dir, "episode_0.mp4")
        Path(video_path).touch()

        writer.add_video_files(tmp_log_dir, step=100)
        writer.add_video_files(tmp_log_dir, step=200)

        # Should only be logged once
        assert _fake_mlflow.log_artifact.call_count == 1

    def test_non_mp4_files_are_ignored(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        Path(os.path.join(tmp_log_dir, "data.csv")).touch()
        Path(os.path.join(tmp_log_dir, "notes.txt")).touch()

        writer.add_video_files(tmp_log_dir, step=1)

        _fake_mlflow.log_artifact.assert_not_called()

    def test_nonexistent_dir_is_noop(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        writer.add_video_files("/does/not/exist", step=1)

        _fake_mlflow.log_artifact.assert_not_called()


class TestSaveModelAndFile:
    def test_save_model(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )
        model_path = os.path.join(tmp_log_dir, "model_0.pt")
        Path(model_path).touch()

        writer.save_model(model_path, iter=0)

        _fake_mlflow.log_artifact.assert_called_once_with(
            model_path, artifact_path="models"
        )

    def test_save_file(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )
        file_path = os.path.join(tmp_log_dir, "git_diff.txt")
        Path(file_path).touch()

        writer.save_file(file_path)

        _fake_mlflow.log_artifact.assert_called_once_with(file_path)


class TestStoreConfig:
    def test_store_config_logs_params(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        runner_cfg = {"lr": 0.001, "nested": {"a": 1}}
        policy_cfg = {"hidden": 256}
        alg_cfg = {"gamma": 0.99}

        # env_cfg as a plain dict (fallback path via asdict will fail, but
        # the writer should still swallow it gracefully via the except branch)
        env_cfg = {"num_envs": 4096}

        writer.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

        # Check that flat keys were logged
        logged_keys = {call.args[0] for call in _fake_mlflow.log_param.call_args_list}
        assert "runner_cfg.lr" in logged_keys
        assert "runner_cfg.nested.a" in logged_keys
        assert "policy_cfg.hidden" in logged_keys
        assert "alg_cfg.gamma" in logged_keys


class TestStop:
    def test_stop_ends_run(self, tmp_log_dir: str):
        from amp_rsl_rl.utils.mlflow_utils import MLflowSummaryWriter

        writer = MLflowSummaryWriter(
            log_dir=tmp_log_dir, flush_secs=10, cfg=_default_cfg()
        )

        writer.stop()

        _fake_mlflow.end_run.assert_called_once()
