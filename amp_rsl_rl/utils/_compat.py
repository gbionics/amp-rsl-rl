# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility shim for rsl-rl-lib v3.0 through v5.x.

This module detects the installed version of ``rsl-rl-lib`` and re-exports
symbols whose locations or signatures changed across major versions:

+---------------------------+--------------------+--------------------+
| Symbol                    | v3 location        | v4+/v5+ location   |
+===========================+====================+====================+
| EmpiricalNormalization    | rsl_rl.networks    | rsl_rl.modules.    |
|                           |                    | normalization      |
+---------------------------+--------------------+--------------------+
| store_code_state          | rsl_rl.utils       | Logger method only |
|                           | (standalone, <=3.1)|   (>=3.2)          |
+---------------------------+--------------------+--------------------+
| resolve_obs_groups        | rsl_rl.utils (all) | rsl_rl.utils (all) |
+---------------------------+--------------------+--------------------+

Version flags exposed:
- ``RSL_RL_VERSION``: full version tuple (major, minor, patch).
- ``RSL_RL_MAJOR``: integer major version.
- ``RSL_RL_V3_3_PLUS``: True if major == 3 and minor >= 3.
- ``RSL_RL_V4_PLUS``: True if major >= 4.
- ``RSL_RL_V5_PLUS``: True if major >= 5.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from importlib.metadata import version, PackageNotFoundError
from typing import List

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def _parse_version(ver_str: str) -> tuple[int, int, int]:
    """Parse a PEP-440 version string into a (major, minor, patch) tuple."""
    parts: list[int] = []
    for segment in str(ver_str).split(".")[:3]:
        digits = "".join(c for c in segment if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


try:
    _rsl_rl_version_str: str = version("rsl-rl-lib")
except PackageNotFoundError:
    _rsl_rl_version_str = "3.0.0"
    warnings.warn(
        "Could not detect rsl-rl-lib version (package not found). "
        "Assuming v3.0.0 for compatibility.",
        stacklevel=2,
    )

RSL_RL_VERSION: tuple[int, int, int] = _parse_version(_rsl_rl_version_str)
RSL_RL_MAJOR: int = RSL_RL_VERSION[0]
RSL_RL_MINOR: int = RSL_RL_VERSION[1]
RSL_RL_V3_3_PLUS: bool = not (
    RSL_RL_MAJOR < 3 or (RSL_RL_MAJOR == 3 and RSL_RL_MINOR < 3)
)
RSL_RL_V4_PLUS: bool = RSL_RL_MAJOR >= 4
RSL_RL_V5_PLUS: bool = RSL_RL_MAJOR >= 5


# ---------------------------------------------------------------------------
# EmpiricalNormalization
# ---------------------------------------------------------------------------
# v3: rsl_rl.networks.EmpiricalNormalization
# v4+: rsl_rl.modules.normalization.EmpiricalNormalization

try:
    from rsl_rl.networks import EmpiricalNormalization  # noqa: F401
except (ImportError, ModuleNotFoundError):
    from rsl_rl.modules.normalization import EmpiricalNormalization  # noqa: F401


# ---------------------------------------------------------------------------
# resolve_obs_groups  (available in all versions under rsl_rl.utils)
# ---------------------------------------------------------------------------

from rsl_rl.utils import resolve_obs_groups  # noqa: F401

# ---------------------------------------------------------------------------
# store_code_state
# ---------------------------------------------------------------------------
# - v3.0.x / v3.1.x: standalone function in rsl_rl.utils with signature
#       store_code_state(logdir: str, repositories: list[str]) -> list[str]
# - v3.2+ / v4+ / v5+: removed from rsl_rl.utils; functionality moved to
#       Logger._store_code_state(self) -> list[str] | None
#
# We provide a unified function with signature:
#       store_code_state(log_dir: str, repos: list[str]) -> list[str]


def _store_code_state_via_logger(log_dir: str, repos: List[str]) -> List[str]:
    """Reimplement store_code_state using the same logic as Logger._store_code_state.

    Rather than constructing a fragile Logger stub, we replicate the straightforward
    git-diff logic that all versions use internally. This avoids depending on Logger's
    internal state or constructor signature.
    """
    try:
        import git  # GitPython — required dependency of rsl-rl-lib
    except ImportError:
        warnings.warn(
            "GitPython not installed; cannot store code state.",
            stacklevel=3,
        )
        return []

    git_log_dir = os.path.join(log_dir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths: List[str] = []

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

        # Don't overwrite existing diff files
        if os.path.isfile(diff_file_name):
            continue

        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = (
                f"--- git commit ---\n{commit_hash}\n\n\n"
                f"--- git status ---\n{repo.git.status()} \n\n\n"
                f"--- git diff ---\n{repo.git.diff(tree)}"
            )
            f.write(content)
        file_paths.append(diff_file_name)

    return file_paths


# Try the standalone function first (v3.0.x / v3.1.x only)
try:
    from rsl_rl.utils import store_code_state  # noqa: F401
except ImportError:
    # v3.2+ / v4+ / v5+: standalone function was removed
    store_code_state = _store_code_state_via_logger


# ---------------------------------------------------------------------------
# Wandb summary writer base class
# ---------------------------------------------------------------------------
# - v3 / v4 / v5.0-v5.3: rsl_rl.utils.wandb_utils.WandbSummaryWriter
#       (provides a ``log_config`` method)
# - v5.4+: logging refactor (PR #211) renamed it to
#       rsl_rl.utils.wandb_log_writer.WandbLogWriter and dropped ``log_config``
#
# Re-exported as ``WandbSummaryWriterBase`` so downstream writers can subclass a
# single stable name. ``RSL_RL_WANDB_HAS_LOG_CONFIG`` records whether the base
# class still provides the legacy ``log_config`` method (removed in v5.4).

try:
    from rsl_rl.utils.wandb_log_writer import (  # noqa: F401
        WandbLogWriter as WandbSummaryWriterBase,
    )
except (ImportError, ModuleNotFoundError):
    from rsl_rl.utils.wandb_utils import (  # noqa: F401
        WandbSummaryWriter as WandbSummaryWriterBase,
    )

RSL_RL_WANDB_HAS_LOG_CONFIG: bool = hasattr(WandbSummaryWriterBase, "log_config")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "RSL_RL_VERSION",
    "RSL_RL_MAJOR",
    "RSL_RL_MINOR",
    "RSL_RL_V3_3_PLUS",
    "RSL_RL_V4_PLUS",
    "RSL_RL_V5_PLUS",
    "EmpiricalNormalization",
    "store_code_state",
    "resolve_obs_groups",
    "WandbSummaryWriterBase",
    "RSL_RL_WANDB_HAS_LOG_CONFIG",
]
