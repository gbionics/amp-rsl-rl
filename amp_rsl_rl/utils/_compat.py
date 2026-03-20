# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

import rsl_rl


def _parse_ver(ver_str: str) -> tuple:
    parts = []
    for seg in str(ver_str).split(".")[:3]:
        digits = "".join(c for c in seg if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


try:
    _rsl_rl_version = version("rsl-rl-lib")
    print(f"Detected rsl-rl version: {_rsl_rl_version}")
except PackageNotFoundError:
    _rsl_rl_version = "3.0.0"
    print("rsl-rl not found, defaulting to version 3.0.0 for compatibility checks")

_VER: tuple = _parse_ver(_rsl_rl_version)
RSL_RL_MAJOR: int = _VER[0]
RSL_RL_V4_PLUS: bool = RSL_RL_MAJOR >= 4
RSL_RL_V5_PLUS: bool = RSL_RL_MAJOR >= 5

try:
    from rsl_rl.networks import EmpiricalNormalization  # noqa: F401
except ImportError:
    from rsl_rl.modules.normalization import EmpiricalNormalization  # noqa: F401

from rsl_rl.utils import resolve_obs_groups  # noqa: F401

if RSL_RL_V4_PLUS:
    from rsl_rl.utils.logger import Logger as _Logger

    def store_code_state(log_dir: str, repos: list) -> list:
        try:
            return _Logger._store_code_state(log_dir, repos)
        except TypeError:
            pass
        try:
            stub = _Logger.__new__(_Logger)
            stub.log_dir = log_dir
            stub.disable_logs = False
            stub.git_status_repos = repos
            return _Logger._store_code_state(stub)
        except Exception:
            return []

else:
    from rsl_rl.utils import store_code_state  # noqa: F401


__all__ = [
    "RSL_RL_MAJOR",
    "RSL_RL_V4_PLUS",
    "RSL_RL_V5_PLUS",
    "EmpiricalNormalization",
    "store_code_state",
    "resolve_obs_groups",
]
