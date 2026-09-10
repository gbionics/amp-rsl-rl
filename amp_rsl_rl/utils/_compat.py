# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin re-export layer for rsl-rl-lib v5.5.0.

Historically this module papered over API differences across rsl-rl-lib v3-v5.
The package now targets rsl-rl-lib >= 5.5.0 natively, so this module only
re-exports a handful of symbols from stable v5.5.0 locations. It is kept because
downstream projects (e.g. gb-rl-locomotion distillation scripts) import
``resolve_obs_groups`` from here.
"""

from __future__ import annotations

from rsl_rl.modules.normalization import EmpiricalNormalization  # noqa: F401
from rsl_rl.utils import resolve_obs_groups  # noqa: F401

__all__ = [
    "EmpiricalNormalization",
    "resolve_obs_groups",
]
