# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the thin amp_rsl_rl.utils._compat re-export layer (rsl-rl v5.5.0).

Run with:
    python -m pytest tests/test_compat.py -v
"""


def test_empirical_normalization_importable():
    """EmpiricalNormalization is re-exported from rsl_rl.modules.normalization."""
    from amp_rsl_rl.utils._compat import EmpiricalNormalization

    assert isinstance(EmpiricalNormalization, type)


def test_resolve_obs_groups_importable():
    """resolve_obs_groups is re-exported from rsl_rl.utils."""
    from amp_rsl_rl.utils._compat import resolve_obs_groups

    assert callable(resolve_obs_groups)


def test_rsl_rl_version_is_v5_5_plus():
    """The installed rsl-rl-lib satisfies the >= 5.5.0 requirement."""
    from importlib.metadata import version

    parts = [int("".join(c for c in seg if c.isdigit()) or 0) for seg in version("rsl-rl-lib").split(".")[:2]]
    major, minor = parts[0], parts[1]
    assert (major, minor) >= (5, 5)
