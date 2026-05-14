# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for amp_rsl_rl.utils._compat module.

Run with:
    python -m pytest tests/test_compat.py -v
"""

import os
import sys
import tempfile

import pytest


def test_version_flags():
    """Version flags are correctly set based on installed rsl-rl-lib."""
    from importlib.metadata import version

    from amp_rsl_rl.utils._compat import (
        RSL_RL_MAJOR,
        RSL_RL_V4_PLUS,
        RSL_RL_V5_PLUS,
        RSL_RL_VERSION,
    )

    installed_version = version("rsl-rl-lib")
    major = int(installed_version.split(".")[0])

    assert RSL_RL_MAJOR == major
    assert RSL_RL_VERSION[0] == major
    assert RSL_RL_V4_PLUS == (major >= 4)
    assert RSL_RL_V5_PLUS == (major >= 5)


def test_empirical_normalization_importable():
    """EmpiricalNormalization can be imported regardless of version."""
    from amp_rsl_rl.utils._compat import EmpiricalNormalization

    assert EmpiricalNormalization is not None
    # It should be a class
    assert isinstance(EmpiricalNormalization, type)


def test_resolve_obs_groups_importable():
    """resolve_obs_groups can be imported regardless of version."""
    from amp_rsl_rl.utils._compat import resolve_obs_groups

    assert callable(resolve_obs_groups)


def test_store_code_state_importable():
    """store_code_state can be imported and is callable."""
    from amp_rsl_rl.utils._compat import store_code_state

    assert callable(store_code_state)


def test_store_code_state_on_git_repo():
    """store_code_state produces a .diff file when pointed at a git repo."""
    from amp_rsl_rl.utils._compat import store_code_state

    # Use this repository itself as the target
    repo_file = os.path.abspath(__file__)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = store_code_state(tmpdir, [repo_file])
        assert isinstance(result, list)
        # Should have produced at least one file
        assert len(result) >= 1
        for path in result:
            assert os.path.isfile(path)
            assert path.endswith(".diff")
            # Check content has expected sections
            with open(path) as f:
                content = f.read()
            assert "git status" in content
            assert "git diff" in content


def test_store_code_state_non_git_dir():
    """store_code_state gracefully handles non-git directories."""
    from amp_rsl_rl.utils._compat import store_code_state

    with tempfile.TemporaryDirectory() as tmpdir:
        non_git_path = os.path.join(tmpdir, "not_a_repo.py")
        with open(non_git_path, "w") as f:
            f.write("# dummy")
        result = store_code_state(tmpdir, [non_git_path])
        assert isinstance(result, list)
        assert len(result) == 0


def test_store_code_state_idempotent():
    """store_code_state does not overwrite existing .diff files."""
    from amp_rsl_rl.utils._compat import store_code_state

    repo_file = os.path.abspath(__file__)

    with tempfile.TemporaryDirectory() as tmpdir:
        result1 = store_code_state(tmpdir, [repo_file])
        assert len(result1) >= 1
        # Read content of first file
        with open(result1[0]) as f:
            original_content = f.read()
        # Call again — should skip (file exists)
        result2 = store_code_state(tmpdir, [repo_file])
        assert len(result2) == 0
        # Original file unchanged
        with open(result1[0]) as f:
            assert f.read() == original_content


def test_all_exports():
    """__all__ matches the actual exports."""
    import amp_rsl_rl.utils._compat as compat

    for name in compat.__all__:
        assert hasattr(compat, name), f"Missing export: {name}"
