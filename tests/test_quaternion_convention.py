# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the configurable quaternion serialization convention in AMPLoader.

Run with:
    python -m pytest tests/test_quaternion_convention.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from amp_rsl_rl.utils.motion_loader import AMPLoader, QuaternionConvention


def _make_fake_dataset(num_frames=50, num_joints=6, seed=0):
    """Create a fake motion dataset dict matching AMPLoader's expected .npy format.

    The root quaternion is stored in `xyzw` order, as required by AMPLoader.
    """
    rng = np.random.default_rng(seed)
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    joint_positions = [
        rng.standard_normal(num_joints).astype(np.float32) for _ in range(num_frames)
    ]
    root_position = [
        rng.standard_normal(3).astype(np.float32) for _ in range(num_frames)
    ]
    root_quaternion = [
        Rotation.random(random_state=rng).as_quat().astype(np.float32)
        for _ in range(num_frames)
    ]
    return {
        "joints_list": joint_names,
        "joint_positions": joint_positions,
        "root_position": root_position,
        "root_quaternion": root_quaternion,
        "fps": 30.0,
    }


@pytest.fixture
def fake_dataset_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(Path(tmpdir) / "walk.npy", _make_fake_dataset(50, 6))
        yield Path(tmpdir)


def _base_quat_from_loader(loader: AMPLoader) -> np.ndarray:
    """Extracts the base quaternion columns (first 4) from the reset-state buffer."""
    return loader.motion_data[0].base_quat.cpu().numpy()


class TestQuaternionConvention:
    def test_default_is_wxyz(self, fake_dataset_dir):
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        assert loader.quaternion_convention == QuaternionConvention.WXYZ

    def test_wxyz_vs_xyzw_are_consistent_reorderings(self, fake_dataset_dir):
        loader_wxyz = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            quaternion_convention=QuaternionConvention.WXYZ,
        )
        loader_xyzw = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            quaternion_convention=QuaternionConvention.XYZW,
        )

        quat_wxyz = _base_quat_from_loader(loader_wxyz)
        quat_xyzw = _base_quat_from_loader(loader_xyzw)

        assert quat_wxyz.shape == quat_xyzw.shape
        assert quat_wxyz.shape[1] == 4

        # wxyz -> xyzw reordering: [w, x, y, z] -> [x, y, z, w]
        reordered = quat_wxyz[:, [1, 2, 3, 0]]
        np.testing.assert_allclose(reordered, quat_xyzw, atol=1e-5)

    def test_reset_state_quat_columns_match_convention(self, fake_dataset_dir):
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            quaternion_convention=QuaternionConvention.XYZW,
        )
        quat, _, _, _, _ = loader.get_state_for_reset(number_of_samples=5)
        assert quat.shape == (5, 4)
