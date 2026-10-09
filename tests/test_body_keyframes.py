# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for body keyframe observations in AMPLoader.

Run with:
    python -m pytest tests/test_body_keyframes.py -v
"""

import torch
import pytest
import numpy as np
import tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation

from amp_rsl_rl.utils.motion_loader import AMPLoader, MotionData, VelocityRepresentation

# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_synthetic_dataset(
    tmpdir: Path,
    name: str = "test_motion",
    n_frames: int = 100,
    fps: float = 50.0,
    n_joints: int = 6,
    joint_names: list = None,
    include_body_links: bool = True,
    body_links_list: list = None,
) -> Path:
    """Create a synthetic .npy motion dataset for testing."""
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(n_joints)]
    if body_links_list is None:
        body_links_list = [
            "root_link",
            "left_knee",
            "right_knee",
            "left_foot",
            "right_foot",
        ]

    n_bodies = len(body_links_list)

    # Generate smooth trajectories
    t = np.linspace(0, 2 * np.pi, n_frames)

    # Joint positions: sinusoidal motion
    joint_positions = np.zeros((n_frames, n_joints))
    for i in range(n_joints):
        joint_positions[:, i] = 0.5 * np.sin(t + i * np.pi / n_joints)

    # Root position: walking forward
    root_position = np.zeros((n_frames, 3))
    root_position[:, 0] = np.linspace(0, 2, n_frames)  # x: forward
    root_position[:, 2] = 0.9  # z: height

    # Root quaternion: slight rotation (xyzw format)
    root_quaternion = np.zeros((n_frames, 4))
    for i in range(n_frames):
        r = Rotation.from_euler("z", t[i] * 0.1)
        root_quaternion[i] = r.as_quat()  # xyzw

    data = {
        "joints_list": joint_names,
        "joint_positions": joint_positions,
        "root_position": root_position,
        "root_quaternion": root_quaternion,
        "fps": np.array([fps]),
    }

    if include_body_links:
        # Generate body link positions/quaternions in world frame
        body_links_pos_w = np.zeros((n_frames, n_bodies, 3))
        body_links_quat_w = np.zeros((n_frames, n_bodies, 4))

        for b in range(n_bodies):
            # Each body moves relative to root with some offset
            offset = np.array([0.0, 0.1 * (b - n_bodies // 2), -0.1 * b])
            for i in range(n_frames):
                R_root = Rotation.from_quat(root_quaternion[i])
                body_links_pos_w[i, b, :] = root_position[i] + R_root.apply(offset)
                # Body orientation: root orientation + small relative rotation
                r_rel = Rotation.from_euler("y", 0.1 * np.sin(t[i] + b))
                body_links_quat_w[i, b, :] = (R_root * r_rel).as_quat()  # xyzw

        data["body_links_list"] = body_links_list
        data["body_links_pos_w"] = body_links_pos_w
        data["body_links_quat_w"] = body_links_quat_w

    filepath = tmpdir / f"{name}.npy"
    np.save(str(filepath), data)
    return filepath


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def dataset_with_bodies(tmpdir):
    """Create a dataset with body keyframes."""
    _create_synthetic_dataset(
        tmpdir,
        name="motion_with_bodies",
        include_body_links=True,
    )
    return tmpdir


@pytest.fixture
def dataset_without_bodies(tmpdir):
    """Create a dataset without body keyframes."""
    _create_synthetic_dataset(
        tmpdir,
        name="motion_no_bodies",
        include_body_links=False,
    )
    return tmpdir


# ── Tests: Backward Compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
    """Verify that existing behavior is unchanged when amp_obs_components is None."""

    def test_default_obs_components(self, dataset_with_bodies):
        """AMPLoader without amp_obs_components should behave as before."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        # Default obs: joint_pos (6) + joint_vel (6) + base_lin_vel (3) + base_ang_vel (3) = 18
        assert loader.all_obs.shape[1] == 18

    def test_explicit_default_components(self, dataset_with_bodies):
        """Explicitly passing default components should give same result."""
        loader_default = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_explicit = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=[
                "joint_pos",
                "joint_vel",
                "base_lin_vel",
                "base_ang_vel",
            ],
        )
        assert torch.allclose(loader_default.all_obs, loader_explicit.all_obs)
        assert torch.allclose(loader_default.all_next_obs, loader_explicit.all_next_obs)

    def test_dataset_without_bodies_default(self, dataset_without_bodies):
        """Dataset without body links should work fine with default obs."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_without_bodies,
            datasets={"motion_no_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        assert loader.all_obs.shape[1] == 18

    def test_reset_states_unchanged(self, dataset_with_bodies):
        """Reset states should not be affected by body keyframes."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=[
                "joint_pos",
                "joint_vel",
                "base_lin_vel",
                "base_ang_vel",
                "body_pos_b",
            ],
            body_links_names=[
                "root_link",
                "left_knee",
                "right_knee",
                "left_foot",
                "right_foot",
            ],
            anchor_body_name="root_link",
        )
        # Reset state: quat(4) + joint_pos(6) + joint_vel(6) + base_lin_vel(3) + base_ang_vel(3) = 22
        assert loader.all_states.shape[1] == 22


# ── Tests: Body Keyframe Observations ───────────────────────────────────────


class TestBodyKeyframeObs:
    """Verify body keyframe observation loading and dimensions."""

    def test_body_pos_b_dimensions(self, dataset_with_bodies):
        """body_pos_b should have (N_bodies-1)*3 dimensions (anchor excluded)."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_pos_b"],
            body_links_names=[
                "root_link",
                "left_knee",
                "right_knee",
                "left_foot",
                "right_foot",
            ],
            anchor_body_name="root_link",
        )
        # 5 bodies - 1 anchor = 4 non-anchor bodies * 3 = 12
        assert loader.all_obs.shape[1] == 12

    def test_body_ori_b_dimensions(self, dataset_with_bodies):
        """body_ori_b should have (N_bodies-1)*6 dimensions (6D rotation repr)."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_ori_b"],
            body_links_names=[
                "root_link",
                "left_knee",
                "right_knee",
                "left_foot",
                "right_foot",
            ],
            anchor_body_name="root_link",
        )
        # 4 non-anchor bodies * 6 = 24
        assert loader.all_obs.shape[1] == 24

    def test_body_lin_vel_b_dimensions(self, dataset_with_bodies):
        """body_lin_vel_b should have (N_bodies-1)*3 dimensions."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_lin_vel_b"],
            body_links_names=[
                "root_link",
                "left_knee",
                "right_knee",
                "left_foot",
                "right_foot",
            ],
            anchor_body_name="root_link",
        )
        # 4 non-anchor bodies * 3 = 12
        assert loader.all_obs.shape[1] == 12

    def test_body_ang_vel_b_dimensions(self, dataset_with_bodies):
        """body_ang_vel_b should have (N_bodies-1)*3 dimensions."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_ang_vel_b"],
            body_links_names=[
                "root_link",
                "left_knee",
                "right_knee",
                "left_foot",
                "right_foot",
            ],
            anchor_body_name="root_link",
        )
        # 4 non-anchor bodies * 3 = 12
        assert loader.all_obs.shape[1] == 12

    def test_combined_components(self, dataset_with_bodies):
        """All components together should sum dimensions correctly."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=[
                "joint_pos",
                "joint_vel",
                "base_lin_vel",
                "base_ang_vel",
                "body_pos_b",
                "body_ori_b",
            ],
            body_links_names=[
                "root_link",
                "left_knee",
                "right_knee",
                "left_foot",
                "right_foot",
            ],
            anchor_body_name="root_link",
        )
        # joint_pos(6) + joint_vel(6) + base_lin_vel(3) + base_ang_vel(3)
        # + body_pos_b(4*3=12) + body_ori_b(4*6=24) = 54
        assert loader.all_obs.shape[1] == 54

    def test_subset_of_bodies(self, dataset_with_bodies):
        """Only a subset of body links can be specified."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_pos_b"],
            body_links_names=["root_link", "left_knee", "right_knee"],
            anchor_body_name="root_link",
        )
        # 3 bodies - 1 anchor = 2 non-anchor * 3 = 6
        assert loader.all_obs.shape[1] == 6

    def test_obs_and_next_obs_shapes_match(self, dataset_with_bodies):
        """obs and next_obs should have same shape."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["joint_pos", "body_pos_b", "body_ori_b"],
            body_links_names=["root_link", "left_knee", "right_knee"],
            anchor_body_name="root_link",
        )
        assert loader.all_obs.shape == loader.all_next_obs.shape

    def test_feed_forward_generator(self, dataset_with_bodies):
        """feed_forward_generator should yield batches of correct shape."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["joint_pos", "body_pos_b"],
            body_links_names=["root_link", "left_knee", "right_knee"],
            anchor_body_name="root_link",
        )
        # joint_pos(6) + body_pos_b(2*3=6) = 12
        batch_size = 32
        for obs, next_obs in loader.feed_forward_generator(2, batch_size):
            assert obs.shape == (batch_size, 12)
            assert next_obs.shape == (batch_size, 12)


# ── Tests: Validation & Error Handling ───────────────────────────────────────


class TestValidation:
    """Test error cases and input validation."""

    def test_missing_body_links_names(self, dataset_with_bodies):
        """Should raise ValueError when body components requested without body_links_names."""
        with pytest.raises(ValueError, match="body_links_names must be provided"):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_with_bodies,
                datasets={"motion_with_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["body_pos_b"],
            )

    def test_missing_anchor_body_name(self, dataset_with_bodies):
        """Should raise ValueError when body components requested without anchor_body_name."""
        with pytest.raises(ValueError, match="anchor_body_name must be provided"):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_with_bodies,
                datasets={"motion_with_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["body_pos_b"],
                body_links_names=["root_link", "left_knee"],
            )

    def test_anchor_not_in_body_links(self, dataset_with_bodies):
        """Should raise ValueError if anchor_body_name not in body_links_names."""
        with pytest.raises(ValueError, match="must be present in body_links_names"):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_with_bodies,
                datasets={"motion_with_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["body_pos_b"],
                body_links_names=["left_knee", "right_knee"],
                anchor_body_name="root_link",
            )

    def test_dataset_missing_body_links(self, dataset_without_bodies):
        """Should raise KeyError when body components requested but dataset lacks body data."""
        with pytest.raises(KeyError, match="does not contain 'body_links_list'"):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_without_bodies,
                datasets={"motion_no_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["body_pos_b"],
                body_links_names=["root_link", "left_knee"],
                anchor_body_name="root_link",
            )

    def test_body_name_not_in_dataset(self, dataset_with_bodies):
        """Should raise ValueError when requested body name not found in dataset."""
        with pytest.raises(ValueError, match="not found in dataset"):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_with_bodies,
                datasets={"motion_with_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["body_pos_b"],
                body_links_names=["root_link", "nonexistent_body"],
                anchor_body_name="root_link",
            )

    def test_invalid_obs_component(self, dataset_with_bodies):
        """Should raise ValueError when an invalid obs component is specified."""
        with pytest.raises(ValueError, match="not available"):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_with_bodies,
                datasets={"motion_with_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["nonexistent_component"],
            )

    @pytest.mark.parametrize(
        "body_links_names, match",
        [
            (["root_link", "left_knee", "left_knee"], "duplicates"),
            (["root_link"], "at least one body besides the anchor"),
        ],
    )
    def test_invalid_body_links_names(
        self, dataset_with_bodies, body_links_names, match
    ):
        with pytest.raises(ValueError, match=match):
            AMPLoader(
                device="cpu",
                dataset_path_root=dataset_with_bodies,
                datasets={"motion_with_bodies": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                amp_obs_components=["body_pos_b"],
                body_links_names=body_links_names,
                anchor_body_name="root_link",
            )


# ── Tests: Numerical Correctness ────────────────────────────────────────────


class TestNumericalCorrectness:
    """Verify the numerical correctness of relative transforms."""

    def test_body_pos_b_is_relative_to_anchor(self, tmpdir):
        """When a body is at the same position as anchor, body_pos_b should be zero."""
        # Create dataset where body 1 is at the same position as root_link
        n_frames = 50
        fps = 50.0
        t = np.linspace(0, 2 * np.pi, n_frames)

        root_position = np.zeros((n_frames, 3))
        root_position[:, 2] = 0.9
        root_quaternion = np.tile([0, 0, 0, 1], (n_frames, 1)).astype(np.float64)

        body_links_pos_w = np.zeros((n_frames, 2, 3))
        body_links_quat_w = np.zeros((n_frames, 2, 4))

        # Body 0 (anchor): same as root
        body_links_pos_w[:, 0, :] = root_position
        body_links_quat_w[:, 0, :] = root_quaternion

        # Body 1: same as anchor (so relative position should be ~0)
        body_links_pos_w[:, 1, :] = root_position
        body_links_quat_w[:, 1, :] = root_quaternion

        data = {
            "joints_list": ["j0", "j1"],
            "joint_positions": np.zeros((n_frames, 2)),
            "root_position": root_position,
            "root_quaternion": root_quaternion,
            "fps": np.array([fps]),
            "body_links_list": ["anchor", "same_body"],
            "body_links_pos_w": body_links_pos_w,
            "body_links_quat_w": body_links_quat_w,
        }
        np.save(str(tmpdir / "test.npy"), data)

        loader = AMPLoader(
            device="cpu",
            dataset_path_root=tmpdir,
            datasets={"test": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_pos_b"],
            body_links_names=["anchor", "same_body"],
            anchor_body_name="anchor",
        )

        # body_pos_b for a body coincident with anchor should be ~0
        assert torch.allclose(
            loader.all_obs, torch.zeros_like(loader.all_obs), atol=1e-5
        )

    def test_body_ori_b_identity_when_same_orientation(self, tmpdir):
        """When body has same orientation as anchor, the 6D repr should be [1,0,0, 0,1,0]."""
        n_frames = 50
        fps = 50.0

        root_position = np.zeros((n_frames, 3))
        root_position[:, 2] = 0.9
        root_quaternion = np.tile([0, 0, 0, 1], (n_frames, 1)).astype(np.float64)

        body_links_pos_w = np.zeros((n_frames, 2, 3))
        body_links_quat_w = np.zeros((n_frames, 2, 4))

        # Both bodies have identity orientation
        body_links_pos_w[:, 0, :] = root_position
        body_links_quat_w[:, 0, :] = root_quaternion
        body_links_pos_w[:, 1, :] = root_position + np.array([1, 0, 0])
        body_links_quat_w[:, 1, :] = root_quaternion

        data = {
            "joints_list": ["j0"],
            "joint_positions": np.zeros((n_frames, 1)),
            "root_position": root_position,
            "root_quaternion": root_quaternion,
            "fps": np.array([fps]),
            "body_links_list": ["anchor", "body1"],
            "body_links_pos_w": body_links_pos_w,
            "body_links_quat_w": body_links_quat_w,
        }
        np.save(str(tmpdir / "test.npy"), data)

        loader = AMPLoader(
            device="cpu",
            dataset_path_root=tmpdir,
            datasets={"test": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_ori_b"],
            body_links_names=["anchor", "body1"],
            anchor_body_name="anchor",
        )

        # Identity rotation → first 2 cols of I = [1,0,0, 0,1,0]
        expected = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        for i in range(loader.all_obs.shape[0]):
            assert torch.allclose(
                loader.all_obs[i], expected, atol=1e-5
            ), f"Frame {i}: {loader.all_obs[i]} != {expected}"

    def test_non_anchor_body_at_known_offset(self, tmpdir):
        """Body at a known offset from anchor should produce correct relative position."""
        n_frames = 50
        fps = 50.0

        root_position = np.zeros((n_frames, 3))
        root_position[:, 2] = 0.9
        # Identity orientation for simplicity
        root_quaternion = np.tile([0, 0, 0, 1], (n_frames, 1)).astype(np.float64)

        body_links_pos_w = np.zeros((n_frames, 2, 3))
        body_links_quat_w = np.zeros((n_frames, 2, 4))

        # Anchor at root
        body_links_pos_w[:, 0, :] = root_position
        body_links_quat_w[:, 0, :] = root_quaternion

        # Body1 at constant offset [0.5, 0.3, -0.2] from anchor in world frame
        # With identity anchor rotation, relative = world offset
        offset = np.array([0.5, 0.3, -0.2])
        body_links_pos_w[:, 1, :] = root_position + offset
        body_links_quat_w[:, 1, :] = root_quaternion

        data = {
            "joints_list": ["j0"],
            "joint_positions": np.zeros((n_frames, 1)),
            "root_position": root_position,
            "root_quaternion": root_quaternion,
            "fps": np.array([fps]),
            "body_links_list": ["anchor", "body1"],
            "body_links_pos_w": body_links_pos_w,
            "body_links_quat_w": body_links_quat_w,
        }
        np.save(str(tmpdir / "test.npy"), data)

        loader = AMPLoader(
            device="cpu",
            dataset_path_root=tmpdir,
            datasets={"test": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_pos_b"],
            body_links_names=["anchor", "body1"],
            anchor_body_name="anchor",
        )

        expected = torch.tensor(offset, dtype=torch.float32)
        for i in range(loader.all_obs.shape[0]):
            assert torch.allclose(
                loader.all_obs[i], expected, atol=1e-4
            ), f"Frame {i}: {loader.all_obs[i]} != {expected}"

    @staticmethod
    def _rotating_anchor_dataset(tmpdir, n_frames=200, fps=100.0):
        """Anchor yawing at 0.3 rad/s while the body rolls at 0.5 rad/s and slides
        along the anchor x axis at 0.1 m/s, both measured in the anchor frame."""
        t = np.arange(n_frames) / fps
        rot_a = Rotation.from_euler("z", 0.3 * t[:, None])
        pos_a = np.stack([0.2 * t, 0.1 * t, np.full_like(t, 0.9)], axis=1)
        rel_pos = np.stack(
            [0.2 + 0.1 * t, -0.1 * np.ones_like(t), -0.5 * np.ones_like(t)], axis=1
        )
        rot_b = rot_a * Rotation.from_euler("x", 0.5 * t[:, None])
        pos_b = pos_a + rot_a.apply(rel_pos)

        data = {
            "joints_list": ["j0"],
            "joint_positions": np.zeros((n_frames, 1)),
            "root_position": pos_a,
            "root_quaternion": rot_a.as_quat(),
            "fps": np.array([fps]),
            "body_links_list": ["anchor", "body1"],
            "body_links_pos_w": np.stack([pos_a, pos_b], axis=1),
            "body_links_quat_w": np.stack([rot_a.as_quat(), rot_b.as_quat()], axis=1),
        }
        np.save(str(tmpdir / "rot.npy"), data)
        return t, rel_pos

    def test_relative_pose_with_rotating_anchor(self, tmpdir):
        """body_pos_b / body_ori_b must be expressed in the (rotating) anchor frame."""
        self._rotating_anchor_dataset(tmpdir)
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=tmpdir,
            datasets={"rot": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_pos_b", "body_ori_b"],
            body_links_names=["anchor", "body1"],
            anchor_body_name="anchor",
        )
        obs = loader.all_obs.numpy()
        assert np.allclose(obs[:, 1:3], [-0.1, -0.5], atol=1e-5)
        # Relative orientation is a pure roll: first column must stay [1, 0, 0]
        assert np.allclose(obs[:, 3:6], [1.0, 0.0, 0.0], atol=1e-5)
        roll = np.arctan2(obs[:, 8], obs[:, 7])
        # rtol absorbs the small time-grid rescaling of the loader resampling
        assert np.allclose(np.diff(roll) / 0.02, 0.5, rtol=2e-2)

    def test_relative_velocities_with_rotating_anchor(self, tmpdir):
        """body_lin_vel_b is d/dt body_pos_b and body_ang_vel_b is R_a^T (w_b - w_a)."""
        self._rotating_anchor_dataset(tmpdir)
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=tmpdir,
            datasets={"rot": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_lin_vel_b", "body_ang_vel_b"],
            body_links_names=["anchor", "body1"],
            anchor_body_name="anchor",
        )
        obs = loader.all_obs.numpy()
        assert np.allclose(obs[:, 0:3], [0.1, 0.0, 0.0], rtol=2e-2, atol=1e-3)
        assert np.allclose(obs[:, 3:6], [0.5, 0.0, 0.0], rtol=2e-2, atol=1e-3)

    def test_body_order_follows_body_links_names(self, dataset_with_bodies):
        """Bodies are emitted in the order of body_links_names, not of the dataset."""
        kwargs = dict(
            device="cpu",
            dataset_path_root=dataset_with_bodies,
            datasets={"motion_with_bodies": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            amp_obs_components=["body_pos_b"],
            anchor_body_name="root_link",
        )
        a = AMPLoader(
            body_links_names=["root_link", "left_knee", "right_foot"], **kwargs
        )
        b = AMPLoader(
            body_links_names=["right_foot", "root_link", "left_knee"], **kwargs
        )
        assert torch.allclose(a.all_obs[:, :3], b.all_obs[:, 3:])
        assert torch.allclose(a.all_obs[:, 3:], b.all_obs[:, :3])
