# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-frame AMP discriminator windows (num_frames, frame_stride).

Run with:
    python -m pytest tests/test_amp_window.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.storage.replay_buffer import ReplayBuffer
from amp_rsl_rl.utils.motion_loader import AMPLoader


def _fake_dataset(num_frames, num_joints=3):
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(num_frames, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return {
        "joints_list": [f"joint_{i}" for i in range(num_joints)],
        "joint_positions": list(rng.normal(size=(num_frames, num_joints))),
        "root_position": list(rng.normal(size=(num_frames, 3))),
        "root_quaternion": list(quats),
        "fps": 50.0,
    }


@pytest.fixture
def dataset_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(Path(tmpdir) / "walk.npy", _fake_dataset(40))
        np.save(Path(tmpdir) / "turn.npy", _fake_dataset(30))
        yield Path(tmpdir)


def _loader(dataset_dir, **kwargs):
    return AMPLoader(
        device="cpu",
        dataset_path_root=dataset_dir,
        datasets={"walk": 1.0, "turn": 1.0},
        simulation_dt=0.02,
        slow_down_factor=1,
        **kwargs,
    )


def _frames(loader, motion):
    data = loader.motion_data[motion]
    return data.get_amp_dataset_obs(torch.arange(len(data)))


def test_default_pairs_are_unchanged(dataset_dir):
    loader = _loader(dataset_dir)
    frames = _frames(loader, 0)
    T = frames.shape[0]
    assert torch.equal(loader.all_obs[:T], frames)
    next_idx = torch.clamp(torch.arange(T) + 1, max=T - 1)
    assert torch.equal(loader.all_next_obs[:T], frames[next_idx])


def test_windows_follow_the_motion(dataset_dir):
    loader = _loader(dataset_dir, num_frames=4, frame_stride=2)
    frames = _frames(loader, 0)
    T, D = frames.shape
    num_windows = T - 6
    assert loader.all_obs.shape[1] == 3 * D
    assert loader.all_next_obs.shape[1] == D
    for t in (0, 5, num_windows - 1):
        assert torch.equal(
            loader.all_obs[t], frames[[t, t + 2, t + 4]].reshape(-1)
        )
        assert torch.equal(loader.all_next_obs[t], frames[t + 6])
    # The second motion starts right after the windows of the first one.
    assert torch.equal(
        loader.all_obs[num_windows], _frames(loader, 1)[[0, 2, 4]].reshape(-1)
    )


def test_window_longer_than_motion_raises(dataset_dir):
    with pytest.raises(ValueError):
        _loader(dataset_dir, num_frames=20, frame_stride=2)


def test_discriminator_normalizes_and_mirrors_each_frame():
    D, n = 5, 3
    disc = Discriminator(
        input_dim=D * n,
        hidden_layer_sizes=[8],
        reward_scale=1.0,
        empirical_normalization=True,
        num_frames=n,
        symmetry_cfg={
            "use_data_augmentation": True,
            "amp_dataset_augmentation_func": lambda obs, obs_type=None: torch.cat(
                [obs, -obs], dim=0
            ),
        },
    )
    frames = torch.randn(64, D) * 3.0 + 1.0
    disc.update_normalization(frames)
    window = frames[:12].reshape(4, n * D)
    expected = disc.amp_normalizer(frames[:12]).reshape(4, n * D)
    assert torch.allclose(disc.normalize(window), expected)

    mirrored = disc.apply_symmetry(window)
    assert mirrored.shape == (8, n * D)
    assert torch.equal(mirrored[:4], window)
    assert torch.equal(mirrored[4:], -window)
    assert disc(window).shape == (4, 1)


def test_discriminator_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        Discriminator(input_dim=10, hidden_layer_sizes=[8], reward_scale=1.0, num_frames=3)


def test_replay_buffer_window_dims():
    buffer = ReplayBuffer(obs_dim=12, buffer_size=10, next_obs_dim=4)
    buffer.insert(torch.ones(3, 12), torch.zeros(3, 4))
    states, next_states = next(buffer.feed_forward_generator(1, 3))
    assert states.shape == (3, 12) and next_states.shape == (3, 4)
