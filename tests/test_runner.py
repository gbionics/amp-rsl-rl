# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke test of AMPOnPolicyRunner with rsl-rl 5 models on a fake environment."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from amp_rsl_rl.runners import AMPOnPolicyRunner
from amp_rsl_rl.utils import AMPLoader

NUM_ENVS = 4
NUM_JOINTS = 3


def _fake_dataset(num_frames):
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(num_frames, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return {
        "joints_list": [f"joint_{i}" for i in range(NUM_JOINTS)],
        "joint_positions": list(rng.normal(size=(num_frames, NUM_JOINTS))),
        "root_position": list(rng.normal(size=(num_frames, 3))),
        "root_quaternion": list(quats),
        "fps": 50.0,
    }


class FakeEnv:
    num_envs = NUM_ENVS
    num_actions = NUM_JOINTS
    max_episode_length = 10
    device = "cpu"

    def __init__(self, amp_dim):
        self.amp_dim = amp_dim
        self.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.01), decimation=2)
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)

    def get_observations(self):
        return TensorDict(
            {
                "policy": torch.randn(NUM_ENVS, 6),
                "critic": torch.randn(NUM_ENVS, 8),
                "amp": torch.randn(NUM_ENVS, self.amp_dim),
            },
            batch_size=[NUM_ENVS],
        )

    def step(self, actions):
        assert actions.shape == (NUM_ENVS, NUM_JOINTS)
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[dones] = 0
        return self.get_observations(), torch.randn(NUM_ENVS), dones.long(), {"log": {"foo": 1.0}}


def _train_cfg(dataset_dir, actor_class="MLPModel", **actor_kwargs):
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "critic"]},
        "actor": {
            "class_name": actor_class,
            "hidden_dims": [16, 16],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": distribution,
            **actor_kwargs,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16], "obs_normalization": True},
        "algorithm": {
            "class_name": "AMP_PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "rnd_cfg": None,  # upstream PPO option, must be dropped by the runner
        },
        "discriminator": {
            "hidden_dims": [16],
            "reward_scale": 1.0,
            "loss_type": "BCEWithLogits",
            "empirical_normalization": True,
        },
        "dataset": {
            "amp_data_path": str(dataset_dir),
            "datasets": {"walk": 1.0},
            "slow_down_factor": 1,
            "amp_joint_names": [f"joint_{i}" for i in range(NUM_JOINTS)],
        },
    }


@pytest.fixture
def dataset_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(Path(tmpdir) / "walk.npy", _fake_dataset(60))
        yield Path(tmpdir)


def _amp_dim(dataset_dir):
    loader = AMPLoader(
        device="cpu",
        dataset_path_root=dataset_dir,
        datasets={"walk": 1.0},
        simulation_dt=0.02,
        slow_down_factor=1,
    )
    return loader.all_next_obs.shape[1]


@pytest.mark.parametrize("actor_class", ["MLPModel", "MoEModel"])
def test_learn_save_load(dataset_dir, actor_class):
    env = FakeEnv(_amp_dim(dataset_dir))
    extra = {"num_experts": 2, "gate_hidden_dims": [8]} if actor_class == "MoEModel" else {}
    cfg = _train_cfg(dataset_dir, actor_class, **extra)
    with tempfile.TemporaryDirectory() as log_dir:
        runner = AMPOnPolicyRunner(env, copy.deepcopy(cfg), log_dir=log_dir)
        runner.learn(num_learning_iterations=2)
        checkpoint = Path(log_dir) / "model_1.pt"
        assert checkpoint.exists()

        policy = runner.get_inference_policy()
        obs = env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_JOINTS)

        other = AMPOnPolicyRunner(env, copy.deepcopy(cfg), log_dir=None)
        other.load(str(checkpoint))
        torch.testing.assert_close(other.get_inference_policy()(obs), policy(obs))
