# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic end-to-end test for the multi-skill AMP additions.

Exercises:
  * AMPLoader per-skill sampling buffers (``skills`` mapping + ``skill_id``).
  * ActorCriticMoE gate-logit exposure.
  * AMP_PPO with multiple discriminators, per-skill replay routing and the
    MoE gate-supervision loss.

Runs on CPU with tiny tensors; no simulator required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import ActorCriticMoE, Discriminator
from amp_rsl_rl.utils import AMPLoader


N_JOINTS = 4
AMP_DIM = N_JOINTS + N_JOINTS + 3 + 3  # jp + jv + lin(3) + ang(3)
POLICY_DIM = 10
CRITIC_DIM = 12
NUM_ACTIONS = 4
NUM_ENVS = 16
STEPS = 8
NUM_SKILLS = 2


def _make_fake_dataset(path: Path, n_frames: int = 60, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    joints = [f"j{i}" for i in range(N_JOINTS)]
    jp = [rng.standard_normal(N_JOINTS).astype(np.float32) for _ in range(n_frames)]
    root_pos = [rng.standard_normal(3).astype(np.float32) for _ in range(n_frames)]
    # random but normalized quaternions in xyzw
    quats = rng.standard_normal((n_frames, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    root_quat = [quats[i] for i in range(n_frames)]
    data = {
        "joints_list": joints,
        "joint_positions": jp,
        "root_position": root_pos,
        "root_quaternion": root_quat,
        "fps": 30.0,
    }
    np.save(str(path), data, allow_pickle=True)


def _build_loader(root: Path) -> AMPLoader:
    _make_fake_dataset(root / "walk_a.npy", seed=1)
    _make_fake_dataset(root / "walk_b.npy", seed=2)
    _make_fake_dataset(root / "turn.npy", seed=3)
    return AMPLoader(
        device="cpu",
        dataset_path_root=root,
        datasets={"walk_a": 1.0, "walk_b": 1.0, "turn": 1.0},
        simulation_dt=1.0 / 30.0,
        slow_down_factor=1,
        expected_joint_names=[f"j{i}" for i in range(N_JOINTS)],
        skills={"locomotion": ["walk_a", "walk_b"], "turn_in_place": ["turn"]},
    )


def _make_obs() -> TensorDict:
    skill = torch.randint(0, NUM_SKILLS, (NUM_ENVS, 1)).float()
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, POLICY_DIM),
            "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
            "amp": torch.randn(NUM_ENVS, AMP_DIM),
            "skill": skill,
        },
        batch_size=[NUM_ENVS],
    )


def test_amploader_per_skill_sampling():
    with tempfile.TemporaryDirectory() as tmp:
        loader = _build_loader(Path(tmp))
        assert loader.num_skills == 2
        assert loader.skill_names == ["locomotion", "turn_in_place"]
        # per-skill generator returns correctly shaped, distinct-sized buffers
        for skill_id in range(2):
            gen = loader.feed_forward_generator(2, 5, skill_id=skill_id)
            s, ns = next(gen)
            assert s.shape == (5, AMP_DIM)
            assert ns.shape == (5, AMP_DIM)
        # full (skill_id=None) generator still works
        s, ns = next(loader.feed_forward_generator(1, 7))
        assert s.shape == (7, AMP_DIM)


def test_shared_dataset_across_skills():
    """A dataset listed under more than one skill contributes to every buffer."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_dataset(root / "walk.npy", seed=1)
        _make_fake_dataset(root / "turn.npy", seed=3)
        _make_fake_dataset(root / "stand.npy", n_frames=40, seed=5)
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=root,
            datasets={"walk": 1.0, "turn": 1.0, "stand": 1.0},
            simulation_dt=1.0 / 30.0,
            slow_down_factor=1,
            expected_joint_names=[f"j{i}" for i in range(N_JOINTS)],
            # "stand" is shared between both skills
            skills={
                "locomotion": ["walk", "stand"],
                "turn_in_place": ["turn", "stand"],
            },
        )
        # Both skill buffers must contain the shared standing frames, so each
        # skill buffer is larger than the shared dataset alone.
        n_stand = len(loader.motion_data[2])
        assert loader.skill_obs[0].shape[0] > n_stand
        assert loader.skill_obs[1].shape[0] > n_stand
        for skill_id in range(2):
            s, ns = next(loader.feed_forward_generator(1, 4, skill_id=skill_id))
            assert s.shape == (4, AMP_DIM)


def test_multiskill_update_runs():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _build_loader(Path(tmp))
        obs = _make_obs()
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        actor_critic = ActorCriticMoE(
            obs,
            obs_groups,
            NUM_ACTIONS,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            num_experts=NUM_SKILLS,
        )
        discriminators = [
            Discriminator(
                input_dim=AMP_DIM * 2,
                hidden_layer_sizes=[16, 16],
                reward_scale=1.0,
                device="cpu",
                empirical_normalization=True,
            )
            for _ in range(NUM_SKILLS)
        ]
        alg = AMP_PPO(
            discriminators=discriminators,
            amp_data=loader,
            actor_critic=actor_critic,
            num_learning_epochs=2,
            num_mini_batches=2,
            gate_loss_coef=0.5,
            device="cpu",
        )
        assert alg.num_skills == NUM_SKILLS
        assert alg._gate_supervision_available

        alg.init_storage(NUM_ENVS, STEPS, obs.clone(), (NUM_ACTIONS,))

        amp_obs = obs["amp"].clone()
        skill_ids = obs["skill"].reshape(-1).long()
        for _ in range(STEPS):
            actions = alg.act(obs)
            alg.act_amp(amp_obs, skill_ids)
            next_obs = _make_obs()
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
            alg.process_env_step(next_obs, rewards, dones, {})
            next_amp_obs = next_obs["amp"].clone()
            alg.process_amp_step(next_amp_obs)
            obs = next_obs
            amp_obs = next_amp_obs
            skill_ids = obs["skill"].reshape(-1).long()

        # both replay buffers should have received transitions
        assert len(alg.amp_storage[0]) > 0
        assert len(alg.amp_storage[1]) > 0

        alg.compute_returns(obs)
        stats = alg.update()
        assert len(stats) == 10
        for v in stats:
            assert np.isfinite(v)
        assert np.isfinite(alg.mean_gate_loss)


def _aug_obs(obs=None, actions=None, obs_type=None):
    """Trivial symmetry augmentation: concatenate [obs, obs] for the requested
    groups only. Mirrors the real ``mirror_observations`` behaviour where the
    returned TensorDict drops non-requested groups (e.g. "skill")."""
    aug_obs = None
    if obs is not None and obs_type is not None:
        if isinstance(obs_type, str):
            obs_type = [obs_type]
        data = {t: torch.cat([obs[t], obs[t]], dim=0) for t in obs_type}
        first = next(iter(data.values()))
        aug_obs = TensorDict(data, batch_size=[first.shape[0]])
    aug_actions = torch.cat([actions, actions], dim=0) if actions is not None else None
    return aug_obs, aug_actions


def _aug_amp(obs=None, obs_type=None):
    if obs is None:
        return None, None
    return torch.cat([obs, obs], dim=0), None


def test_multiskill_update_with_augmentation():
    """Gate supervision must still work when symmetry augmentation rebuilds
    obs_batch without the 'skill' group."""
    torch.manual_seed(0)
    symmetry_cfg = {
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "data_augmentation_func": _aug_obs,
        "amp_dataset_augmentation_func": _aug_amp,
    }
    with tempfile.TemporaryDirectory() as tmp:
        loader = _build_loader(Path(tmp))
        obs = _make_obs()
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        actor_critic = ActorCriticMoE(
            obs,
            obs_groups,
            NUM_ACTIONS,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            num_experts=NUM_SKILLS,
        )
        discriminators = [
            Discriminator(
                input_dim=AMP_DIM * 2,
                hidden_layer_sizes=[16, 16],
                reward_scale=1.0,
                device="cpu",
                empirical_normalization=True,
                symmetry_cfg=symmetry_cfg,
            )
            for _ in range(NUM_SKILLS)
        ]
        alg = AMP_PPO(
            discriminators=discriminators,
            amp_data=loader,
            actor_critic=actor_critic,
            num_learning_epochs=2,
            num_mini_batches=2,
            gate_loss_coef=0.5,
            symmetry_cfg=symmetry_cfg,
            device="cpu",
        )
        alg.init_storage(NUM_ENVS, STEPS, obs.clone(), (NUM_ACTIONS,))

        amp_obs = obs["amp"].clone()
        skill_ids = obs["skill"].reshape(-1).long()
        for _ in range(STEPS):
            actions = alg.act(obs)
            alg.act_amp(amp_obs, skill_ids)
            next_obs = _make_obs()
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
            alg.process_env_step(next_obs, rewards, dones, {})
            next_amp_obs = next_obs["amp"].clone()
            alg.process_amp_step(next_amp_obs)
            obs = next_obs
            amp_obs = next_amp_obs
            skill_ids = obs["skill"].reshape(-1).long()

        alg.compute_returns(obs)
        stats = alg.update()
        for v in stats:
            assert np.isfinite(v)
        # gate loss should be > 0 (it was actually computed under augmentation)
        assert alg.mean_gate_loss > 0.0



if __name__ == "__main__":
    test_amploader_per_skill_sampling()
    test_shared_dataset_across_skills()
    test_multiskill_update_runs()
    test_multiskill_update_with_augmentation()
    print("all multi-skill AMP tests passed")
