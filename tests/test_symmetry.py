# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetry augmentation in AMP_PPO and Discriminator.

Run with:
    python -m pytest tests/test_symmetry.py -v
"""

import torch
import pytest
import numpy as np
import tempfile
from pathlib import Path
from tensordict import TensorDict


from amp_rsl_rl.utils.motion_loader import (
    _call_augmentation_func,
    AMPLoader,
    VelocityRepresentation,
)
from amp_rsl_rl.networks import Discriminator

# ── Test devices ─────────────────────────────────────────────────────────────

_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")


# ── Fake augmentation functions (mimics what gb-rl-locomotion provides) ──────


def _mirror_obs_and_actions(obs, actions, obs_type=None):
    """Fake augmentation that doubles the batch by appending negated copy."""
    aug_obs = None
    aug_actions = None
    if obs is not None:
        if isinstance(obs, TensorDict):
            types = obs_type if isinstance(obs_type, list) else [obs_type]
            data = {}
            for t in types:
                current = obs.get(t)
                mirrored = -current  # simple negation as mock mirror
                data[t] = torch.cat([current, mirrored], dim=0)
            first = next(iter(data.values()))
            aug_obs = TensorDict(data, batch_size=[first.shape[0]], device=first.device)
        else:
            aug_obs = torch.cat([obs, -obs], dim=0)
    if actions is not None:
        aug_actions = torch.cat([actions, -actions], dim=0)
    return aug_obs, aug_actions


def _mirror_obs_only(obs):
    """Fake augmentation function that only takes obs (no actions/obs_type)."""
    if obs is None:
        return None
    return torch.cat([obs, -obs], dim=0)


def _mirror_amp_dataset(obs, obs_type=None):
    """Fake AMP dataset augmentation that doubles batch."""
    if obs is None:
        return None
    return torch.cat([obs, -obs], dim=0)


# ── Tests for _call_augmentation_func ────────────────────────────────────────


class TestCallAugmentationFunc:
    """Tests for the flexible call dispatcher utility."""

    def test_full_signature(self):
        """Function with obs, actions, obs_type all present."""
        result = _call_augmentation_func(
            _mirror_obs_and_actions,
            obs=torch.ones(4, 3),
            actions=torch.ones(4, 2),
            obs_type="policy",
        )
        aug_obs, aug_actions = result
        # The function receives a plain tensor, so it doubles via cat
        assert aug_obs.shape == (8, 3)
        assert aug_actions.shape == (8, 2)

    def test_obs_only_signature(self):
        """Function with only obs parameter."""
        result = _call_augmentation_func(
            _mirror_obs_only,
            obs=torch.ones(4, 5),
            actions=torch.ones(4, 2),
            obs_type="amp",
        )
        # _mirror_obs_only returns a single tensor, not a tuple
        aug_obs, aug_actions = result
        assert aug_obs.shape == (8, 5)
        assert aug_actions is None  # function doesn't handle actions

    def test_none_obs(self):
        """Passing obs=None should forward None."""
        result = _call_augmentation_func(
            _mirror_obs_only,
            obs=None,
            actions=torch.ones(4, 2),
        )
        aug_obs, aug_actions = result
        assert aug_obs is None
        assert aug_actions is None

    def test_none_actions(self):
        """Passing actions=None should forward None."""
        result = _call_augmentation_func(
            _mirror_obs_and_actions,
            obs=torch.ones(4, 3),
            actions=None,
            obs_type="policy",
        )
        aug_obs, aug_actions = result
        assert aug_obs.shape == (8, 3)
        assert aug_actions is None


# ── Tests for Discriminator.apply_symmetry ───────────────────────────────────


class TestDiscriminatorApplySymmetry:
    """Tests for Discriminator.apply_symmetry method."""

    def _make_discriminator(self, symmetry_cfg=None, device="cpu"):
        return Discriminator(
            input_dim=20,
            hidden_layer_sizes=[64, 32],
            reward_scale=1.0,
            device=device,
            symmetry_cfg=symmetry_cfg,
        )

    def test_no_symmetry_cfg_passthrough(self):
        """With no symmetry_cfg, apply_symmetry returns input unchanged."""
        disc = self._make_discriminator(symmetry_cfg=None)
        x = torch.randn(8, 10)
        out = disc.apply_symmetry(x, obs_type="amp")
        assert torch.equal(out, x)

    def test_augmentation_disabled_passthrough(self):
        """With use_data_augmentation=False, returns input unchanged."""
        cfg = {
            "use_data_augmentation": False,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        disc = self._make_discriminator(symmetry_cfg=cfg)
        x = torch.randn(8, 10)
        out = disc.apply_symmetry(x, obs_type="amp")
        assert torch.equal(out, x)

    def test_augmentation_doubles_batch(self):
        """With valid symmetry_cfg, apply_symmetry should augment the tensor."""
        cfg = {
            "use_data_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        disc = self._make_discriminator(symmetry_cfg=cfg)
        x = torch.randn(8, 10)
        out = disc.apply_symmetry(x, obs_type="amp")
        assert out.shape == (16, 10)
        # First half should be original
        assert torch.allclose(out[:8], x)
        # Second half should be negated (our mock)
        assert torch.allclose(out[8:], -x)

    def test_augmentation_none_func_raises(self):
        """If amp_dataset_augmentation_func is None but augmentation enabled, raises ValueError."""
        cfg = {
            "use_data_augmentation": True,
            "amp_dataset_augmentation_func": None,
        }
        disc = self._make_discriminator(symmetry_cfg=cfg)
        x = torch.randn(8, 10)
        with pytest.raises(ValueError, match="no amp_dataset_augmentation_func"):
            disc.apply_symmetry(x, obs_type="amp")


# ── Tests for AMP_PPO symmetry methods ──────────────────────────────────────


class TestAMPPPOSymmetryMethods:
    """Tests for AMP_PPO._apply_symmetry, _augment_batch_size, _repeat_along_batch."""

    def _make_amp_ppo_stub(self, symmetry_cfg=None):
        """Create a minimal AMP_PPO-like object with only the methods we need."""
        from amp_rsl_rl.algorithms.amp_ppo import AMP_PPO

        # We can't easily instantiate AMP_PPO without all deps, so test the methods directly
        # by calling them as unbound functions with a mock self
        class Stub:
            pass

        stub = Stub()
        stub.symmetry_cfg = symmetry_cfg
        stub._apply_symmetry = AMP_PPO._apply_symmetry.__get__(stub)
        stub._augment_batch_size = AMP_PPO._augment_batch_size.__get__(stub)
        stub._repeat_along_batch = AMP_PPO._repeat_along_batch.__get__(stub)
        return stub

    # -- _augment_batch_size --

    def test_augment_batch_size_no_augmentation(self):
        """Returns 1 when augmented is None."""
        stub = self._make_amp_ppo_stub()
        assert stub._augment_batch_size(10, None) == 1

    def test_augment_batch_size_zero_original(self):
        """Returns 1 when original_size is 0."""
        stub = self._make_amp_ppo_stub()
        assert stub._augment_batch_size(0, torch.randn(10, 5)) == 1

    def test_augment_batch_size_doubled(self):
        """Returns 2 when augmented is double the original."""
        stub = self._make_amp_ppo_stub()
        assert stub._augment_batch_size(8, torch.randn(16, 5)) == 2

    def test_augment_batch_size_tripled(self):
        """Returns 3 when augmented is triple the original."""
        stub = self._make_amp_ppo_stub()
        assert stub._augment_batch_size(8, torch.randn(24, 5)) == 3

    def test_augment_batch_size_incompatible_raises(self):
        """Raises ValueError when sizes are incompatible."""
        stub = self._make_amp_ppo_stub()
        with pytest.raises(ValueError, match="incompatible"):
            stub._augment_batch_size(8, torch.randn(13, 5))

    # -- _repeat_along_batch --

    def test_repeat_along_batch_none(self):
        """Returns None when tensor is None."""
        stub = self._make_amp_ppo_stub()
        assert stub._repeat_along_batch(None, 2) is None

    def test_repeat_along_batch_factor_one(self):
        """Returns same tensor when factor is 1."""
        stub = self._make_amp_ppo_stub()
        x = torch.randn(4, 3)
        out = stub._repeat_along_batch(x, 1)
        assert out is x

    @pytest.mark.parametrize("device", _devices)
    def test_repeat_along_batch_factor_two(self, device):
        """Doubles the tensor along batch dimension."""
        stub = self._make_amp_ppo_stub()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = stub._repeat_along_batch(x, 2)
        assert out.shape == (4, 2)
        assert torch.allclose(out[:2], x)
        assert torch.allclose(out[2:], x)

    @pytest.mark.parametrize("device", _devices)
    def test_repeat_along_batch_preserves_higher_dims(self, device):
        """Higher-dimensional tensors only repeat along dim 0."""
        stub = self._make_amp_ppo_stub()
        x = torch.randn(4, 3, 5, device=device)
        out = stub._repeat_along_batch(x, 3)
        assert out.shape == (12, 3, 5)
        assert torch.allclose(out[:4], x)
        assert torch.allclose(out[4:8], x)
        assert torch.allclose(out[8:12], x)

    # -- _apply_symmetry --

    def test_apply_symmetry_no_cfg(self):
        """With no symmetry_cfg, _apply_symmetry returns inputs unchanged."""
        stub = self._make_amp_ppo_stub(symmetry_cfg=None)
        obs = torch.randn(4, 10)
        actions = torch.randn(4, 6)
        out_obs, out_actions = stub._apply_symmetry(
            obs=obs, actions=actions, obs_type="policy"
        )
        assert out_obs is obs
        assert out_actions is actions

    def test_apply_symmetry_no_func(self):
        """With symmetry_cfg but no func, returns inputs unchanged."""
        stub = self._make_amp_ppo_stub(symmetry_cfg={"use_data_augmentation": True})
        obs = torch.randn(4, 10)
        actions = torch.randn(4, 6)
        out_obs, out_actions = stub._apply_symmetry(
            obs=obs, actions=actions, obs_type="policy"
        )
        assert out_obs is obs
        assert out_actions is actions

    @pytest.mark.parametrize("device", _devices)
    def test_apply_symmetry_with_tensordict(self, device):
        """With valid func and TensorDict obs, augmentation is applied."""
        cfg = {
            "use_data_augmentation": True,
            "data_augmentation_func": _mirror_obs_and_actions,
        }
        stub = self._make_amp_ppo_stub(symmetry_cfg=cfg)

        obs_data = torch.randn(4, 6, device=device)
        obs = TensorDict({"policy": obs_data}, batch_size=[4], device=device)
        actions = torch.randn(4, 3, device=device)

        out_obs, out_actions = stub._apply_symmetry(
            obs=obs, actions=actions, obs_type=["policy"]
        )

        assert out_obs["policy"].shape == (8, 6)
        assert out_actions.shape == (8, 3)
        # First half is original
        assert torch.allclose(out_obs["policy"][:4], obs_data)
        assert torch.allclose(out_actions[:4], actions)

    @pytest.mark.parametrize("device", _devices)
    def test_apply_symmetry_obs_none_actions_augmented(self, device):
        """With obs=None, only actions should be augmented."""
        cfg = {
            "use_data_augmentation": True,
            "data_augmentation_func": _mirror_obs_and_actions,
        }
        stub = self._make_amp_ppo_stub(symmetry_cfg=cfg)
        actions = torch.randn(4, 3, device=device)

        out_obs, out_actions = stub._apply_symmetry(
            obs=None, actions=actions, obs_type=["policy"]
        )
        # Function returns (None, augmented_actions) for obs=None
        assert out_obs is None
        assert out_actions.shape == (8, 3)


# ── Tests for AMP_PPO recurrent symmetry helpers ────────────────────────────


def _reference_hidden_state_selection(saved, dones, start, stop):
    """Hidden-state selection of rsl_rl ``RolloutStorage.recurrent_mini_batch_generator``."""
    dones = dones.squeeze(-1)
    last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    last_was_done[1:] = dones[:-1]
    last_was_done[0] = True
    last_was_done = last_was_done.permute(1, 0)
    first_traj = torch.cat(
        [torch.zeros(1, dtype=torch.long), last_was_done.sum(dim=1).cumsum(0)]
    )
    return (
        saved.permute(2, 0, 1, 3)[last_was_done][first_traj[start] : first_traj[stop]]
        .transpose(1, 0)
        .contiguous()
    )


class TestAMPPPORecurrentSymmetry:
    """Tests for the time-major (recurrent) symmetry helpers of AMP_PPO."""

    def _make_stub(self, symmetry_cfg=None):
        from amp_rsl_rl.algorithms.amp_ppo import AMP_PPO

        class Stub:
            pass

        stub = Stub()
        stub.symmetry_cfg = symmetry_cfg
        stub._sym_num_aug = 1
        stub._sym_saved_hidden_states = [None, None]
        for name in (
            "_apply_symmetry",
            "_apply_symmetry_time_major",
            "_augment_recurrent_inputs",
            "_symmetric_hidden_state_generator",
        ):
            setattr(stub, name, getattr(AMP_PPO, name).__get__(stub))
        stub._unflatten_time_major = AMP_PPO._unflatten_time_major
        stub._cat_hidden_states = AMP_PPO._cat_hidden_states
        return stub

    @staticmethod
    def _cfg():
        return {
            "use_data_augmentation": True,
            "data_augmentation_func": _mirror_obs_and_actions,
        }

    @pytest.mark.parametrize("device", _devices)
    def test_time_major_augmentation_stacks_along_batch(self, device):
        stub = self._make_stub(self._cfg())
        T, N = 5, 3
        policy = torch.randn(T, N, 4, device=device)
        critic = torch.randn(T, N, 6, device=device)
        obs = TensorDict(
            {"policy": policy, "critic": critic}, batch_size=[T, N], device=device
        )
        actions = torch.randn(T, N, 2, device=device)

        aug_obs, aug_actions = stub._apply_symmetry_time_major(
            obs=obs, actions=actions, obs_type=["policy", "critic"]
        )

        assert tuple(aug_obs.batch_size) == (T, 2 * N)
        assert aug_actions.shape == (T, 2 * N, 2)
        assert torch.equal(aug_obs["policy"][:, :N], policy)
        assert torch.equal(aug_obs["policy"][:, N:], -policy)
        assert torch.equal(aug_obs["critic"][:, N:], -critic)
        assert torch.equal(aug_actions[:, :N], actions)
        assert torch.equal(aug_actions[:, N:], -actions)

    def test_time_major_incompatible_size_raises(self):
        stub = self._make_stub()
        with pytest.raises(ValueError, match="incompatible"):
            stub._unflatten_time_major(torch.randn(7, 2), torch.randn(2, 3, 2))

    def test_cat_hidden_states(self):
        stub = self._make_stub()
        h = torch.randn(1, 4, 8)
        h_sym = torch.randn(1, 4, 8)
        assert stub._cat_hidden_states(None, None) is None
        out = stub._cat_hidden_states(h, h_sym)
        assert out.shape == (1, 8, 8)
        assert torch.equal(out[:, 4:], h_sym)
        lstm = stub._cat_hidden_states((h, h), (h_sym, h_sym))
        assert isinstance(lstm, tuple) and len(lstm) == 2
        assert torch.equal(lstm[1][:, :4], h)
        with pytest.raises(RuntimeError):
            stub._cat_hidden_states(h, None)

    @pytest.mark.parametrize("num_aug", [2, 3])
    def test_symmetric_hidden_state_generator_matches_storage(self, num_aug):
        torch.manual_seed(0)
        T, N, L, H = 6, 4, 2, 5
        num_mini_batches = 2
        dones = (torch.rand(T, N, 1) < 0.3).to(torch.uint8)

        class Storage:
            pass

        stub = self._make_stub(self._cfg())
        stub.storage = Storage()
        stub.storage.num_envs = N
        stub.storage.dones = dones
        stub._sym_num_aug = num_aug
        saved_actor = torch.randn(T, L, (num_aug - 1) * N, H)
        saved_critic = [torch.randn(T, L, (num_aug - 1) * N, H) for _ in range(2)]
        stub._sym_saved_hidden_states = [[saved_actor], saved_critic]

        batches = list(stub._symmetric_hidden_state_generator(num_mini_batches, 2))
        assert len(batches) == 2 * num_mini_batches

        mbs = N // num_mini_batches
        for i, (h_actor, h_critic) in enumerate(batches[:num_mini_batches]):
            start, stop = i * mbs, (i + 1) * mbs
            expected_actor = torch.cat(
                [
                    _reference_hidden_state_selection(
                        saved_actor[:, :, j * N : (j + 1) * N], dones, start, stop
                    )
                    for j in range(num_aug - 1)
                ],
                dim=1,
            )
            assert torch.equal(h_actor, expected_actor)
            assert isinstance(h_critic, tuple) and len(h_critic) == 2
            for component, saved in zip(h_critic, saved_critic):
                expected = torch.cat(
                    [
                        _reference_hidden_state_selection(
                            saved[:, :, j * N : (j + 1) * N], dones, start, stop
                        )
                        for j in range(num_aug - 1)
                    ],
                    dim=1,
                )
                assert torch.equal(component, expected)

    def test_augment_recurrent_inputs(self):
        stub = self._make_stub(self._cfg())
        stub._sym_num_aug = 2
        T, n_traj, L, H = 4, 5, 1, 3
        obs = TensorDict(
            {
                "policy": torch.randn(T, n_traj, 2),
                "critic": torch.randn(T, n_traj, 3),
            },
            batch_size=[T, n_traj],
        )
        masks = torch.rand(T, n_traj) > 0.5
        h_a, h_a_sym = torch.randn(L, n_traj, H), torch.randn(L, n_traj, H)

        aug_obs, aug_masks, (aug_h_a, aug_h_c), num_aug = (
            stub._augment_recurrent_inputs(obs, masks, (h_a, None), (h_a_sym, None))
        )

        assert num_aug == 2
        assert tuple(aug_obs.batch_size) == (T, 2 * n_traj)
        assert torch.equal(aug_masks, torch.cat([masks, masks], dim=1))
        assert torch.equal(aug_h_a, torch.cat([h_a, h_a_sym], dim=1))
        assert aug_h_c is None


# ── Integration: symmetry augmentation flow ──────────────────────────────────


class TestSymmetryAugmentationFlow:
    """Integration tests verifying the full augmentation pipeline properties."""

    @pytest.mark.parametrize("device", _devices)
    def test_augmentation_preserves_original_data(self, device):
        """Augmentation should not modify the original batch data."""
        obs_data = torch.randn(8, 10, device=device)
        actions = torch.randn(8, 4, device=device)
        obs_orig = obs_data.clone()
        actions_orig = actions.clone()

        _mirror_obs_and_actions(
            TensorDict({"policy": obs_data}, batch_size=[8], device=device),
            actions,
            obs_type=["policy"],
        )
        # Original tensors should be unchanged
        assert torch.allclose(obs_data, obs_orig)
        assert torch.allclose(actions, actions_orig)

    @pytest.mark.parametrize("device", _devices)
    def test_augmented_batch_is_consistent_size(self, device):
        """All tensors in the augmented batch should have matching batch size."""
        batch = 16
        obs_dim, act_dim = 12, 6
        obs_data = torch.randn(batch, obs_dim, device=device)
        actions = torch.randn(batch, act_dim, device=device)

        obs = TensorDict(
            {"policy": obs_data, "critic": obs_data.clone()},
            batch_size=[batch],
            device=device,
        )

        aug_obs, aug_actions = _mirror_obs_and_actions(
            obs, actions, obs_type=["policy", "critic"]
        )

        # Both obs types and actions should have same augmented batch size
        assert aug_obs["policy"].shape[0] == aug_obs["critic"].shape[0]
        assert aug_obs["policy"].shape[0] == aug_actions.shape[0]
        assert aug_obs["policy"].shape[0] == 2 * batch

    @pytest.mark.parametrize("device", _devices)
    def test_repeat_matches_augmentation_factor(self, device):
        """_repeat_along_batch should produce tensors matching the augmented batch."""
        from amp_rsl_rl.algorithms.amp_ppo import AMP_PPO

        class Stub:
            pass

        stub = Stub()
        stub.symmetry_cfg = None
        stub._augment_batch_size = AMP_PPO._augment_batch_size.__get__(stub)
        stub._repeat_along_batch = AMP_PPO._repeat_along_batch.__get__(stub)

        original_batch = 16
        augmented = torch.randn(32, 10, device=device)  # doubled
        num_aug = stub._augment_batch_size(original_batch, augmented)
        assert num_aug == 2

        advantages = torch.randn(original_batch, device=device)
        repeated = stub._repeat_along_batch(advantages, num_aug)
        assert repeated.shape[0] == augmented.shape[0]

    @pytest.mark.parametrize("device", _devices)
    def test_involution_property(self, device):
        """Applying a well-formed mirror augmentation twice should return original.

        This tests a realistic joint-swap mirror function.
        """
        # Simulate 4-joint biped: [l_hip, r_hip, l_knee, r_knee]
        mirror_perm = [1, 0, 3, 2]  # swap left/right
        negate_indices = []  # no inversions

        def realistic_mirror(obs, actions, obs_type=None):
            aug_obs = None
            aug_actions = None
            if obs is not None:
                if isinstance(obs, TensorDict):
                    types = obs_type if isinstance(obs_type, list) else [obs_type]
                    data = {}
                    for t in types:
                        current = obs.get(t)
                        mirrored = current[:, mirror_perm].clone()
                        data[t] = torch.cat([current, mirrored], dim=0)
                    first = next(iter(data.values()))
                    aug_obs = TensorDict(
                        data, batch_size=[first.shape[0]], device=first.device
                    )
                else:
                    mirrored = obs[:, mirror_perm].clone()
                    aug_obs = torch.cat([obs, mirrored], dim=0)
            if actions is not None:
                mirrored_a = actions[:, mirror_perm].clone()
                aug_actions = torch.cat([actions, mirrored_a], dim=0)
            return aug_obs, aug_actions

        batch = 8
        obs_data = torch.randn(batch, 4, device=device)
        actions = torch.randn(batch, 4, device=device)
        obs = TensorDict({"policy": obs_data}, batch_size=[batch], device=device)

        # First augmentation
        aug_obs, aug_actions = realistic_mirror(obs, actions, obs_type=["policy"])
        mirrored_obs = aug_obs["policy"][batch:]
        mirrored_actions = aug_actions[batch:]

        # Second augmentation on mirrored half
        obs2 = TensorDict({"policy": mirrored_obs}, batch_size=[batch], device=device)
        aug_obs2, aug_actions2 = realistic_mirror(
            obs2, mirrored_actions, obs_type=["policy"]
        )
        double_mirrored_obs = aug_obs2["policy"][batch:]
        double_mirrored_actions = aug_actions2[batch:]

        # Double mirror should be identity (involution)
        assert torch.allclose(double_mirrored_obs, obs_data, atol=1e-6)
        assert torch.allclose(double_mirrored_actions, actions, atol=1e-6)

    @pytest.mark.parametrize("device", _devices)
    def test_involution_with_negation(self, device):
        """Mirror with joint swap + sign negation is still an involution."""
        # Simulate: [l_hip, r_hip, l_rod, r_rod]
        # l_rod and r_rod are inverted after swap
        mirror_perm = [1, 0, 3, 2]
        negate_indices = [2, 3]  # rod joints get sign-flipped

        def mirror_with_negate(obs, actions, obs_type=None):
            aug_obs = None
            aug_actions = None
            if obs is not None:
                if isinstance(obs, TensorDict):
                    types = obs_type if isinstance(obs_type, list) else [obs_type]
                    data = {}
                    for t in types:
                        current = obs.get(t)
                        mirrored = current[:, mirror_perm].clone()
                        mirrored[:, negate_indices] *= -1.0
                        data[t] = torch.cat([current, mirrored], dim=0)
                    first = next(iter(data.values()))
                    aug_obs = TensorDict(
                        data, batch_size=[first.shape[0]], device=first.device
                    )
            if actions is not None:
                mirrored_a = actions[:, mirror_perm].clone()
                mirrored_a[:, negate_indices] *= -1.0
                aug_actions = torch.cat([actions, mirrored_a], dim=0)
            return aug_obs, aug_actions

        batch = 8
        obs_data = torch.randn(batch, 4, device=device)
        actions = torch.randn(batch, 4, device=device)
        obs = TensorDict({"policy": obs_data}, batch_size=[batch], device=device)

        # First augmentation
        aug_obs, aug_actions = mirror_with_negate(obs, actions, obs_type=["policy"])
        mirrored_obs = aug_obs["policy"][batch:]
        mirrored_actions = aug_actions[batch:]

        # Second augmentation on mirrored half
        obs2 = TensorDict({"policy": mirrored_obs}, batch_size=[batch], device=device)
        aug_obs2, aug_actions2 = mirror_with_negate(
            obs2, mirrored_actions, obs_type=["policy"]
        )
        double_mirrored_obs = aug_obs2["policy"][batch:]
        double_mirrored_actions = aug_actions2[batch:]

        # Double mirror should be identity
        assert torch.allclose(double_mirrored_obs, obs_data, atol=1e-6)
        assert torch.allclose(double_mirrored_actions, actions, atol=1e-6)


# ── Test Discriminator symmetry integration ──────────────────────────────────


class TestDiscriminatorSymmetryIntegration:
    """Integration tests for symmetry augmentation in the discriminator forward path."""

    @pytest.mark.parametrize("device", _devices)
    def test_discriminator_forward_with_augmented_input(self, device):
        """Discriminator should handle augmented (doubled) batch without errors."""
        amp_obs_dim = 10
        disc = Discriminator(
            input_dim=amp_obs_dim * 2,
            hidden_layer_sizes=[64, 32],
            reward_scale=1.0,
            device=device,
        )

        # Simulate augmented input: state + next_state, doubled batch
        batch = 8
        state = torch.randn(2 * batch, amp_obs_dim, device=device)
        next_state = torch.randn(2 * batch, amp_obs_dim, device=device)
        x = torch.cat([state, next_state], dim=-1)

        output = disc(x)
        assert output.shape == (2 * batch, 1)

    @pytest.mark.parametrize("device", _devices)
    def test_discriminator_reward_with_augmented_input(self, device):
        """Discriminator reward computation works with augmented batches."""
        amp_obs_dim = 10
        disc = Discriminator(
            input_dim=amp_obs_dim * 2,
            hidden_layer_sizes=[64, 32],
            reward_scale=2.0,
            device=device,
        )

        batch = 8
        state = torch.randn(2 * batch, amp_obs_dim, device=device)
        next_state = torch.randn(2 * batch, amp_obs_dim, device=device)

        reward = disc.predict_reward(state, next_state)
        assert reward.shape == (2 * batch,)
        # Rewards should be finite
        assert torch.isfinite(reward).all()

    @pytest.mark.parametrize("device", _devices)
    def test_augmented_vs_non_augmented_consistency(self, device):
        """First half of augmented output should match non-augmented output."""
        amp_obs_dim = 10
        disc = Discriminator(
            input_dim=amp_obs_dim * 2,
            hidden_layer_sizes=[64, 32],
            reward_scale=1.0,
            device=device,
            use_minibatch_std=False,  # Disable for deterministic comparison
        )
        disc.eval()

        batch = 8
        state = torch.randn(batch, amp_obs_dim, device=device)
        next_state = torch.randn(batch, amp_obs_dim, device=device)

        # Non-augmented
        x_orig = torch.cat([state, next_state], dim=-1)
        out_orig = disc(x_orig)

        # Augmented (doubled with copy)
        x_aug = torch.cat([x_orig, x_orig], dim=0)
        out_aug = disc(x_aug)

        # First half should give same results (without minibatch_std)
        assert torch.allclose(out_aug[:batch], out_orig, atol=1e-5)


# ── Tests for AMPLoader symmetry augmentation ────────────────────────────────


def _make_fake_dataset(num_frames=50, num_joints=6):
    """Create a fake motion dataset dict matching AMPLoader's expected .npy format."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    joint_positions = [
        np.random.randn(num_joints).astype(np.float32) for _ in range(num_frames)
    ]
    # Random 3D root positions
    root_position = [np.random.randn(3).astype(np.float32) for _ in range(num_frames)]
    # Random quaternions (xyzw), normalized
    root_quaternion = []
    for _ in range(num_frames):
        q = np.random.randn(4).astype(np.float32)
        q /= np.linalg.norm(q)
        root_quaternion.append(q)

    return {
        "joints_list": joint_names,
        "joint_positions": joint_positions,
        "root_position": root_position,
        "root_quaternion": root_quaternion,
        "fps": 30.0,
    }


@pytest.fixture
def fake_dataset_dir():
    """Create a temporary directory with fake .npy motion datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _make_fake_dataset(num_frames=50, num_joints=6)
        np.save(Path(tmpdir) / "walk.npy", dataset)
        dataset2 = _make_fake_dataset(num_frames=40, num_joints=6)
        np.save(Path(tmpdir) / "run.npy", dataset2)
        yield Path(tmpdir)


class TestAMPLoaderSymmetry:
    """Tests for AMPLoader symmetry augmentation."""

    def test_loader_without_symmetry(self, fake_dataset_dir):
        """AMPLoader works without symmetry_cfg."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        assert loader.all_obs.shape[0] == loader.all_next_obs.shape[0]
        assert loader.symmetry_cfg is None

    def test_loader_with_symmetry_disabled(self, fake_dataset_dir):
        """AMPLoader with symmetry_cfg but augmentation disabled doesn't augment."""
        cfg = {
            "use_amp_dataset_augmentation": False,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader_no_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_sym_disabled = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        # Same size since augmentation is disabled
        assert loader_sym_disabled.all_obs.shape == loader_no_sym.all_obs.shape

    def test_loader_with_symmetry_doubles_obs(self, fake_dataset_dir):
        """AMPLoader with symmetry augmentation enabled should double the obs buffer."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader_no_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        # Augmented loader should have double the observations
        assert loader_sym.all_obs.shape[0] == 2 * loader_no_sym.all_obs.shape[0]
        assert (
            loader_sym.all_next_obs.shape[0] == 2 * loader_no_sym.all_next_obs.shape[0]
        )

    def test_loader_symmetry_does_not_augment_reset_states(self, fake_dataset_dir):
        """Reset states should NOT be augmented (only AMP obs are)."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader_no_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        # Reset states should have same size (not augmented)
        assert loader_sym.all_states.shape == loader_no_sym.all_states.shape

    def test_loader_augmented_obs_contains_originals(self, fake_dataset_dir):
        """The first half of augmented obs should be the original observations."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader_no_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        n_orig = loader_no_sym.all_obs.shape[0]
        # First half of augmented buffer = original obs
        assert torch.allclose(loader_sym.all_obs[:n_orig], loader_no_sym.all_obs)
        assert torch.allclose(
            loader_sym.all_next_obs[:n_orig], loader_no_sym.all_next_obs
        )

    def test_loader_augmented_second_half_is_mirrored(self, fake_dataset_dir):
        """The second half should be the mirrored (negated in our mock) observations."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader_no_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        n_orig = loader_no_sym.all_obs.shape[0]
        # Second half = negated (our mock mirror)
        assert torch.allclose(loader_sym.all_obs[n_orig:], -loader_no_sym.all_obs)

    def test_loader_feed_forward_generator_yields_correct_shapes(
        self, fake_dataset_dir
    ):
        """feed_forward_generator should yield batches from the (possibly augmented) buffer."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        mini_batch_size = 16
        gen = loader.feed_forward_generator(
            num_mini_batch=3, mini_batch_size=mini_batch_size
        )
        for state, next_state in gen:
            assert state.shape[0] == mini_batch_size
            assert next_state.shape[0] == mini_batch_size
            assert state.shape[1] == loader.all_obs.shape[1]

    def test_loader_multiple_datasets_augmented(self, fake_dataset_dir):
        """AMPLoader with multiple datasets augments all of them."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader_no_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 0.7, "run": 0.3},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        loader_sym = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 0.7, "run": 0.3},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        assert loader_sym.all_obs.shape[0] == 2 * loader_no_sym.all_obs.shape[0]

    def test_loader_apply_symmetry_passthrough_when_no_cfg(self, fake_dataset_dir):
        """AMPLoader._apply_symmetry returns obs unchanged when no symmetry_cfg."""
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
        )
        x = torch.randn(10, 12)
        out = loader._apply_symmetry(obs=x, obs_type="amp")
        assert torch.equal(out, x)

    def test_loader_apply_symmetry_none_func_passthrough(self, fake_dataset_dir):
        """AMPLoader._apply_symmetry returns obs unchanged when func is None."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": None,
        }
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 1.0},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        x = torch.randn(10, 12)
        out = loader._apply_symmetry(obs=x, obs_type="amp")
        assert torch.equal(out, x)

    def test_loader_invalid_augmentation_func_raises(self, fake_dataset_dir):
        """AMPLoader should raise if augmentation func is not callable."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": 12345,  # not callable, not a string
        }
        with pytest.raises(ValueError, match="not callable"):
            AMPLoader(
                device="cpu",
                dataset_path_root=fake_dataset_dir,
                datasets={"walk": 1.0},
                simulation_dt=0.02,
                slow_down_factor=1,
                symmetry_cfg=cfg,
            )

    def test_loader_per_frame_weights_sum_to_one(self, fake_dataset_dir):
        """Per-frame sampling weights should sum to 1 after augmentation."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 0.6, "run": 0.4},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        assert torch.allclose(
            loader.per_frame_weights.sum(), torch.tensor(1.0), atol=1e-5
        )
        assert torch.allclose(
            loader.per_frame_weights_reset.sum(), torch.tensor(1.0), atol=1e-5
        )

    def test_loader_per_frame_weights_length_matches_obs(self, fake_dataset_dir):
        """Per-frame weights should have same length as all_obs."""
        cfg = {
            "use_amp_dataset_augmentation": True,
            "amp_dataset_augmentation_func": _mirror_amp_dataset,
        }
        loader = AMPLoader(
            device="cpu",
            dataset_path_root=fake_dataset_dir,
            datasets={"walk": 0.6, "run": 0.4},
            simulation_dt=0.02,
            slow_down_factor=1,
            symmetry_cfg=cfg,
        )
        assert loader.per_frame_weights.shape[0] == loader.all_obs.shape[0]
        assert loader.per_frame_weights_reset.shape[0] == loader.all_states.shape[0]
