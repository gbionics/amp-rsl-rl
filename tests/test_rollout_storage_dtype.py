# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Tests for the dtype-preserving rollout storage backport.

Verifies that AMP training keeps non-float observation groups (e.g. ``uint8``
images) at their native dtype in the rollout buffer, mirroring rsl-rl-lib
v5.4.0's behaviour on every supported version.

Run with:
    python -m pytest tests/test_rollout_storage_dtype.py -v
"""

import torch
from tensordict import TensorDict

from amp_rsl_rl.storage import build_rollout_storage, preserve_observation_dtypes


def _make_obs(num_envs: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(num_envs, 48, dtype=torch.float32),
            "critic": torch.zeros(num_envs, 60, dtype=torch.float32),
            "camera": torch.zeros(num_envs, 1, 32, 32, dtype=torch.uint8),
        },
        batch_size=[num_envs],
    )


def _obs_buffer_bytes(storage) -> int:
    return sum(v.element_size() * v.nelement() for v in storage.observations.values())


def test_build_rollout_storage_preserves_dtypes():
    """Each observation group keeps its native dtype in the buffer."""
    num_envs, horizon = 64, 8
    obs = _make_obs(num_envs)

    storage = build_rollout_storage(
        num_envs=num_envs,
        num_transitions_per_env=horizon,
        obs=obs,
        actions_shape=(12,),
        device="cpu",
    )

    assert storage.observations["policy"].dtype == torch.float32
    assert storage.observations["critic"].dtype == torch.float32
    assert storage.observations["camera"].dtype == torch.uint8

    # Shapes must match the (horizon, num_envs, *feature) layout.
    assert tuple(storage.observations["camera"].shape) == (horizon, num_envs, 1, 32, 32)


def test_uint8_obs_uses_less_memory_than_float32():
    """Preserving uint8 cuts the buffer to a quarter of the coerced size."""
    num_envs, horizon = 128, 8
    obs = _make_obs(num_envs)

    storage = build_rollout_storage(
        num_envs=num_envs,
        num_transitions_per_env=horizon,
        obs=obs,
        actions_shape=(12,),
        device="cpu",
    )
    preserved = storage.observations["camera"]

    coerced_bytes = preserved.nelement() * torch.empty(0, dtype=torch.float32).element_size()
    preserved_bytes = preserved.element_size() * preserved.nelement()

    assert preserved_bytes * 4 == coerced_bytes


def test_preserve_is_noop_when_dtypes_match():
    """Float32-only observations are left untouched (idempotent)."""
    num_envs, horizon = 32, 4
    obs = TensorDict(
        {"policy": torch.zeros(num_envs, 16, dtype=torch.float32)},
        batch_size=[num_envs],
    )
    storage = build_rollout_storage(
        num_envs=num_envs,
        num_transitions_per_env=horizon,
        obs=obs,
        actions_shape=(4,),
        device="cpu",
    )
    before = storage.observations["policy"]
    preserve_observation_dtypes(storage, obs)
    # Same object identity: no reallocation happened.
    assert storage.observations["policy"] is before


def test_stored_uint8_values_round_trip():
    """Writing an observation into the buffer preserves uint8 values exactly."""
    num_envs, horizon = 16, 4
    obs = _make_obs(num_envs)
    storage = build_rollout_storage(
        num_envs=num_envs,
        num_transitions_per_env=horizon,
        obs=obs,
        actions_shape=(12,),
        device="cpu",
    )

    step_obs = obs.clone()
    step_obs["camera"] = torch.randint(0, 256, (num_envs, 1, 32, 32), dtype=torch.uint8)

    # This mirrors what RolloutStorage.add_transition does internally.
    storage.observations[0].copy_(step_obs)

    assert storage.observations["camera"][0].dtype == torch.uint8
    assert torch.equal(storage.observations["camera"][0], step_obs["camera"])
