# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Rollout storage helpers for AMP training.

This module backports the observation-dtype preservation that landed in
``rsl-rl-lib`` v5.4.0 (upstream PR #212, "Preserve observation tensors dtype in
the rollout storage") so that AMP training benefits from it on *every* supported
``rsl-rl-lib`` version (v3.x through v5.x).

Why it matters
--------------
Older ``rsl-rl-lib`` allocated the rollout observation buffer with
``torch.zeros(..., device=device)``, which always yields ``float32`` regardless
of the observation dtype. Vision-based observations (e.g. ``uint8`` depth/RGB
images) were therefore stored at 4x their real size, wasting GPU memory and, at
large environment counts, causing out-of-memory failures. Preserving the native
dtype keeps ``uint8`` observations as ``uint8`` in the buffer.

For the common AMP case where all observations are ``float32`` this helper is a
no-op, and on ``rsl-rl-lib`` >= 5.4.0 it is a no-op as well (the native storage
already preserves dtypes).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage

__all__ = ["build_rollout_storage", "preserve_observation_dtypes"]


def preserve_observation_dtypes(
    storage: RolloutStorage, obs_prototype: TensorDict
) -> RolloutStorage:
    """Ensure the rollout observation buffer matches the prototype dtypes.

    Reallocates only the observation groups whose buffer dtype differs from the
    prototype (i.e. those the underlying storage upcast to ``float32``). Groups
    that already match are left untouched, so this is a no-op on dtype-preserving
    ``rsl-rl-lib`` versions and for ``float32``-only observations.

    Parameters
    ----------
    storage : RolloutStorage
        Freshly constructed storage whose ``observations`` buffer may have been
        allocated with a coerced dtype.
    obs_prototype : TensorDict
        The observation TensorDict used to size the storage; carries the desired
        per-group dtypes.

    Returns
    -------
    RolloutStorage
        The same storage instance, with dtype-corrected observation buffers.
    """
    observations = storage.observations
    for key, proto in obs_prototype.items():
        proto_dtype = getattr(proto, "dtype", None)
        if proto_dtype is None:
            continue
        buffer = observations[key]
        if buffer.dtype == proto_dtype:
            continue
        observations.set(
            key,
            torch.zeros(buffer.shape, dtype=proto_dtype, device=buffer.device),
        )
    return storage


def build_rollout_storage(
    *,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape,
    device: str,
    training_type: str = "rl",
) -> RolloutStorage:
    """Build an ``rsl_rl`` ``RolloutStorage`` with observation dtypes preserved.

    Drop-in replacement for constructing ``RolloutStorage`` directly. See the
    module docstring for the motivation.
    """
    storage = RolloutStorage(
        training_type=training_type,
        num_envs=num_envs,
        num_transitions_per_env=num_transitions_per_env,
        obs=obs,
        actions_shape=actions_shape,
        device=device,
    )
    return preserve_observation_dtypes(storage, obs)
