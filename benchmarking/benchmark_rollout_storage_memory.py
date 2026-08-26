# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Quantify the rollout-buffer memory saved by preserving observation dtypes.

This reproduces the effect behind the rsl-rl-lib v5.4.0 speed/memory improvement
(upstream PR #212) that ``amp_rsl_rl.storage.build_rollout_storage`` backports to
all supported versions: non-float observations (e.g. ``uint8`` camera frames)
are kept at their native dtype instead of being upcast to ``float32``.

Run:
    python benchmarking/benchmark_rollout_storage_memory.py
"""

import torch
from tensordict import TensorDict
from importlib.metadata import version

from rsl_rl.storage import RolloutStorage
from amp_rsl_rl.storage import build_rollout_storage

# =============================================
# CONFIGURATION
# =============================================
device_str = "cuda" if torch.cuda.is_available() else "cpu"
num_envs = 4096
horizon = 24
proprio_dim = 48
critic_dim = 60
camera_shape = (1, 58, 87)  # single-channel depth image, uint8


def _obs_prototype() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(num_envs, proprio_dim, dtype=torch.float32),
            "critic": torch.zeros(num_envs, critic_dim, dtype=torch.float32),
            "camera": torch.zeros(num_envs, *camera_shape, dtype=torch.uint8),
        },
        batch_size=[num_envs],
    )


def _buffer_mb(storage) -> float:
    total = sum(v.element_size() * v.nelement() for v in storage.observations.values())
    return total / 1e6


def main() -> None:
    device = torch.device(device_str)
    print(f"\n[RolloutStorage Memory Benchmark] Device: {device}")
    print(f"rsl-rl-lib=={version('rsl-rl-lib')}")
    print(f"num_envs={num_envs}, horizon={horizon}, camera={camera_shape} (uint8)\n")

    # Native rsl_rl storage (dtype behaviour depends on installed version).
    native = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=horizon,
        obs=_obs_prototype(),
        actions_shape=(12,),
        device=device_str,
    )
    native_mb = _buffer_mb(native)
    native_dtypes = {k: str(v.dtype) for k, v in native.observations.items()}

    # amp_rsl_rl storage with the dtype-preserving backport applied.
    preserved = build_rollout_storage(
        num_envs=num_envs,
        num_transitions_per_env=horizon,
        obs=_obs_prototype(),
        actions_shape=(12,),
        device=device_str,
    )
    preserved_mb = _buffer_mb(preserved)
    preserved_dtypes = {k: str(v.dtype) for k, v in preserved.observations.items()}

    print(f"[native  RolloutStorage] dtypes={native_dtypes}")
    print(f"[native  RolloutStorage] obs buffer = {native_mb:8.2f} MB")
    print(f"[amp build_rollout_storage] dtypes={preserved_dtypes}")
    print(f"[amp build_rollout_storage] obs buffer = {preserved_mb:8.2f} MB")

    if preserved_mb < native_mb:
        print(f"\n=> {native_mb / preserved_mb:.2f}x less rollout-buffer memory "
              f"({native_mb - preserved_mb:.1f} MB saved).")
    else:
        print("\n=> Installed rsl-rl-lib already preserves dtypes; backport is a no-op.")


if __name__ == "__main__":
    main()
