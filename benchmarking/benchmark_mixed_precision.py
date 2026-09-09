# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Benchmark the opt-in bfloat16 mixed-precision path in ``AMP_PPO.update``.

This ports the idea behind rsl-rl-lib #219 ("Add opt-in bfloat16 mixed precision
to PPO"): the actor/critic/discriminator forward passes and losses inside
``AMP_PPO.update`` can run under ``torch.amp.autocast(dtype=bfloat16)`` when
``use_mixed_precision=True``. Default (off) keeps full FP32 numerics.

The script:
  1. builds a realistic AMP_PPO (humanoid-scale actor/critic + discriminator);
  2. prints an equivalence table (FP32 vs bf16 losses on identical data);
  3. times ``update()`` with mixed precision off vs on (warmup + CUDA sync).

Run:
    python benchmarking/benchmark_mixed_precision.py
"""

from __future__ import annotations

import statistics
import time
from importlib.metadata import version

import torch
from tensordict import TensorDict

from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils._compat import RSL_RL_V4_PLUS, resolve_obs_groups

# =============================================
# CONFIGURATION (humanoid-scale AMP defaults)
# =============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ENVS = 4096
HORIZON = 24
NUM_ACTIONS = 12
POLICY_DIM = 48
CRITIC_DIM = 60
AMP_DIM = 33
ACTOR_HIDDEN = [512, 256, 128]
CRITIC_HIDDEN = [512, 256, 128]
DISC_HIDDEN = [1024, 512]
NUM_LEARNING_EPOCHS = 5
NUM_MINI_BATCHES = 4

WARMUP = 3
REPEATS = 15


class _StubAMPData:
    """Minimal stand-in for AMPLoader: yields random (state, next_state) pairs."""

    def __init__(self, dim: int, device: str) -> None:
        self.dim = dim
        self.device = device

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            yield (
                torch.randn(mini_batch_size, self.dim, device=self.device),
                torch.randn(mini_batch_size, self.dim, device=self.device),
            )


def _make_obs(n: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(n, POLICY_DIM, device=DEVICE),
            "critic": torch.randn(n, CRITIC_DIM, device=DEVICE),
            "amp": torch.randn(n, AMP_DIM, device=DEVICE),
        },
        batch_size=[n],
        device=DEVICE,
    )


def _build_algo(use_mixed_precision: bool, seed: int) -> AMP_PPO:
    from rsl_rl.modules import ActorCritic

    torch.manual_seed(seed)
    obs = _make_obs(NUM_ENVS)
    obs_groups = resolve_obs_groups(
        obs, {"actor": ["policy"], "critic": ["critic"]}, ["critic"]
    )
    actor_critic = ActorCritic(
        obs,
        obs_groups,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=ACTOR_HIDDEN,
        critic_hidden_dims=CRITIC_HIDDEN,
    ).to(DEVICE)
    discriminator = Discriminator(
        input_dim=AMP_DIM * 2,
        hidden_layer_sizes=DISC_HIDDEN,
        reward_scale=1.0,
        device=DEVICE,
    ).to(DEVICE)
    algo = AMP_PPO(
        discriminator=discriminator,
        amp_data=_StubAMPData(AMP_DIM, DEVICE),
        actor_critic=actor_critic,
        num_learning_epochs=NUM_LEARNING_EPOCHS,
        num_mini_batches=NUM_MINI_BATCHES,
        schedule="fixed",
        device=DEVICE,
        use_mixed_precision=use_mixed_precision,
    )
    algo.init_storage(NUM_ENVS, HORIZON, obs, (NUM_ACTIONS,))
    return algo


def _fill_rollout(algo: AMP_PPO, seed: int) -> None:
    torch.manual_seed(seed)
    for _ in range(HORIZON):
        obs = _make_obs(NUM_ENVS)
        algo.act(obs)
        algo.act_amp(obs["amp"])
        next_obs = _make_obs(NUM_ENVS)
        rewards = torch.randn(NUM_ENVS, 1, device=DEVICE)
        dones = (torch.rand(NUM_ENVS, 1, device=DEVICE) < 0.02).float()
        algo.process_env_step(next_obs, rewards, dones, {})
        algo.process_amp_step(next_obs["amp"])
    algo.compute_returns(_make_obs(NUM_ENVS))


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _time_updates(use_mixed_precision: bool) -> list[float]:
    """Time update() over several iterations, refilling the storage untimed."""
    algo = _build_algo(use_mixed_precision, seed=0)
    times: list[float] = []
    for i in range(WARMUP + REPEATS):
        _fill_rollout(algo, seed=100 + i)  # untimed
        _sync()
        t0 = time.perf_counter()
        algo.update()
        _sync()
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            times.append(dt)
    return times


def _equivalence() -> None:
    """Run one update() FP32 vs bf16 on identical data and print the losses."""
    names = [
        "value_loss",
        "surrogate_loss",
        "amp_loss",
        "grad_pen_loss",
        "policy_pred",
        "expert_pred",
        "acc_policy",
        "acc_expert",
        "kl",
        "symmetry_loss",
    ]

    def run(mixed: bool):
        algo = _build_algo(mixed, seed=42)
        _fill_rollout(algo, seed=43)
        torch.manual_seed(44)  # identical mini-batch + AMP sampling for both
        return algo.update()

    off = run(False)
    on = run(True)
    print("\n== Equivalence (one update on identical data) ==")
    print(f"{'metric':<16}{'fp32 (off)':>14}{'bf16 (on)':>14}{'abs diff':>12}")
    worst = 0.0
    for name, a, b in zip(names, off, on):
        worst = max(worst, abs(a - b))
        print(f"{name:<16}{a:>14.6f}{b:>14.6f}{abs(a - b):>12.6f}")
    print(f"\n=> max abs difference across all metrics: {worst:.6f}")


def main() -> None:
    if RSL_RL_V4_PLUS:
        print("This benchmark harness targets the v3 ActorCritic path.")
    print("\n[AMP_PPO Mixed-Precision Benchmark]")
    print(f"device={DEVICE}  torch={torch.__version__}  rsl-rl-lib=={version('rsl-rl-lib')}")
    if DEVICE == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(
        f"num_envs={NUM_ENVS} horizon={HORIZON} -> {NUM_ENVS * HORIZON} transitions; "
        f"{NUM_LEARNING_EPOCHS} epochs x {NUM_MINI_BATCHES} mini-batches"
    )
    print(
        f"actor/critic hidden={ACTOR_HIDDEN}, discriminator hidden={DISC_HIDDEN}, amp_dim={AMP_DIM}"
    )

    _equivalence()

    off = _time_updates(use_mixed_precision=False)
    on = _time_updates(use_mixed_precision=True)
    off_ms = statistics.median(off) * 1e3
    on_ms = statistics.median(on) * 1e3

    print("\n== Speed (median of "
          f"{REPEATS} timed update() calls, {WARMUP} warmup) ==")
    print(f"fp32 (use_mixed_precision=False): {off_ms:8.3f} ms/update")
    print(f"bf16 (use_mixed_precision=True) : {on_ms:8.3f} ms/update")
    if on_ms > 0:
        print(f"\n=> speedup: {off_ms / on_ms:.2f}x  ({off_ms - on_ms:+.3f} ms/update)")


if __name__ == "__main__":
    main()
