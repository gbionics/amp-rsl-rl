# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the opt-in bfloat16 mixed-precision path in AMP_PPO.

This backports the idea behind rsl-rl-lib #219 ("Add opt-in bfloat16 mixed
precision to PPO"): the network forward passes and loss computation inside
``AMP_PPO.update`` can run under ``torch.amp.autocast(dtype=bfloat16)`` when
``use_mixed_precision=True``.

The feature is opt-in and defaults to ``False``. When disabled, ``autocast`` is a
documented no-op, so the FP32 numerics are *bit-identical* to before the change.
These tests verify:

1. the flag defaults to ``False``;
2. wrapping the discriminator forward + loss in ``autocast(enabled=False)`` is
   bit-identical to not wrapping it at all (the equivalence guarantee);
3. a full ``update()`` with the flag off is deterministic;
4. a full ``update()`` with the flag on stays finite and close to the FP32 run.

Run with:
    python -m pytest tests/test_mixed_precision.py -v
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils._compat import RSL_RL_V4_PLUS, resolve_obs_groups

# ── Test devices ─────────────────────────────────────────────────────────────

_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")

# Small, fast, representative dimensions.
NUM_ENVS = 64
HORIZON = 4
NUM_ACTIONS = 4
POLICY_DIM = 12
CRITIC_DIM = 16
AMP_DIM = 8


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


def _make_obs(n: int, device: str) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(n, POLICY_DIM, device=device),
            "critic": torch.randn(n, CRITIC_DIM, device=device),
            "amp": torch.randn(n, AMP_DIM, device=device),
        },
        batch_size=[n],
        device=device,
    )


def _build_algo(device: str, use_mixed_precision: bool, seed: int) -> AMP_PPO:
    """Construct a full AMP_PPO on the installed rsl-rl version."""
    if RSL_RL_V4_PLUS:  # pragma: no cover - depends on installed rsl-rl-lib
        pytest.skip("Test harness builds the v3 ActorCritic path; skipping on rsl-rl>=4.")

    from rsl_rl.modules import ActorCritic

    torch.manual_seed(seed)
    obs = _make_obs(NUM_ENVS, device)
    obs_groups = resolve_obs_groups(
        obs, {"actor": ["policy"], "critic": ["critic"]}, ["critic"]
    )
    actor_critic = ActorCritic(
        obs,
        obs_groups,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
    ).to(device)
    discriminator = Discriminator(
        input_dim=AMP_DIM * 2,
        hidden_layer_sizes=[32, 32],
        reward_scale=1.0,
        device=device,
    ).to(device)
    algo = AMP_PPO(
        discriminator=discriminator,
        amp_data=_StubAMPData(AMP_DIM, device),
        actor_critic=actor_critic,
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",  # isolate the precision effect from LR-schedule drift
        device=device,
        use_mixed_precision=use_mixed_precision,
    )
    algo.init_storage(NUM_ENVS, HORIZON, obs, (NUM_ACTIONS,))
    return algo


def _fill_rollout(algo: AMP_PPO, device: str, seed: int) -> None:
    """Simulate a rollout so the storage and AMP replay buffer are populated."""
    torch.manual_seed(seed)
    for _ in range(HORIZON):
        obs = _make_obs(NUM_ENVS, device)
        algo.act(obs)
        algo.act_amp(obs["amp"])
        next_obs = _make_obs(NUM_ENVS, device)
        rewards = torch.randn(NUM_ENVS, 1, device=device)
        dones = (torch.rand(NUM_ENVS, 1, device=device) < 0.05).float()
        algo.process_env_step(next_obs, rewards, dones, {})
        algo.process_amp_step(next_obs["amp"])
    algo.compute_returns(_make_obs(NUM_ENVS, device))


def _run_one_update(device: str, use_mixed_precision: bool, seed: int):
    algo = _build_algo(device, use_mixed_precision, seed)
    _fill_rollout(algo, device, seed + 1)
    # Seed identically right before update() so mini-batch shuffling and AMP
    # sampling are the same regardless of precision: the only difference is bf16.
    torch.manual_seed(seed + 2)
    return algo.update()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_default_flag_is_off():
    """Mixed precision must default to off so behaviour is unchanged by default."""
    import inspect

    default = inspect.signature(AMP_PPO.__init__).parameters["use_mixed_precision"].default
    assert default is False


@pytest.mark.parametrize("device", _devices)
def test_autocast_disabled_is_bit_identical(device: str):
    """autocast(enabled=False) around the discriminator must be a bit-exact no-op.

    This is the core equivalence guarantee: with ``use_mixed_precision=False`` the
    only added code is ``with torch.amp.autocast(enabled=False)``, which PyTorch
    documents as a no-op. Here we prove it holds for the discriminator forward +
    R1 gradient-penalty loss on this hardware.
    """
    torch.manual_seed(0)
    disc = Discriminator(
        input_dim=AMP_DIM * 2,
        hidden_layer_sizes=[64, 32],
        reward_scale=1.0,
        device=device,
    ).to(device)

    def forward_and_loss(use_autocast: bool):
        torch.manual_seed(1)
        ps = torch.randn(128, AMP_DIM, device=device)
        pn = torch.randn(128, AMP_DIM, device=device)
        es = torch.randn(128, AMP_DIM, device=device)
        en = torch.randn(128, AMP_DIM, device=device)
        di = torch.cat((torch.cat([ps, pn], -1), torch.cat([es, en], -1)), 0)
        ctx = (
            torch.amp.autocast(device_type=torch.device(device).type, enabled=False)
            if use_autocast
            else _nullcontext()
        )
        with ctx:
            out = disc(di)
            pd, ed = out[:128], out[128:]
            amp_loss, gp = disc.compute_loss(pd, ed, (es, en), (ps, pn), lambda_=10)
        return out, amp_loss, gp

    out_a, amp_a, gp_a = forward_and_loss(True)
    out_b, amp_b, gp_b = forward_and_loss(False)

    assert out_a.dtype == torch.float32
    assert torch.equal(out_a, out_b)
    assert torch.equal(amp_a, amp_b)
    assert torch.equal(gp_a, gp_b)


@pytest.mark.parametrize("device", _devices)
def test_update_off_is_deterministic(device: str):
    """Two identical off-runs must produce identical losses (no added nondeterminism)."""
    out1 = _run_one_update(device, use_mixed_precision=False, seed=7)
    out2 = _run_one_update(device, use_mixed_precision=False, seed=7)
    for a, b in zip(out1, out2):
        assert a == pytest.approx(b, rel=0, abs=0)


@pytest.mark.parametrize("device", _devices)
def test_update_on_is_finite_and_close(device: str):
    """The mixed-precision run stays finite and close to the FP32 run."""
    off = _run_one_update(device, use_mixed_precision=False, seed=11)
    on = _run_one_update(device, use_mixed_precision=True, seed=11)

    # Names for readable diagnostics on failure.
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
    for name, a, b in zip(names, off, on):
        assert b == b, f"{name} is NaN under mixed precision"  # NaN check
        assert abs(b) != float("inf"), f"{name} is inf under mixed precision"
        # bfloat16 has ~2-3 significant digits; allow a generous but finite band.
        assert abs(a - b) <= 0.15 * (abs(a) + 1e-3) + 0.05, (
            f"{name}: fp32={a:.5f} vs bf16={b:.5f} diverged too much"
        )


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
