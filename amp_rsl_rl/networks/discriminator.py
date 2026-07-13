# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from amp_rsl_rl.utils._compat import EmpiricalNormalization
from amp_rsl_rl.utils.motion_loader import _call_augmentation_func
from amp_rsl_rl.utils._compat import RSL_RL_V3_3_PLUS

if RSL_RL_V3_3_PLUS:
    from rsl_rl.utils import resolve_callable
else:
    from rsl_rl.utils import string_to_callable as resolve_callable


class Discriminator(nn.Module):
    """Discriminator implements the discriminator network for the AMP algorithm.

    This network is trained to distinguish between expert and policy-generated data.
    It also provides reward signals for the policy through adversarial learning.

    Args:
        input_dim (int): Dimension of the concatenated input state (state + next state).
        hidden_layer_sizes (list): List of hidden layer sizes.
        reward_scale (float): Scale factor for the computed reward.
        reward_clamp_epsilon (float): Numerical epsilon used when clamping rewards.
        device (str | torch.device): Device to run the model on.
        loss_type (str): Type of loss function to use ('BCEWithLogits' or 'Wasserstein').
        eta_wgan (float): Scaling factor for the Wasserstein loss (if used).
        use_minibatch_std (bool): Whether to use minibatch standard deviation in the network
        empirical_normalization (bool): Whether to normalize AMP observations empirically before scoring.
        num_skills (int): Number of commanded skills. ``0`` (default) restores the
            original unconditional single-head discriminator. When ``> 0`` the AMP
            observation is expected to carry a skill one-hot in its last
            ``num_skills`` dimensions, and the network becomes a shared trunk with
            ``num_skills`` linear heads (one per skill).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1.0e-4,
        device: str | torch.device = "cpu",
        loss_type: str = "BCEWithLogits",
        eta_wgan: float = 0.3,
        use_minibatch_std: bool = True,
        empirical_normalization: bool = False,
        symmetry_cfg: Optional[Dict[str, Any]] = None,
        num_skills: int = 0,
    ):
        super().__init__()

        self.device = torch.device(device)
        # ``input_dim`` keeps meaning ``2 * D_aug`` (state + next_state, each
        # carrying the skill one-hot). AMP_PPO reads it to size the replay buffer,
        # so its meaning must NOT change.
        self.input_dim = input_dim
        self.num_skills = num_skills
        self.obs_dim = input_dim // 2  # D_aug (physical dims + skill one-hot)
        self.phys_dim = self.obs_dim - num_skills  # D_phys (physical dims only)
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        layers = []
        # The skill one-hot never enters the trunk: only the physical dims of both
        # the current and next observation do.
        curr_in_dim = 2 * self.phys_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        final_in_dim = hidden_layer_sizes[-1] + (1 if use_minibatch_std else 0)
        # Shared trunk + K linear heads (a single head when num_skills == 0).
        self.linear = nn.Linear(final_in_dim, max(1, num_skills))

        self.empirical_normalization = empirical_normalization
        # Never normalize the one-hot dims: the normalizer operates on D_phys only.
        if empirical_normalization:
            self.amp_normalizer = EmpiricalNormalization(shape=[self.phys_dim])
        else:
            self.amp_normalizer = nn.Identity()

        self.to(self.device)
        self.train()
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type if loss_type is not None else "BCEWithLogits"
        if self.loss_type == "BCEWithLogits":
            self.loss_fun = torch.nn.BCEWithLogitsLoss()
        elif self.loss_type == "Wasserstein":
            self.loss_fun = None
            self.eta_wgan = eta_wgan
            print("The Wasserstein-like loss is experimental")
        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported types are 'BCEWithLogits' and 'Wasserstein'."
            )

        # ─── Check symmetry augmentation configuration ───
        if symmetry_cfg is not None:
            aug_fn = symmetry_cfg.get("amp_dataset_augmentation_func", None)
            if isinstance(aug_fn, str):
                symmetry_cfg["amp_dataset_augmentation_func"] = resolve_callable(aug_fn)
            aug_fn = symmetry_cfg.get("amp_dataset_augmentation_func", None)
            if aug_fn is not None and not callable(aug_fn):
                raise ValueError(
                    f"Discriminator symmetry augmentation function must be callable. Got: {aug_fn} of type {type(aug_fn)}"
                )
        self.symmetry_cfg = symmetry_cfg
        # ─────────────────────────────────────────────────

    def apply_symmetry(
        self, tensor: torch.Tensor, obs_type: str = "amp"
    ) -> torch.Tensor:
        """Apply the configured symmetry (mirror) augmentation to an AMP tensor.

        Layout note (skill-conditioned discriminator)
        ----------------------------------------------
        When ``num_skills > 0`` the tensor handed to the augmentation function is
        ``[*, D_phys + K]``: the physical AMP observation followed by the skill
        one-hot occupying the last ``K`` dimensions. This is exactly the same
        layout that :meth:`AMPLoader._apply_symmetry` sees at dataset-build time,
        so a single mirror function serves both paths. The mirror function must
        handle the one-hot tail explicitly:

        - Chirally-symmetric skills (forward walk mirrors to forward walk,
          backward to backward): pass the one-hot through unchanged.
        - Chirally-paired skills (e.g. separate ``turn_left`` / ``turn_right``):
          the mirror must permute those two one-hot entries, because a mirrored
          left turn is a right turn. A single combined ``turn_in_place`` skill
          passes through unchanged.

        Getting this wrong silently trains heads on mirrored data belonging to the
        wrong skill.
        """
        if self.symmetry_cfg is None or not self.symmetry_cfg.get(
            "use_data_augmentation", False
        ):
            return tensor

        fn = self.symmetry_cfg.get("amp_dataset_augmentation_func", None)
        if fn is None:
            raise ValueError(
                "Symmetry configuration specifies use_data_augmentation=True but no amp_dataset_augmentation_func provided for the discriminator."
            )

        augmented, _ = _call_augmentation_func(fn, obs=tensor, obs_type=obs_type)

        return augmented if augmented is not None else tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x (Tensor): Input tensor ``[batch, 2 * D_aug]`` (state + next_state,
                each carrying its skill one-hot when ``num_skills > 0``).

        Returns:
            Tensor: Discriminator output logits ``[batch, 1]``. When
                ``num_skills > 0`` the logit is taken from the head selected by
                the **current state's** skill one-hot; ``next_state``'s skill dims
                are ignored.
        """

        # Split state and next_state. Only the physical dims are normalized and
        # fed to the trunk; the one-hot never enters the network.
        state, next_state = torch.split(x, self.obs_dim, dim=-1)

        s_phys = state[..., : self.phys_dim]
        ns_phys = next_state[..., : self.phys_dim]

        s_phys = self.amp_normalizer(s_phys)
        ns_phys = self.amp_normalizer(ns_phys)

        h = self.trunk(torch.cat([s_phys, ns_phys], dim=-1))
        if self.use_minibatch_std:
            s = self._minibatch_std_scalar(h)
            h = torch.cat([h, s], dim=-1)

        logits = self.linear(h)  # [B, K] (or [B, 1] when num_skills == 0)
        if self.num_skills > 0:
            skill_idx = state[..., self.phys_dim :].argmax(dim=-1, keepdim=True)
            logits = logits.gather(1, skill_idx)  # [B, 1]
        return logits

    def _minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor:
        """Mean over feature-wise std across the batch; shape (B,1)."""
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        s = h.float().std(dim=0, unbiased=False).mean()
        return s.expand(h.shape[0], 1).to(h.dtype)

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts reward based on discriminator output using a log-style formulation.

        Args:
            state (Tensor): Current state tensor.
            next_state (Tensor): Next state tensor.

        Returns:
            Tensor: Computed adversarial reward.
        """
        with torch.no_grad():

            # No need to normalize here as normalization is done in forward()
            discriminator_logit = self.forward(torch.cat([state, next_state], dim=-1))

            if self.loss_type == "Wasserstein":
                discriminator_logit = torch.tanh(self.eta_wgan * discriminator_logit)
                return self.reward_scale * torch.exp(discriminator_logit).squeeze()
            # softplus(logit) == -log(1 - sigmoid(logit))
            reward = F.softplus(discriminator_logit)
            reward = self.reward_scale * reward
            return reward.squeeze()

    def policy_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the discriminator when classifying policy-generated transitions.
        Uses binary cross-entropy loss where the target label for policy transitions is 0.

        Parameters
        ----------
        discriminator_output : torch.Tensor
            The raw logits output from the discriminator for policy data.

        Returns
        -------
        torch.Tensor
            The computed policy loss.
        """
        expected = torch.zeros_like(discriminator_output, device=self.device)
        return self.loss_fun(discriminator_output, expected)

    def expert_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the discriminator when classifying expert transitions.
        Uses binary cross-entropy loss where the target label for expert transitions is 1.

        Parameters
        ----------
        discriminator_output : torch.Tensor
            The raw logits output from the discriminator for expert data.

        Returns
        -------
        torch.Tensor
            The computed expert loss.
        """
        expected = torch.ones_like(discriminator_output, device=self.device)
        return self.loss_fun(discriminator_output, expected)

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Update empirical statistics using provided AMP batches.

        Each batch is a full ``[B, D_aug]`` observation; only the physical dims
        (the first ``D_phys`` columns) are used to update the normalizer. The
        skill one-hot is a near-constant channel and must never be normalized.
        """
        if not self.empirical_normalization:
            return
        with torch.no_grad():
            for batch in batches:
                self.amp_normalizer.update(batch[..., : self.phys_dim])

    def compute_loss(
        self,
        policy_d,
        expert_d,
        sample_amp_expert,
        sample_amp_policy,
        lambda_: float = 10,
    ):

        # ``policy_d`` / ``expert_d`` arrive already head-selected as [B, 1], so
        # the BCE terms are unchanged. ``sample_amp_*`` are full [B, D_aug]
        # tuples (state, next_state); the gradient penalty slices off the skill
        # one-hot, normalizes only the physical dims and selects the correct head
        # internally, so we pass the raw tuples straight through.
        grad_pen_loss = self.compute_grad_pen(
            expert_states=sample_amp_expert,
            policy_states=sample_amp_policy,
            lambda_=lambda_,
        )
        if self.loss_type == "BCEWithLogits":
            expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
            policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
            # AMP loss is the average of expert and policy losses.
            amp_loss = 0.5 * (expert_loss + policy_loss)
        elif self.loss_type == "Wasserstein":
            amp_loss = self.wgan_loss(policy_d=policy_d, expert_d=expert_d)
        return amp_loss, grad_pen_loss

    def compute_grad_pen(
        self,
        expert_states: tuple[torch.Tensor, torch.Tensor],
        policy_states: tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10,
    ) -> torch.Tensor:
        """Computes the gradient penalty used to regularize the discriminator.

        Args:
            expert_states (tuple[Tensor, Tensor]): A tuple containing batches of
                expert states and expert next states, each ``[B, D_aug]``.
            policy_states (tuple[Tensor, Tensor]): A tuple containing batches of
                policy states and policy next states, each ``[B, D_aug]``.
            lambda_ (float): Penalty coefficient.

        Returns:
            Tensor: Gradient penalty value.

        Notes:
            When ``num_skills > 0`` the gradient is taken w.r.t. the physical dims
            only (the gradient of a score w.r.t. a one-hot input is meaningless),
            and the head is selected from the **expert's** skill one-hot.
        """
        if self.loss_type == "Wasserstein":
            if self.num_skills > 0:
                raise NotImplementedError(
                    "Gradient penalty for the Wasserstein loss is not supported "
                    "when num_skills > 0: interpolating skill one-hots between an "
                    "expert sample of one skill and a policy sample of another is "
                    "ill-defined. Use loss_type='BCEWithLogits' for skill-"
                    "conditioned training."
                )
            expert = torch.cat([self.amp_normalizer(s) for s in expert_states], -1)
            policy = torch.cat([self.amp_normalizer(s) for s in policy_states], -1)
            alpha = torch.rand(expert.size(0), 1, device=expert.device)
            alpha = alpha.expand_as(expert)
            data = alpha * expert + (1 - alpha) * policy
            data = data.detach().requires_grad_(True)
            h = self.trunk(data)
            if self.use_minibatch_std:
                with torch.no_grad():
                    s = self._minibatch_std_scalar(h)
                h = torch.cat([h, s], dim=-1)
            scores = self.linear(h)
            grad = autograd.grad(
                outputs=scores,
                inputs=data,
                grad_outputs=torch.ones_like(scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()
        elif self.loss_type == "BCEWithLogits":
            expert_state, expert_next_state = expert_states
            # Slice off the skill one-hot and normalize only the physical dims.
            expert_s_phys = self.amp_normalizer(expert_state[..., : self.phys_dim])
            expert_ns_phys = self.amp_normalizer(
                expert_next_state[..., : self.phys_dim]
            )
            # R1 regularizer on REAL: 0.5 * lambda * ||∇_x D(x_real)||^2, where x
            # is the physical dims only.
            data = (
                torch.cat([expert_s_phys, expert_ns_phys], dim=-1)
                .detach()
                .requires_grad_(True)
            )
            # Compute D(x_real) with minibatch-std DETACHED,
            # so gradients are w.r.t. the sample itself, not the batch statistics.
            h = self.trunk(data)
            if self.use_minibatch_std:
                with torch.no_grad():
                    s = self._minibatch_std_scalar(h)
                h = torch.cat([h, s], dim=-1)
            logits = self.linear(h)
            if self.num_skills > 0:
                skill_idx = expert_state[..., self.phys_dim :].argmax(
                    dim=-1, keepdim=True
                )
                scores = logits.gather(1, skill_idx)
            else:
                scores = logits

            grad = autograd.grad(
                outputs=scores.sum(),
                inputs=data,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return 0.5 * lambda_ * (grad.pow(2).sum(dim=1)).mean()

        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported types are 'BCEWithLogits' and 'Wasserstein'."
            )

    def wgan_loss(self, policy_d, expert_d):
        """
        This loss function computes a modified Wasserstein loss for the discriminator.
        The original Wasserstein loss is D(policy) - D(expert), but here we apply a tanh
        transformation to the discriminator outputs scaled by eta_wgan. This helps in stabilizing the training.
        Args:
            policy_d (Tensor): Discriminator output for policy data.
            expert_d (Tensor): Discriminator output for expert data.
        """
        policy_d = torch.tanh(self.eta_wgan * policy_d)
        expert_d = torch.tanh(self.eta_wgan * expert_d)
        return policy_d.mean() - expert_d.mean()
