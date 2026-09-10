# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
)
from rsl_rl.extensions import resolve_rnd_config

from amp_rsl_rl.storage import ReplayBuffer
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import AMPLoader, VelocityRepresentation, _call_augmentation_func
from amp_rsl_rl.utils.registry import resolve_amp_callable, resolve_amp_class


class AMP_PPO(PPO):
    """
    AMP_PPO implements Adversarial Motion Priors (AMP) combined with Proximal Policy Optimization (PPO).
    The implementation is on top of rsl-rl ``PPO``.

    This subclass augments PPO with an AMP discriminator that provides a style
    reward and is trained adversarially against expert motion data. It reuses
    the base algorithm's mixed-precision, adaptive learning rate, multi-GPU and
    ``torch.compile`` machinery while injecting the discriminator loss into the
    optimization step and mixing the style reward into the environment reward.

    The style reward is mixed inside :meth:`process_env_step` so the standard
    rollout loop remains agnostic to AMP:

        ``reward = (1 - style_weight) * task_reward + style_weight * style_reward``

    The actor, critic, storage, discriminator and expert-motion loader are built
    by :meth:`construct_algorithm`, the factory the runner invokes.

    Parameters
    ----------
    actor : MLPModel
        Policy model consuming the ``"actor"`` observation group.
    critic : MLPModel
        Value model consuming the ``"critic"`` observation group.
    storage : RolloutStorage
        Pre-built rsl-rl rollout storage for the on-policy transitions.
    discriminator : Discriminator
        AMP discriminator distinguishing expert vs policy motion pairs and
        producing the style reward.
    amp_data : AMPLoader
        Data loader that provides batches of expert motion transitions.
    style_weight : float, default=0.5
        Weight of the AMP style reward relative to the task reward when mixing.
    num_learning_epochs : int, default=5
        Number of passes over the rollout buffer per update.
    num_mini_batches : int, default=4
        Number of mini-batches to divide each epoch's data into.
    clip_param : float, default=0.2
        PPO clipping parameter that bounds the policy update step.
    gamma : float, default=0.99
        Discount factor.
    lam : float, default=0.95
        Lambda parameter for Generalized Advantage Estimation (GAE).
    value_loss_coef : float, default=1.0
        Coefficient for the value function loss term in the PPO loss.
    entropy_coef : float, default=0.01
        Coefficient for the entropy regularization term (encouraging exploration).
    learning_rate : float, default=1e-3
        Initial learning rate.
    max_grad_norm : float, default=1.0
        Maximum gradient norm for clipping gradients during backpropagation.
    optimizer : str, default="adam"
        Optimizer name resolved by :func:`rsl_rl.utils.resolve_optimizer`
        (e.g. ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``).
    use_clipped_value_loss : bool, default=True
        Enables the clipped value loss variant of PPO.
    schedule : str, default="adaptive"
        Either ``"fixed"`` or ``"adaptive"`` (KL-based learning-rate schedule).
    desired_kl : float, default=0.01
        Target KL divergence when using the adaptive schedule.
    normalize_advantage_per_mini_batch : bool, default=False
        Whether to normalize advantages within each mini-batch instead of over
        the whole rollout.
    use_mixed_precision : bool, default=False
        Enables ``torch.amp`` autocast (bfloat16) for the PPO forward/loss pass.
    device : str, default="cpu"
        Torch device used by the algorithm.
    rnd_cfg : dict | None, default=None
        Random Network Distillation configuration (``None`` disables RND).
    symmetry_cfg : dict | None, default=None
        Symmetry configuration enabling AMP data augmentation and mirror loss.
        Handled internally by AMP (its augmentation functions use a different
        signature than the base rsl-rl ``Symmetry`` extension).
    multi_gpu_cfg : dict | None, default=None
        Distributed-training configuration (``None`` disables multi-GPU).
    amp_replay_buffer_size : int, default=100_000
        Size of the replay buffer storing policy-generated AMP transitions.
    use_smooth_ratio_clipping : bool, default=False
        Enables smooth ratio clipping instead of the hard PPO clamp.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        style_weight: float = 0.5,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        rnd_cfg: Optional[dict] = None,
        symmetry_cfg: Optional[dict] = None,
        multi_gpu_cfg: Optional[dict] = None,
        amp_replay_buffer_size: int = 100000,
        use_smooth_ratio_clipping: bool = False,
    ) -> None:
        # AMP handles symmetry itself (its augmentation functions use a different
        # signature than the rsl-rl ``Symmetry`` extension), so we pass
        # ``symmetry_cfg=None`` to the base class and keep our own copy.
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            use_mixed_precision=use_mixed_precision,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=None,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.style_weight = style_weight
        self.use_smooth_ratio_clipping = use_smooth_ratio_clipping

        # Discriminator + expert data + policy replay buffer
        self.discriminator = discriminator.to(device)
        self.amp_data = amp_data
        obs_dim = self.discriminator.input_dim // 2
        self.amp_storage = ReplayBuffer(
            obs_dim=obs_dim, buffer_size=amp_replay_buffer_size, device=device
        )
        self.amp_transition = RolloutStorage.Transition()

        # Rebuild the optimizer to jointly train actor, critic and discriminator,
        # keeping the AMP discriminator's per-group weight decay (trunk vs head).
        opt_class = resolve_optimizer(optimizer)
        self.optimizer = opt_class(
            [
                {"params": list(self.actor.parameters()), "name": "actor"},
                {"params": list(self.critic.parameters()), "name": "critic"},
                {
                    "params": list(self.discriminator.trunk.parameters()),
                    "weight_decay": 10e-4,
                    "name": "amp_trunk",
                },
                {
                    "params": list(self.discriminator.linear.parameters()),
                    "weight_decay": 10e-2,
                    "name": "amp_head",
                },
            ],
            lr=learning_rate,
        )

        # AMP-custom symmetry handling (separate from the base ``Symmetry`` ext).
        self.symmetry_cfg = self._resolve_symmetry_cfg(symmetry_cfg)

        # AMP reward statistics (means) exposed for logging.
        self.style_reward = torch.zeros((), device=device)
        self.task_reward = torch.zeros((), device=device)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _flatten_amp_obs(amp_obs) -> torch.Tensor:
        """Flatten an AMP observation group into a single 2D tensor."""
        if isinstance(amp_obs, torch.Tensor):
            return amp_obs
        if hasattr(amp_obs, "keys"):
            if "joint_pos" in amp_obs and "joint_vel" in amp_obs:
                return torch.cat([amp_obs["joint_pos"], amp_obs["joint_vel"]], dim=-1)
            keys = sorted(amp_obs.keys())
            return torch.cat([amp_obs[k] for k in keys], dim=-1)
        raise TypeError(f"Unsupported AMP observation type: {type(amp_obs)}")

    def _resolve_symmetry_cfg(
        self, symmetry_cfg: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Validate the AMP symmetry configuration and resolve its callables."""
        if symmetry_cfg is None:
            return None
        aug_fn = symmetry_cfg.get("data_augmentation_func", None)
        if isinstance(aug_fn, str):
            symmetry_cfg["data_augmentation_func"] = resolve_callable(aug_fn)
        aug_fn = symmetry_cfg.get("data_augmentation_func", None)
        if aug_fn is not None and not callable(aug_fn):
            raise ValueError(
                f"Symmetry data_augmentation_func is not callable: {aug_fn}"
            )
        if getattr(self.actor, "is_recurrent", False):
            raise ValueError(
                "Symmetry augmentation is not supported for recurrent policies in AMP_PPO."
            )
        return symmetry_cfg

    def _augment_batch_size(
        self, original_size: int, augmented: Optional[torch.Tensor]
    ) -> int:
        if augmented is None or original_size == 0:
            return 1
        if augmented.shape[0] % original_size != 0:
            raise ValueError(
                "Symmetry augmentation returned an incompatible batch size."
                f" Original={original_size}, augmented={augmented.shape[0]}"
            )
        return augmented.shape[0] // original_size

    def _repeat_along_batch(
        self, tensor: Optional[torch.Tensor], num_aug: int
    ) -> Optional[torch.Tensor]:
        if tensor is None or num_aug == 1:
            return tensor
        repeat_dims = [num_aug] + [1] * (tensor.dim() - 1)
        return tensor.repeat(*repeat_dims)

    def _apply_symmetry(
        self,
        *,
        obs: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        obs_type: Union[str, Sequence[str], None] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.symmetry_cfg is None:
            return obs, actions
        aug_fn = self.symmetry_cfg.get("data_augmentation_func", None)
        if aug_fn is None:
            return obs, actions
        aug_obs, aug_actions = _call_augmentation_func(
            aug_fn, obs=obs, actions=actions, obs_type=obs_type
        )
        return (
            aug_obs if aug_obs is not None else obs,
            aug_actions if aug_actions is not None else actions,
        )

    # ------------------------------------------------------------------ modes

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    # ------------------------------------------------------------ rollout hooks

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        # Stash the AMP observation to pair with the next-step observation.
        self.amp_transition.observations = self._flatten_amp_obs(obs["amp"])
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Dict[str, Any],
    ) -> None:
        next_amp_obs = self._flatten_amp_obs(obs["amp"])
        style_rewards = self.discriminator.predict_reward(
            self.amp_transition.observations, next_amp_obs
        )
        # Record means for logging before mixing.
        self.task_reward = rewards.mean().detach()
        self.style_reward = style_rewards.mean().detach()
        mixed_rewards = (
            1.0 - self.style_weight
        ) * rewards + self.style_weight * style_rewards
        # Store the (state, next_state) AMP pair for the discriminator update.
        self.amp_storage.insert(self.amp_transition.observations, next_amp_obs)
        self.amp_transition.clear()
        super().process_env_step(obs, mixed_rewards, dones, extras)

    # ------------------------------------------------------------------ update

    def update(self) -> Dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_accuracy_policy = 0.0
        mean_accuracy_expert = 0.0
        mean_accuracy_policy_elem = 0.0
        mean_accuracy_expert_elem = 0.0
        mean_kl_divergence = 0.0
        mean_symmetry_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        num_amp_batches = self.num_learning_epochs * self.num_mini_batches
        amp_mini_batch_size = (
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches
        )
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            num_mini_batch=num_amp_batches,
            mini_batch_size=amp_mini_batch_size,
            allow_replacement=True,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            num_amp_batches, amp_mini_batch_size
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            obs_batch = batch.observations
            actions_batch = batch.actions
            target_values_batch = batch.values
            advantages_batch = batch.advantages
            returns_batch = batch.returns
            old_actions_log_prob_batch = batch.old_actions_log_prob
            old_distribution_params_batch = batch.old_distribution_params
            masks_batch = batch.masks
            hidden_states_batch = batch.hidden_states
            hidden_state_actor, hidden_state_critic = (
                hidden_states_batch
                if hidden_states_batch is not None
                else (None, None)
            )

            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # PPO symmetry data augmentation
            if self.symmetry_cfg and self.symmetry_cfg.get(
                "use_data_augmentation", False
            ):
                aug_obs, aug_actions = self._apply_symmetry(
                    obs=obs_batch,
                    actions=actions_batch,
                    obs_type=["policy", "critic"],
                )
                num_aug = self._augment_batch_size(original_batch_size, aug_obs)
                obs_batch = aug_obs
                actions_batch = aug_actions
                old_actions_log_prob_batch = self._repeat_along_batch(
                    old_actions_log_prob_batch, num_aug
                )
                target_values_batch = self._repeat_along_batch(
                    target_values_batch, num_aug
                )
                advantages_batch = self._repeat_along_batch(advantages_batch, num_aug)
                returns_batch = self._repeat_along_batch(returns_batch, num_aug)

            with torch.amp.autocast(
                device_type=torch.device(self.device).type,
                enabled=self.use_mixed_precision,
                dtype=torch.bfloat16,
            ):
                self.actor(
                    obs_batch,
                    masks=masks_batch,
                    hidden_state=hidden_state_actor,
                    stochastic_output=True,
                )
                actions_log_prob_batch = self.actor.get_output_log_prob(actions_batch)
                value_batch = self.critic(
                    obs_batch, masks=masks_batch, hidden_state=hidden_state_critic
                )
                distribution_params = tuple(
                    p[:original_batch_size]
                    for p in self.actor.output_distribution_params
                )
                entropy_batch = self.actor.output_entropy[:original_batch_size]

                # Adaptive learning rate based on KL divergence
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(
                            old_distribution_params_batch, distribution_params
                        )
                        kl_mean = torch.mean(kl)
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(
                                kl_mean, op=torch.distributed.ReduceOp.SUM
                            )
                            kl_mean /= self.gpu_world_size
                        mean_kl_divergence += kl_mean.item()
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(
                                self.learning_rate, device=self.device
                            )
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(
                    actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
                )
                min_ = 1.0 - self.clip_param
                max_ = 1.0 + self.clip_param
                if self.use_smooth_ratio_clipping:
                    clipped_ratio = (
                        1
                        / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
                        * (max_ - min_)
                        + min_
                    )
                else:
                    clipped_ratio = torch.clamp(ratio, min_, max_)
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (
                        value_batch - target_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                ppo_loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                )

                rnd_loss = (
                    self.rnd.compute_loss(obs_batch[:original_batch_size])
                    if self.rnd
                    else None
                )

            # Mirror loss (kept outside autocast for numerical stability)
            symmetry_loss_value = torch.zeros((), device=self.device)
            if self.symmetry_cfg and self.symmetry_cfg.get("use_mirror_loss", False):
                if not self.symmetry_cfg.get("use_data_augmentation", False):
                    sym_obs_batch, _ = self._apply_symmetry(
                        obs=obs_batch[:original_batch_size],
                        actions=None,
                        obs_type="policy",
                    )
                else:
                    sym_obs_batch = obs_batch
                if sym_obs_batch is not None:
                    sym_obs_detached = sym_obs_batch.detach().clone()
                    mean_actions_batch = self.actor(
                        sym_obs_detached, stochastic_output=False
                    )
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, sym_actions = self._apply_symmetry(
                        obs=None, actions=action_mean_orig, obs_type="policy"
                    )
                    if sym_actions is None:
                        sym_actions = mean_actions_batch
                    symmetry_loss_value = torch.nn.functional.mse_loss(
                        mean_actions_batch[original_batch_size:],
                        sym_actions.detach()[original_batch_size:],
                    )
                    coeff = self.symmetry_cfg.get("mirror_loss_coeff", 0.0)
                    ppo_loss = ppo_loss + coeff * symmetry_loss_value

            # AMP discriminator loss (outside autocast; uses double backward)
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.symmetry_cfg and self.symmetry_cfg.get(
                "use_data_augmentation", False
            ):
                policy_state = self.discriminator.apply_symmetry(
                    policy_state, obs_type="amp"
                )
                policy_next_state = self.discriminator.apply_symmetry(
                    policy_next_state, obs_type="amp"
                )
                expert_state = self.discriminator.apply_symmetry(
                    expert_state, obs_type="amp"
                )
                expert_next_state = self.discriminator.apply_symmetry(
                    expert_next_state, obs_type="amp"
                )

            policy_state = policy_state.to(self.device)
            policy_next_state = policy_next_state.to(self.device)
            expert_state = expert_state.to(self.device)
            expert_next_state = expert_next_state.to(self.device)

            policy_state_raw = policy_state.detach().clone()
            policy_next_state_raw = policy_next_state.detach().clone()
            expert_state_raw = expert_state.detach().clone()
            expert_next_state_raw = expert_next_state.detach().clone()

            b_pol = policy_state.size(0)
            discriminator_input = torch.cat(
                (
                    torch.cat([policy_state, policy_next_state], dim=-1),
                    torch.cat([expert_state, expert_next_state], dim=-1),
                ),
                dim=0,
            )
            discriminator_output = self.discriminator(discriminator_input)
            policy_d, expert_d = (
                discriminator_output[:b_pol],
                discriminator_output[b_pol:],
            )
            amp_loss, grad_pen_loss = self.discriminator.compute_loss(
                policy_d=policy_d,
                expert_d=expert_d,
                sample_amp_expert=(expert_state, expert_next_state),
                sample_amp_policy=(policy_state, policy_next_state),
                lambda_=10,
            )

            loss = ppo_loss + amp_loss + grad_pen_loss

            # Optimization step (actor + critic + discriminator)
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self._reduce_amp_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            # Update the discriminator normalizer with raw (unnormalized) AMP obs.
            self.discriminator.update_normalization(
                expert_state_raw,
                expert_next_state_raw,
                policy_state_raw,
                policy_next_state_raw,
            )

            policy_d_prob = torch.sigmoid(policy_d)
            expert_d_prob = torch.sigmoid(expert_d)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d_prob.mean().item()
            mean_expert_pred += expert_d_prob.mean().item()
            mean_symmetry_loss += symmetry_loss_value.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            mean_accuracy_policy += torch.sum(
                torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
            ).item()
            mean_accuracy_expert += torch.sum(
                torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
            ).item()
            mean_accuracy_expert_elem += expert_d_prob.numel()
            mean_accuracy_policy_elem += policy_d_prob.numel()

        # Update policy/value (and RND) observation normalizers.
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_kl_divergence /= num_updates
        mean_symmetry_loss /= num_updates
        mean_accuracy_policy /= max(1, mean_accuracy_policy_elem)
        mean_accuracy_expert /= max(1, mean_accuracy_expert_elem)
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "grad_pen": mean_grad_pen_loss,
            "policy_pred": mean_policy_pred,
            "expert_pred": mean_expert_pred,
            "accuracy_policy": mean_accuracy_policy,
            "accuracy_expert": mean_accuracy_expert,
            "kl_divergence": mean_kl_divergence,
        }
        if self.symmetry_cfg is not None:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict

    # --------------------------------------------------------- save / load / mp

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        if getattr(self.discriminator, "empirical_normalization", False):
            saved_dict["amp_normalizer"] = self.discriminator.amp_normalizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: Optional[dict], strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        do_load_disc = True if load_cfg is None else load_cfg.get("discriminator", True)
        if do_load_disc and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=False
            )
            amp_normalizer = loaded_dict.get("amp_normalizer")
            if amp_normalizer is not None and getattr(
                self.discriminator, "empirical_normalization", False
            ):
                # Accept both a raw state_dict and a legacy module checkpoint.
                if hasattr(amp_normalizer, "state_dict"):
                    amp_normalizer = amp_normalizer.state_dict()
                self.discriminator.amp_normalizer.load_state_dict(amp_normalizer)
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        disc_params = [self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(disc_params, src=0)
        self.discriminator.load_state_dict(disc_params[0])

    def _reduce_amp_parameters(self) -> None:
        """All-reduce gradients of actor, critic and discriminator across GPUs."""
        params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.discriminator.parameters())
        )
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                offset += numel

    # ----------------------------------------------------------- construction

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "AMP_PPO":
        """Build the AMP_PPO algorithm (actor, critic, storage, discriminator)."""
        alg_class, alg_cfg = resolve_amp_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_amp_class(cfg["actor"])
        critic_class, critic_cfg = resolve_amp_class(cfg["critic"])

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if alg_cfg.get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups"), default_sets)

        # Resolve RND config (symmetry stays AMP-managed).
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        # Build actor and critic models
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
        ).to(device)
        print(f"Actor Model: {actor}")
        if alg_cfg.pop("share_cnn_encoders", None):
            critic_cfg["cnns"] = actor.cnns
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **critic_cfg
        ).to(device)
        print(f"Critic Model: {critic}")

        # Rollout storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # AMP components
        discriminator, amp_data = AMP_PPO._build_amp_components(
            obs, env, cfg, alg_cfg, device
        )

        alg: AMP_PPO = alg_class(
            actor,
            critic,
            storage,
            discriminator,
            amp_data,
            style_weight=cfg.get("style_weight", 0.5),
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    @staticmethod
    def _build_amp_components(
        obs: TensorDict, env: VecEnv, cfg: dict, alg_cfg: dict, device: str
    ) -> Tuple[Discriminator, AMPLoader]:
        """Construct the AMP discriminator and expert-motion loader from config."""
        dataset_cfg = cfg["dataset"]
        disc_cfg = dict(cfg["discriminator"])
        disc_class_name = disc_cfg.pop("class_name", None)
        disc_class = (
            resolve_amp_callable(disc_class_name)
            if disc_class_name is not None
            else Discriminator
        )

        amp_joint_names = dataset_cfg.get("amp_joint_names", None)
        if amp_joint_names is None:
            try:
                amp_joint_names = env.cfg.observations.amp.joint_pos.params[
                    "asset_cfg"
                ].joint_names
            except (AttributeError, KeyError, TypeError):
                warnings.warn(
                    "Could not resolve amp_joint_names from"
                    " env.cfg.observations.amp.joint_pos.params['asset_cfg'].joint_names."
                    " Falling back to None. Set 'amp_joint_names' in dataset_cfg"
                    " explicitly to silence this warning.",
                    stacklevel=2,
                )
                amp_joint_names = None

        sim_cfg = getattr(env.cfg, "sim", None)
        if sim_cfg is None or not hasattr(sim_cfg, "dt"):
            raise AttributeError(
                "env.cfg.sim.dt is not set. Please ensure your environment config "
                "defines `sim.dt` (the simulation timestep)."
            )
        if not hasattr(env.cfg, "decimation"):
            raise AttributeError(
                "env.cfg.decimation is not set. Please ensure your environment config "
                "defines `decimation` (the action repeat factor)."
            )
        simulation_dt = env.cfg.sim.dt * env.cfg.decimation

        num_amp_obs = AMP_PPO._flatten_amp_obs(obs["amp"]).shape[1]

        vel_repr_str = dataset_cfg.get("velocity_representation", "body_fixed")
        velocity_representation = VelocityRepresentation(vel_repr_str)

        symmetry_cfg = alg_cfg.get("symmetry_cfg")

        amp_data = AMPLoader(
            device=device,
            dataset_path_root=dataset_cfg["amp_data_path"],
            datasets=dataset_cfg["datasets"],
            simulation_dt=simulation_dt,
            slow_down_factor=dataset_cfg["slow_down_factor"],
            expected_joint_names=amp_joint_names,
            velocity_representation=velocity_representation,
            symmetry_cfg=symmetry_cfg,
        )

        discriminator = disc_class(
            input_dim=num_amp_obs * 2,
            hidden_layer_sizes=disc_cfg["hidden_dims"],
            reward_scale=disc_cfg["reward_scale"],
            device=device,
            loss_type=disc_cfg["loss_type"],
            use_minibatch_std=disc_cfg.get("use_minibatch_std", True),
            empirical_normalization=disc_cfg["empirical_normalization"],
            symmetry_cfg=symmetry_cfg,
        ).to(device)

        return discriminator, amp_data
