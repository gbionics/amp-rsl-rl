# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage

from amp_rsl_rl.storage import ReplayBuffer
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import AMPLoader, _call_augmentation_func
from amp_rsl_rl.utils._compat import RSL_RL_V3_3_PLUS, RSL_RL_V4_PLUS, RSL_RL_V5_PLUS

if RSL_RL_V3_3_PLUS:
    from rsl_rl.utils import resolve_callable
else:
    from rsl_rl.utils import string_to_callable as resolve_callable


class AMP_PPO:
    """
    AMP_PPO implements Adversarial Motion Priors (AMP) combined with Proximal Policy Optimization (PPO).

    The algorithm mirrors the structure of upstream ``PPO`` from ``rsl_rl`` but augments each update
    with a discriminator trained on expert trajectories. Observations feed into the policy as
    TensorDicts, allowing the actor and critic to consume different observation groups.

    Parameters
    ----------
    discriminator : Discriminator
        AMP discriminator distinguishing expert vs policy motion pairs.
    amp_data : AMPLoader
        Data loader that provides batches of expert motion data.
    num_learning_epochs : int, default=1
        Number of passes over the rollout buffer per update.
    num_mini_batches : int, default=1
        Number of mini-batches to divide each epoch's data into.
    clip_param : float, default=0.2
        PPO clipping parameter that bounds the policy update step.
    gamma : float, default=0.998
        Discount factor.
    lam : float, default=0.95
        Lambda parameter for Generalized Advantage Estimation (GAE).
    value_loss_coef : float, default=1.0
        Coefficient for the value function loss term in the PPO loss.
    entropy_coef : float, default=0.0
        Coefficient for the entropy regularization term (encouraging exploration).
    learning_rate : float, default=1e-3
        Initial learning rate.
    max_grad_norm : float, default=1.0
        Maximum gradient norm for clipping gradients during backpropagation.
    use_clipped_value_loss : bool, default=True
        Enables the clipped value loss variant of PPO.
    schedule : str, default="fixed"
        Either ``"fixed"`` or ``"adaptive"`` (based on KL).
    desired_kl : float, default=0.01
        Target KL divergence when using the adaptive schedule.
    amp_replay_buffer_size : int, default=100_000
        Size of the replay buffer storing policy-generated AMP samples.
    use_smooth_ratio_clipping : bool, default=False
        Enables smooth ratio clipping instead of hard clamping.
    normalize_advantage_per_mini_batch : bool, default=False
        Whether to normalize advantages within each mini-batch (instead of the entire rollout).
    symmetry_cfg : dict | None, default=None
        Configuration dict enabling symmetry-based data augmentation and mirror loss.
    device : str, default="cpu"
        Torch device used by the module.
    """

    def __init__(
        self,
        discriminator: Discriminator = None,
        amp_data: AMPLoader = None,
        actor_critic=None,
        actor=None,
        critic=None,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        amp_replay_buffer_size: int = 100000,
        use_smooth_ratio_clipping: bool = False,
        normalize_advantage_per_mini_batch: bool = False,
        symmetry_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        discriminators: Optional[Sequence[Discriminator]] = None,
        gate_loss_coef: float = 0.0,
        discriminator_learning_rate: Optional[float] = None,
        grad_pen_scale: float = 10.0,
    ) -> None:
        # Set device and learning hyperparameters
        self.device: str = device
        self.desired_kl: float = desired_kl
        self.schedule: str = schedule
        self.learning_rate: float = learning_rate
        self.normalize_advantage_per_mini_batch: bool = (
            normalize_advantage_per_mini_batch
        )
        self.gate_loss_coef: float = gate_loss_coef
        # Optional dedicated (fixed) learning rate for the discriminator(s). When
        # set, the discriminator parameter groups are trained at this rate and are
        # NOT touched by the adaptive-KL learning-rate schedule that governs the
        # actor/critic. This prevents the discriminator from overpowering the
        # policy (a common cause of discriminator saturation / vanishing AMP
        # reward). When ``None`` the legacy behaviour is kept (the discriminator
        # shares the policy learning rate, including adaptive updates).
        self.discriminator_learning_rate: Optional[float] = discriminator_learning_rate
        # Coefficient of the discriminator gradient penalty (R1 / WGAN-GP).
        self.grad_pen_scale: float = grad_pen_scale

        if RSL_RL_V4_PLUS:
            if actor is None or critic is None:
                raise ValueError(
                    "rsl-rl-lib >= 4 detected: 'actor' and 'critic' must both be provided to AMP_PPO."
                )
            self.actor = actor.to(self.device)
            self.critic = critic.to(self.device)
            self.actor_critic = None
        else:
            if actor_critic is None:
                raise ValueError(
                    "rsl-rl-lib < 4 detected: 'actor_critic' must be provided to AMP_PPO."
                )
            # Set up the actor-critic (policy) and move it to the device.
            self.actor_critic = actor_critic.to(self.device)
            self.actor = self.actor_critic
            self.critic = None

        # Set up the discriminator(s) and move them to the appropriate device.
        # ``discriminators`` (a per-skill list) takes precedence over the legacy
        # single ``discriminator`` argument. Internally everything is stored as a
        # list so that the single-skill case is just ``num_skills == 1``.
        if discriminators is not None:
            disc_list = list(discriminators)
        elif discriminator is not None:
            disc_list = [discriminator]
        else:
            raise ValueError(
                "AMP_PPO requires either 'discriminator' or 'discriminators'."
            )
        self.discriminators: nn.ModuleList = nn.ModuleList(
            [d.to(self.device) for d in disc_list]
        )
        self.num_skills: int = len(self.discriminators)
        # Backwards-compatible alias for code/checkpoints that expect a single one.
        self.discriminator: Discriminator = self.discriminators[0]
        # Per-env skill labels for the most recent AMP transition (set by act_amp).
        self._amp_skill_ids: Optional[torch.Tensor] = None
        # Populated at the end of each update() for logging.
        self.mean_gate_loss: float = 0.0

        self.amp_transition: RolloutStorage.Transition = RolloutStorage.Transition()
        # Determine observation dimension used in the replay buffer.
        # The discriminator expects concatenated observations, so the replay buffer uses half the dimension.
        obs_dim: int = self.discriminators[0].input_dim // 2
        # One replay buffer per skill so each discriminator is trained only on the
        # policy transitions generated while the corresponding skill was active.
        self.amp_storage: list[ReplayBuffer] = [
            ReplayBuffer(
                obs_dim=obs_dim, buffer_size=amp_replay_buffer_size, device=device
            )
            for _ in range(self.num_skills)
        ]
        self.amp_data: AMPLoader = amp_data
        self.storage: Optional[RolloutStorage] = (
            None  # Will be initialized later once environment parameters are known
        )

        # Create optimizer for both the actor-critic and the discriminator.
        # Note: Weight decay is set differently for discriminator trunk and head.
        # When a dedicated discriminator learning rate is configured, it is set
        # directly on the discriminator param groups so it stays fixed (the
        # adaptive schedule below only touches the actor/critic groups).
        disc_lr = (
            self.discriminator_learning_rate
            if self.discriminator_learning_rate is not None
            else learning_rate
        )
        disc_params = []
        for skill_id, disc in enumerate(self.discriminators):
            disc_params.append(
                {
                    "params": disc.trunk.parameters(),
                    "weight_decay": 10e-4,
                    "lr": disc_lr,
                    "name": f"amp_trunk_{skill_id}",
                }
            )
            disc_params.append(
                {
                    "params": disc.linear.parameters(),
                    "weight_decay": 10e-2,
                    "lr": disc_lr,
                    "name": f"amp_head_{skill_id}",
                }
            )
        if RSL_RL_V4_PLUS:
            params = [
                {"params": self.actor.parameters(), "name": "actor"},
                {"params": self.critic.parameters(), "name": "critic"},
                *disc_params,
            ]
        else:
            params = [
                {"params": self.actor_critic.parameters(), "name": "actor_critic"},
                *disc_params,
            ]
        self.optimizer: optim.Adam = optim.Adam(params, lr=learning_rate)
        self.transition: RolloutStorage.Transition = RolloutStorage.Transition()

        # PPO-specific parameters
        self.clip_param: float = clip_param
        self.num_learning_epochs: int = num_learning_epochs
        self.num_mini_batches: int = num_mini_batches
        self.value_loss_coef: float = value_loss_coef
        self.entropy_coef: float = entropy_coef
        self.gamma: float = gamma
        self.lam: float = lam
        self.max_grad_norm: float = max_grad_norm
        self.use_clipped_value_loss: bool = use_clipped_value_loss
        self.use_smooth_ratio_clipping: bool = use_smooth_ratio_clipping

        # Symmetry configuration for PPO and AMP augmentation
        self.symmetry_cfg: Optional[Dict[str, Any]] = None
        if symmetry_cfg is not None:
            use_data_augmentation = symmetry_cfg.get("use_data_augmentation", False)
            use_mirror_loss = symmetry_cfg.get("use_mirror_loss", False)
            if not use_data_augmentation:
                print(
                    "Data augmentation is not enabled. Symmetry utilities will only be available for logging."
                )
            if not use_mirror_loss:
                print(
                    "Mirror loss is not enabled. Symmetry utilities will only be available for logging."
                )
            aug_fn = symmetry_cfg.get("data_augmentation_func", None)
            if isinstance(aug_fn, str):
                symmetry_cfg["data_augmentation_func"] = resolve_callable(aug_fn)
            aug_fn = symmetry_cfg.get("data_augmentation_func", None)
            if aug_fn is not None and not callable(aug_fn):
                raise ValueError(
                    "Symmetry configuration exists but the function is not callable: "
                    f"{aug_fn}"
                )
            if getattr(actor_critic, "is_recurrent", False):
                raise ValueError(
                    "Symmetry augmentation is not supported for recurrent policies in AMP_PPO."
                )
            self.symmetry_cfg = symmetry_cfg

        # Reference to the policy module that owns the MoE gate (if any). Gate
        # supervision (binding expert i -> skill i) is only possible when the
        # actor exposes ``get_gate_logits`` and there is more than one skill.
        self._policy_module = self.actor if RSL_RL_V4_PLUS else self.actor_critic
        self._gate_supervision_available = (
            self.num_skills > 1
            and self.gate_loss_coef > 0.0
            and hasattr(self._policy_module, "get_gate_logits")
        )
        if self.gate_loss_coef > 0.0 and not hasattr(
            self._policy_module, "get_gate_logits"
        ):
            print(
                "AMP_PPO: gate_loss_coef > 0 but the policy does not expose "
                "'get_gate_logits' (not a Mixture-of-Experts actor). "
                "Gate supervision will be disabled."
            )

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        observations: TensorDict,
        action_shape: Tuple[int, ...],
    ) -> None:
        """Initialize rollout storage for TensorDict observations.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments.
        num_transitions_per_env : int
            Horizon (per environment) stored inside the rollout buffer.
        observations : TensorDict
            Prototype observation structure used to determine buffer shapes.
        action_shape : Tuple[int, ...]
            Shape of the action vector output by the policy.
        """
        self.storage = RolloutStorage(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=observations,
            actions_shape=action_shape,
            device=self.device,
        )

    def _augment_batch_size(
        self, original_size: int, augmented: Optional[torch.Tensor]
    ) -> int:
        """Compute augmentation factor given the original and augmented batch sizes."""

        if augmented is None or original_size == 0:
            return 1
        if augmented.shape[0] % original_size != 0:
            raise ValueError(
                "Symmetry augmentation function returned a batch size incompatible with the original size."
                f" Original={original_size}, augmented={augmented.shape[0]}"
            )
        return augmented.shape[0] // original_size

    def _repeat_along_batch(
        self, tensor: Optional[torch.Tensor], num_aug: int
    ) -> Optional[torch.Tensor]:
        """Repeat a tensor along the first dimension to match augmentation factor."""

        if tensor is None or num_aug == 1:
            return tensor
        repeat_dims = [num_aug] + [1] * (tensor.dim() - 1)
        return tensor.repeat(*repeat_dims)

    def _apply_symmetry(
        self,
        *,
        obs: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        obs_type: Union[str, Sequence[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply configured symmetry augmentation to observations/actions."""

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

    def test_mode(self) -> None:
        """
        Sets the actor-critic model to evaluation mode.
        """
        if RSL_RL_V4_PLUS:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor_critic.eval()

    def train_mode(self) -> None:
        """
        Sets the actor-critic model to training mode.
        """
        if RSL_RL_V4_PLUS:
            self.actor.train()
            self.critic.train()
        else:
            self.actor_critic.train()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Select an action and value estimate for the current observation.

        Parameters
        ----------
        obs : TensorDict
            Batched observation TensorDict provided by the environment.

        Returns
        -------
        torch.Tensor
            Detached action tensor sampled from the actor-critic policy.
        """
        if RSL_RL_V4_PLUS:
            if self.actor.is_recurrent:
                self.transition.hidden_states = self.actor.get_hidden_state()
            self.transition.actions = self.actor(obs, stochastic_output=True).detach()
            self.transition.values = self.critic(obs).detach()
            self.transition.actions_log_prob = self.actor.get_output_log_prob(
                self.transition.actions
            ).detach()
            if RSL_RL_V5_PLUS:
                self.transition.distribution_params = [
                    self.actor.output_mean.detach(),
                    self.actor.output_std.detach(),
                ]
            else:
                self.transition.action_mean = self.actor.output_mean.detach()
                self.transition.action_sigma = self.actor.output_std.detach()
        else:
            if self.actor_critic.is_recurrent:
                self.transition.hidden_states = self.actor_critic.get_hidden_states()
            self.transition.actions = self.actor_critic.act(obs).detach()
            self.transition.values = self.actor_critic.evaluate(obs).detach()
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
                self.transition.actions
            ).detach()
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()

        self.transition.observations = obs
        return self.transition.actions

    def act_amp(
        self, amp_obs: torch.Tensor, skill_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Store the latest AMP policy observation for later replay insertion.

        Parameters
        ----------
        amp_obs : torch.Tensor
            Concatenated AMP observation representing the current policy state.
        skill_ids : torch.Tensor | None
            Per-environment integer skill labels (shape ``[num_envs]``) used to
            route each transition to the replay buffer of the corresponding
            skill. When ``None`` all transitions go to the single (skill-0)
            buffer, preserving the original single-skill behaviour.
        """
        self.amp_transition.observations = amp_obs
        self._amp_skill_ids = skill_ids

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Dict[str, Any],
    ) -> None:
        """Record the outcome of an environment step and update normalizers.

        Parameters
        ----------
        obs : TensorDict
            Observation returned by the environment after stepping.
        rewards : torch.Tensor
            Reward tensor (batch x 1) after mixing task/style components.
        dones : torch.Tensor
            Episode termination flags.
        extras : dict[str, Any]
            Additional metadata from the environment (e.g. ``time_outs``).
        """
        if RSL_RL_V4_PLUS:
            self.actor.update_normalization(obs)
            self.critic.update_normalization(obs)
        else:
            self.actor_critic.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        if RSL_RL_V4_PLUS:
            self.storage.add_transition(self.transition)
            self.transition.clear()
            self.actor.reset(dones)
            self.critic.reset(dones)
        else:
            self.storage.add_transitions(self.transition)
            self.transition.clear()
            self.actor_critic.reset(dones)

    def process_amp_step(self, amp_obs: torch.Tensor) -> None:
        """Insert a policy-generated AMP transition into the replay buffer.

        When multiple skills are used, the transition of each environment is
        routed to the replay buffer of the skill that was active for that
        environment (as recorded by :meth:`act_amp`).

        Parameters
        ----------
        amp_obs : torch.Tensor
            Next AMP observation paired with the previously stored policy state.
        """
        state = self.amp_transition.observations
        next_state = amp_obs
        skill_ids = self._amp_skill_ids

        if self.num_skills == 1 or skill_ids is None:
            self.amp_storage[0].insert(state, next_state)
        else:
            skill_ids = skill_ids.to(state.device).view(-1)
            for skill_id in range(self.num_skills):
                mask = skill_ids == skill_id
                if bool(mask.any()):
                    self.amp_storage[skill_id].insert(state[mask], next_state[mask])

        self.amp_transition.clear()
        self._amp_skill_ids = None

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute and store GAE-lambda returns from the final observation.

        Parameters
        ----------
        obs : TensorDict
            Last observation gathered after rollout completion.
        """
        if RSL_RL_V4_PLUS:
            # v4+ removed compute_returns from RolloutStorage
            # (see https://github.com/leggedrobotics/rsl_rl/pull/137)
            last_values = self.critic(obs).detach()
            st = self.storage
            advantage = 0
            for step in reversed(range(st.num_transitions_per_env)):
                next_values = (
                    last_values
                    if step == st.num_transitions_per_env - 1
                    else st.values[step + 1]
                )
                next_is_not_terminal = 1.0 - st.dones[step].float()
                delta = (
                    st.rewards[step]
                    + next_is_not_terminal * self.gamma * next_values
                    - st.values[step]
                )
                advantage = (
                    delta + next_is_not_terminal * self.gamma * self.lam * advantage
                )
                st.returns[step] = advantage + st.values[step]
            st.advantages = st.returns - st.values
            st.advantages = (st.advantages - st.advantages.mean()) / (
                st.advantages.std() + 1e-8
            )
        else:
            last_values = self.actor_critic.evaluate(obs).detach()
            self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(
        self,
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Performs a single update step for both the actor-critic (PPO) and the AMP discriminator.
        It iterates over mini-batches of data, computes surrogate, value, AMP and gradient penalty losses,
        performs adaptive learning rate scheduling (if enabled), and updates model parameters.

        Returns
        -------
        tuple
            A tuple containing mean losses and statistics:
            (mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss,
             mean_policy_pred, mean_expert_pred, mean_accuracy_policy, mean_accuracy_expert,
             mean_kl_divergence, mean_symmetry_loss)
        """
        # Initialize mean loss and accuracy statistics.
        mean_value_loss: float = 0.0
        mean_surrogate_loss: float = 0.0
        mean_amp_loss: float = 0.0
        mean_grad_pen_loss: float = 0.0
        mean_policy_pred: float = 0.0
        mean_expert_pred: float = 0.0
        mean_accuracy_policy: float = 0.0
        mean_accuracy_expert: float = 0.0
        mean_accuracy_policy_elem: float = 0.0
        mean_accuracy_expert_elem: float = 0.0
        mean_kl_divergence: float = 0.0
        mean_symmetry_loss: float = 0.0
        mean_gate_loss: float = 0.0
        # Number of per-skill discriminator mini-batches actually processed
        # (skills whose replay buffer is empty are skipped), used to average the
        # discriminator statistics correctly.
        num_amp_batches: int = 0

        # Create data generators for mini-batch sampling.
        _is_recurrent = (
            self.actor.is_recurrent
            if RSL_RL_V4_PLUS
            else self.actor_critic.is_recurrent
        )

        if _is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # Generator for policy-generated AMP transitions.
        num_updates_total = self.num_learning_epochs * self.num_mini_batches
        amp_mini_batch_size = (
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches
        )
        # Only skills that have collected policy transitions this rollout are
        # trained (an empty replay buffer would make sampling ill-defined). Their
        # discriminator is simply left unchanged for this update.
        active_skills = [
            skill_id
            for skill_id in range(self.num_skills)
            if len(self.amp_storage[skill_id]) > 0
        ]
        amp_policy_generators = {
            skill_id: self.amp_storage[skill_id].feed_forward_generator(
                num_mini_batch=num_updates_total,
                mini_batch_size=amp_mini_batch_size,
                allow_replacement=True,
            )
            for skill_id in active_skills
        }
        # Expert generators. In the single-skill case sample from the full dataset
        # (skill_id=None) to exactly reproduce the original behaviour.
        if self.num_skills == 1:
            amp_expert_generators = {
                0: self.amp_data.feed_forward_generator(
                    num_updates_total, amp_mini_batch_size, skill_id=None
                )
            }
        else:
            amp_expert_generators = {
                skill_id: self.amp_data.feed_forward_generator(
                    num_updates_total, amp_mini_batch_size, skill_id=skill_id
                )
                for skill_id in active_skills
            }

        # Loop over mini-batches from the environment transitions.
        for sample in generator:
            # Unpack the mini-batch sample from the environment.
            if hasattr(sample, "observations"):
                obs_batch = sample.observations
                actions_batch = sample.actions
                target_values_batch = sample.values
                advantages_batch = sample.advantages
                returns_batch = sample.returns
                old_actions_log_prob_batch = sample.old_actions_log_prob
                old_mu_batch = sample.old_distribution_params[0]
                old_sigma_batch = sample.old_distribution_params[1]
                hidden_states_batch = sample.hidden_states
                masks_batch = sample.masks
            elif isinstance(sample, tuple) and len(sample) == 9:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_distribution_params_batch,
                    hidden_states_batch,
                    masks_batch,
                ) = sample
                old_mu_batch = old_distribution_params_batch[0]
                old_sigma_batch = old_distribution_params_batch[1]
            else:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hidden_states_batch,
                    masks_batch,
                ) = sample

            hidden_state_actor, hidden_state_critic = (None, None)
            if hidden_states_batch is not None:
                hidden_state_actor, hidden_state_critic = hidden_states_batch

            original_batch_size = obs_batch.shape[0]

            # Capture the per-sample skill labels BEFORE any symmetry augmentation:
            # the augmentation helper rebuilds ``obs_batch`` with only the policy
            # and critic groups, dropping the "skill" group used for gate
            # supervision. Mirroring left/right does not change the skill.
            skill_labels_batch = (
                obs_batch["skill"] if "skill" in obs_batch.keys() else None
            )

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Symmetry data augmentation for PPO inputs
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

            # Forward pass through the actor to get current policy outputs.
            if RSL_RL_V4_PLUS:
                _ = self.actor(
                    obs_batch,
                    masks=masks_batch,
                    hidden_state=hidden_state_actor,
                    stochastic_output=True,
                )
                actions_log_prob_batch = self.actor.get_output_log_prob(actions_batch)
                value_batch = self.critic(
                    obs_batch, masks=masks_batch, hidden_state=hidden_state_critic
                )
                if hasattr(self.actor, "get_distribution_params"):
                    dist_params = self.actor.get_distribution_params()
                    mu_batch = dist_params[0][:original_batch_size]
                    sigma_batch = dist_params[1][:original_batch_size]
                    entropy_batch = self.actor.get_entropy()[:original_batch_size]
                else:
                    mu_batch = self.actor.output_mean[:original_batch_size]
                    sigma_batch = self.actor.output_std[:original_batch_size]
                    entropy_batch = self.actor.output_entropy[:original_batch_size]
            else:
                # v3
                self.actor_critic.act(
                    obs_batch, masks=masks_batch, hidden_states=hidden_state_actor
                )
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                    actions_batch
                )
                value_batch = self.actor_critic.evaluate(
                    obs_batch, masks=masks_batch, hidden_states=hidden_state_critic
                )
                mu_batch = self.actor_critic.action_mean[:original_batch_size]
                sigma_batch = self.actor_critic.action_std[:original_batch_size]
                entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # Adaptive learning rate adjustment based on KL divergence if schedule is "adaptive".
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    mean_kl_divergence += kl_mean.item()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        # When the discriminator has its own dedicated learning
                        # rate, keep it fixed: the adaptive KL schedule only
                        # governs the actor/critic param groups.
                        if (
                            self.discriminator_learning_rate is not None
                            and str(param_group.get("name", "")).startswith("amp_")
                        ):
                            continue
                        param_group["lr"] = self.learning_rate

            # Compute the PPO surrogate loss.
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )

            min_ = 1.0 - self.clip_param
            max_ = 1.0 + self.clip_param
            # Smooth clipping for the ratio if enabled.
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

            # Compute the value function loss.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Combine surrogate loss, value loss and entropy regularization to form PPO loss.
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Mirror loss (only when it actually feeds the gradient; computing it
            # for logging only would add a full extra actor forward pass per
            # mini-batch, inflating Perf/learning_time even when symmetry is off).
            symmetry_loss_value = torch.zeros(1, device=self.device)
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
                    with torch.no_grad():
                        sym_obs_detached = sym_obs_batch.detach().clone()
                    if RSL_RL_V4_PLUS:
                        mean_actions_batch = self.actor(
                            sym_obs_detached, stochastic_output=False
                        )
                    else:
                        mean_actions_batch = self.actor_critic.act_inference(
                            sym_obs_detached
                        )
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, sym_actions = self._apply_symmetry(
                        obs=None,
                        actions=action_mean_orig,
                        obs_type="policy",
                    )
                    if sym_actions is None:
                        sym_actions = mean_actions_batch
                    symmetry_loss_value = torch.nn.functional.mse_loss(
                        mean_actions_batch[original_batch_size:],
                        sym_actions.detach()[original_batch_size:],
                    )
                    coeff = self.symmetry_cfg.get("mirror_loss_coeff", 0.0)
                    ppo_loss = ppo_loss + coeff * symmetry_loss_value

            # ─── MoE gate-supervision loss ───
            # Push the gating distribution towards the (environment-provided)
            # skill label of each sample so that expert i specialises in skill i.
            gate_loss = torch.zeros((), device=self.device)
            if self._gate_supervision_available and skill_labels_batch is not None:
                # get_gate_logits is computed over obs_batch (which may be the
                # augmented batch); only the original samples have matching labels.
                gate_logits = self._policy_module.get_gate_logits(obs_batch)[
                    :original_batch_size
                ]
                skill_targets = skill_labels_batch.reshape(-1).long()
                gate_loss = F.cross_entropy(gate_logits, skill_targets)
                mean_gate_loss += gate_loss.item()

            # ─── AMP discriminator loss (per skill) ───
            # Accumulate the discriminator loss across all active skills so that a
            # single backward pass jointly updates the shared PPO/actor params and
            # every skill discriminator.
            total_amp_loss = torch.zeros((), device=self.device)
            total_grad_pen_loss = torch.zeros((), device=self.device)
            # Raw observations for each skill are cached and used to update the
            # empirical normalizers *after* the optimizer step (under no_grad).
            normalizer_updates = []

            for skill_id in active_skills:
                policy_state, policy_next_state = next(amp_policy_generators[skill_id])
                expert_state, expert_next_state = next(amp_expert_generators[skill_id])
                disc = self.discriminators[skill_id]

                if self.symmetry_cfg and self.symmetry_cfg.get(
                    "use_data_augmentation", False
                ):
                    policy_state = disc.apply_symmetry(policy_state, obs_type="amp")
                    policy_next_state = disc.apply_symmetry(
                        policy_next_state, obs_type="amp"
                    )
                    expert_state = disc.apply_symmetry(expert_state, obs_type="amp")
                    expert_next_state = disc.apply_symmetry(
                        expert_next_state, obs_type="amp"
                    )

                # Ensure everything is on the right device (AMPLoader may yield CPU tensors)
                policy_state = policy_state.to(self.device)
                policy_next_state = policy_next_state.to(self.device)
                expert_state = expert_state.to(self.device)
                expert_next_state = expert_next_state.to(self.device)

                # Keep raw tensors for normalizer updates
                normalizer_updates.append(
                    (
                        skill_id,
                        expert_state.detach().clone(),
                        expert_next_state.detach().clone(),
                        policy_state.detach().clone(),
                        policy_next_state.detach().clone(),
                    )
                )

                # Concatenate policy and expert AMP observations for the discriminator input.
                B_pol = policy_state.size(0)
                discriminator_input = torch.cat(
                    (
                        torch.cat([policy_state, policy_next_state], dim=-1),
                        torch.cat([expert_state, expert_next_state], dim=-1),
                    ),
                    dim=0,
                )
                discriminator_output = disc(discriminator_input)
                policy_d, expert_d = (
                    discriminator_output[:B_pol],
                    discriminator_output[B_pol:],
                )

                # Compute discriminator losses for this skill
                amp_loss_k, grad_pen_loss_k = disc.compute_loss(
                    policy_d=policy_d,
                    expert_d=expert_d,
                    sample_amp_expert=(expert_state, expert_next_state),
                    sample_amp_policy=(policy_state, policy_next_state),
                    lambda_=self.grad_pen_scale,
                )
                total_amp_loss = total_amp_loss + amp_loss_k
                total_grad_pen_loss = total_grad_pen_loss + grad_pen_loss_k

                # Discriminator diagnostics (aggregated across skills).
                with torch.no_grad():
                    policy_d_prob = torch.sigmoid(policy_d)
                    expert_d_prob = torch.sigmoid(expert_d)
                mean_amp_loss += amp_loss_k.item()
                mean_grad_pen_loss += grad_pen_loss_k.item()
                mean_policy_pred += policy_d_prob.mean().item()
                mean_expert_pred += expert_d_prob.mean().item()
                mean_accuracy_policy += torch.sum(
                    torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
                ).item()
                mean_accuracy_expert += torch.sum(
                    torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
                ).item()
                mean_accuracy_expert_elem += expert_d_prob.numel()
                mean_accuracy_policy_elem += policy_d_prob.numel()
                num_amp_batches += 1

            # The final loss combines the PPO loss, gate-supervision loss and AMP losses.
            loss = ppo_loss + self.gate_loss_coef * gate_loss + (
                total_amp_loss + total_grad_pen_loss
            )

            # Backpropagation and optimizer step.
            self.optimizer.zero_grad()
            loss.backward()
            if RSL_RL_V4_PLUS:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            # Update the normalizers with RAW (unnormalized) observations under no_grad
            for (
                skill_id,
                expert_state_raw,
                expert_next_state_raw,
                policy_state_raw,
                policy_next_state_raw,
            ) in normalizer_updates:
                self.discriminators[skill_id].update_normalization(
                    expert_state_raw,
                    expert_next_state_raw,
                    policy_state_raw,
                    policy_next_state_raw,
                )

            # Update running PPO statistics.
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_symmetry_loss += symmetry_loss_value.item()

        # Average the statistics over all mini-batches.
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # Discriminator statistics are averaged over the number of per-skill
        # discriminator batches that were actually processed.
        _amp_denom = max(1, num_amp_batches)
        mean_amp_loss /= _amp_denom
        mean_grad_pen_loss /= _amp_denom
        mean_policy_pred /= _amp_denom
        mean_expert_pred /= _amp_denom
        mean_accuracy_policy /= max(1, mean_accuracy_policy_elem)
        mean_accuracy_expert /= max(1, mean_accuracy_expert_elem)
        mean_kl_divergence /= num_updates
        mean_symmetry_loss /= num_updates
        mean_gate_loss /= num_updates
        self.mean_gate_loss = mean_gate_loss

        # Clear the storage for the next update cycle.
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
            mean_kl_divergence,
            mean_symmetry_loss,
        )
