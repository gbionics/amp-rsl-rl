# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from amp_rsl_rl.utils._compat import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class MLP_net(nn.Sequential):
    def __init__(self, in_dim, hidden_dims, out_dim, act):
        layers = [nn.Linear(in_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act])
        super().__init__(*layers)


class ActorMoE(nn.Module):
    """
    Mixture-of-Experts actor:  ⎡expert_1(x) … expert_K(x)⎤·softmax(gate(x))
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        num_experts: int = 4,
        gate_hidden_dims: list[int] | None = None,
        activation="elu",
    ):
        """Build an MoE actor with a batched expert path for K>1.

        The previous implementation evaluated each expert sequentially in Python.
        This implementation stacks expert parameters and computes all experts in
        parallel using batched tensor math, while keeping a single-expert fallback
        path for compatibility.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        act = resolve_nn_activation(activation)
        self.expert_activation = act
        self.hidden_dims = list(hidden_dims)

        # experts
        self.experts: nn.ModuleList | None = None
        self.expert_weights: nn.ParameterList | None = None
        self.expert_biases: nn.ParameterList | None = None
        if num_experts == 1:
            self.experts = nn.ModuleList([MLP_net(obs_dim, hidden_dims, act_dim, act)])
        else:
            dims = [obs_dim, *self.hidden_dims, act_dim]
            self.expert_weights = nn.ParameterList()
            self.expert_biases = nn.ParameterList()
            for in_features, out_features in zip(dims[:-1], dims[1:]):
                weight = nn.Parameter(
                    torch.empty(num_experts, out_features, in_features)
                )
                bias = nn.Parameter(torch.empty(num_experts, out_features))
                self._reset_batched_linear(weight, bias)
                self.expert_weights.append(weight)
                self.expert_biases.append(bias)

        # gating network
        gate_layers = []
        last_dim = obs_dim
        gate_hidden_dims = gate_hidden_dims or []
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), act]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        self.softmax = nn.Softmax(dim=-1)  # kept separate for ONNX clarity

    @staticmethod
    def _reset_batched_linear(weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Initialize stacked expert linear layers like ``nn.Linear`` defaults."""
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in = weight.shape[-1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the MoE action mean while preserving previous outputs.

        Args:
            x: [batch, obs_dim]
        Returns:
            mean action: [batch, act_dim]
        """
        if self.num_experts == 1 and self.experts is not None:
            expert_out = self.experts[0](x).unsqueeze(-1)
        else:
            assert self.expert_weights is not None
            assert self.expert_biases is not None
            x_exp = x.unsqueeze(1).expand(-1, self.num_experts, -1)
            h = x_exp
            for layer_idx, (weight, bias) in enumerate(
                zip(self.expert_weights, self.expert_biases)
            ):
                h = torch.einsum("bki,koi->bko", h, weight) + bias.unsqueeze(0)
                if layer_idx < len(self.expert_weights) - 1:
                    h = self.expert_activation(h)
            expert_out = h.transpose(1, 2)

        gate_logits = self.gate(x)  # [batch, K]
        weights = self.softmax(gate_logits).unsqueeze(1)  # [batch, 1, K]
        return (expert_out * weights).sum(-1)  # weighted sum -> [batch, A]


class ActorCriticMoE(nn.Module):
    """Actor-critic module powered by a Mixture-of-Experts policy network.

    The API mirrors :class:`rsl_rl.modules.ActorCritic` so the class can be
    referenced via the standard policy ``class_name`` in configuration files.
    Observations are provided as TensorDict (or dict-like) containers and are
    grouped via ``obs_groups`` exactly like the upstream implementation.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        num_experts: int = 4,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                (
                    "ActorCriticMoE.__init__ ignored unexpected arguments: "
                    + str(list(kwargs.keys()))
                )
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "ActorCriticMoE only supports 1D flattened observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "ActorCriticMoE only supports 1D flattened observations."
            num_critic_obs += obs[obs_group].shape[-1]

        act = resolve_nn_activation(activation)

        self.actor = ActorMoE(
            obs_dim=num_actor_obs,
            act_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=actor_hidden_dims[:-1],
            activation=activation,
        )
        self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor (MoE) structure:\n{self.actor}")
        print(f"Critic MLP structure:\n{self.critic}")

    def reset(self, dones=None):  # noqa: D401
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:  # "log"
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs):
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # unchanged load_state_dict so checkpoints from the old class still load
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
