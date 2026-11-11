# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
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
    Mixture-of-Experts network:  ⎡expert_1(x) … expert_K(x)⎤·softmax(gate(x))
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
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        act = resolve_nn_activation(activation)

        # experts
        self.experts = nn.ModuleList(
            [MLP_net(obs_dim, hidden_dims, act_dim, act) for _ in range(num_experts)]
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, obs_dim]
        Returns:
            mean action: [batch, act_dim]
        """
        expert_out = torch.stack([e(x) for e in self.experts], dim=-1)
        gate_logits = self.gate(x)  # [batch, K]
        weights = self.softmax(gate_logits).unsqueeze(1)  # [batch, 1, K]
        return (expert_out * weights).sum(-1)  # weighted sum -> [batch, A]


class MoEModel(MLPModel):
    """:class:`rsl_rl.models.MLPModel` whose MLP is replaced by a Mixture-of-Experts network.

    It plugs into any rsl-rl actor/critic slot (``class_name: "MoEModel"``) and keeps the
    MLPModel interface (normalization, distribution, JIT/ONNX export).
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        gate_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = ActorMoE(
            obs_dim=self._get_latent_dim(),
            act_dim=mlp_output_dim,
            hidden_dims=list(hidden_dims),
            num_experts=num_experts,
            gate_hidden_dims=list(hidden_dims[:-1]) if gate_hidden_dims is None else gate_hidden_dims,
            activation=activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)
