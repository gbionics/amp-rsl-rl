# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Code taken from https://github.com/isaac-sim/IsaacLab/blob/5716d5600a1a0e45345bc01342a70bd81fac7889/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

import os
import torch


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


def _clone_module(module):
    """Clone a module by pickling via torch.save/torch.load to avoid deepcopy issues with non-leaf tensors."""
    import io
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location='cpu', weights_only=False)


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self._is_mlp_model = False
        # Handle both ActorCritic (v3) and direct actor module (v4+)
        if hasattr(actor_critic, 'actor'):
            actor = actor_critic.actor
        else:
            actor = actor_critic
        
        # Check if this is an MLPModel (v4+ style with mlp attribute)
        if hasattr(actor, 'mlp'):
            self._is_mlp_model = True
            # Clone using torch.save/load to avoid deepcopy issues with non-leaf tensors
            actor_clone = _clone_module(actor)
            self.mlp = actor_clone.mlp
            self.distribution = actor_clone.distribution
            self.actor = None  # Not used for MLPModel
        else:
            self.actor = _clone_module(actor)
            self.mlp = None
            self.distribution = None
            
        self.is_recurrent = getattr(actor_critic, 'is_recurrent', False)
        if self.is_recurrent:
            self.rnn = _clone_module(actor_critic.memory_a.rnn)
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = _clone_module(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        x = self.normalizer(x)
        if self._is_mlp_model:
            mlp_output = self.mlp(x)
            if self.distribution is not None:
                return self.distribution.deterministic_output(mlp_output)
            return mlp_output
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self._is_mlp_model = False
        # Handle both ActorCritic (v3) and direct actor module (v4+)
        if hasattr(actor_critic, 'actor'):
            actor = actor_critic.actor
        else:
            actor = actor_critic
        
        # Check if this is an MLPModel (v4+ style with mlp attribute)
        if hasattr(actor, 'mlp'):
            self._is_mlp_model = True
            # Clone using torch.save/load to avoid deepcopy issues with non-leaf tensors
            actor_clone = _clone_module(actor)
            self.mlp = actor_clone.mlp
            self.distribution = actor_clone.distribution
            self.actor = None  # Not used for MLPModel
        else:
            self.actor = _clone_module(actor)
            self.mlp = None
            self.distribution = None
            
        self.is_recurrent = getattr(actor_critic, 'is_recurrent', False)
        if self.is_recurrent:
            self.rnn = _clone_module(actor_critic.memory_a.rnn)
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = _clone_module(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        x = self.normalizer(x)
        if self._is_mlp_model:
            mlp_output = self.mlp(x)
            if self.distribution is not None:
                return self.distribution.deterministic_output(mlp_output)
            return mlp_output
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=18,  # see  https://github.com/isaac-sim/IsaacLab/blob/ddb044eb5b2300792de41e82d53b032f3632b489/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py#L172.
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
                dynamo=False,
            )
        else:
            # Handle ActorMoE, MLPModel (v4+), and nn.Sequential (v3)
            if self._is_mlp_model:
                # For MLPModel, get input dimension from the MLP's first layer
                obs_dim = self.mlp[0].in_features
            elif hasattr(self.actor, 'obs_dim'):
                obs_dim = self.actor.obs_dim
            else:
                obs_dim = self.actor[0].in_features
            obs = torch.zeros(1, obs_dim)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=18,  # see  https://github.com/isaac-sim/IsaacLab/blob/ddb044eb5b2300792de41e82d53b032f3632b489/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py#L172.
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
                dynamo=False,
            )
