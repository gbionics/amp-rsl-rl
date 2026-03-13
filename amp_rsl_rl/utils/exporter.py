# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Code taken from https://github.com/isaac-sim/IsaacLab/blob/5716d5600a1a0e45345bc01342a70bd81fac7889/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

import os
import torch


def export_policy_as_onnx(
    policy_model: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export plain policy model into an ONNX file.

    Args:
        policy_model: The policy torch module exposing ``as_onnx()``.
        normalizer: Unused (kept for backward compatibility).
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if not hasattr(policy_model, "as_onnx"):
        raise TypeError(
            "export_policy_as_onnx only supports plain policy models exposing as_onnx()."
        )

    onnx_model = policy_model.as_onnx(verbose=verbose).cpu()
    dummy_inputs = (
        onnx_model.get_dummy_inputs()
        if hasattr(onnx_model, "get_dummy_inputs")
        else (torch.zeros(1, onnx_model.input_size),)
    )
    if not isinstance(dummy_inputs, tuple):
        dummy_inputs = (dummy_inputs,)

    input_names = getattr(onnx_model, "input_names", ["obs"])
    output_names = getattr(onnx_model, "output_names", ["actions"])
    torch.onnx.export(
        onnx_model,
        dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0],
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
    )
