# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export rsl-rl models to ONNX."""

from __future__ import annotations

import os

import torch


def export_policy_as_onnx(
    model,
    path: str,
    normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export an rsl-rl model (``MLPModel`` or a subclass) into an ONNX file.

    The model's own ``as_onnx`` export wrapper is used, so its observation normalizer and
    deterministic distribution head are included.

    Args:
        model: The actor model to export.
        path: The path to the saving directory.
        normalizer: Unused, the model carries its own normalizer. Kept for signature
            compatibility with Isaac Lab's exporter.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    onnx_model = model.as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamo=False,
    )
