# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class resolution that understands AMP's own short names.

rsl-rl's :func:`resolve_callable` resolves bare class names only within the
``rsl_rl`` namespace. AMP configuration files reference classes such as
``"AMP_PPO"`` or ``"Discriminator"`` by their short names, so this module maps
those to their fully-qualified paths before delegating to rsl-rl.
"""

from __future__ import annotations

import copy
from typing import Callable

# Short name -> fully-qualified "module:attr" path for AMP-provided classes.
_AMP_CLASS_MAP = {
    "AMP_PPO": "amp_rsl_rl.algorithms:AMP_PPO",
    "Discriminator": "amp_rsl_rl.networks:Discriminator",
    "ActorCriticMoE": "amp_rsl_rl.networks:ActorCriticMoE",
    "ActorMoE": "amp_rsl_rl.networks:ActorMoE",
    "MoEModel": "amp_rsl_rl.networks:MoEModel",
}


def resolve_amp_callable(name) -> Callable:
    """Resolve a callable, mapping AMP short names before falling back to rsl-rl."""
    from rsl_rl.utils import resolve_callable

    if callable(name):
        return name
    if isinstance(name, str) and name in _AMP_CLASS_MAP:
        name = _AMP_CLASS_MAP[name]
    return resolve_callable(name)


def resolve_amp_class(cfg: dict) -> tuple[Callable, dict]:
    """AMP-aware counterpart of :func:`rsl_rl.utils.resolve_class`."""
    class_cfg = copy.deepcopy(cfg)
    return resolve_amp_callable(class_cfg.pop("class_name")), class_cfg
