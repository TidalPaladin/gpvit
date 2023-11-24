#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .gpvit import GPViT
from .layers import (
    GroupPropagationMLPMixer,
    GroupPropagationTransformer,
    MLPMixer,
    MLPMixerPooling,
    WindowAttention,
    WindowMLPMixer,
)


__version__ = importlib.metadata.version("gpvit")
__all__ = [
    "GPViT",
    "GroupPropagationMLPMixer",
    "GroupPropagationTransformer",
    "WindowAttention",
    "MLPMixer",
    "MLPMixerPooling",
    "WindowMLPMixer",
]
