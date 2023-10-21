#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .gpvit import GPViT


__version__ = importlib.metadata.version("gpvit")
__all__ = ["GPViT"]
