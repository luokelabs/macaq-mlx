# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
"""MACaQ's pre-scale + block-Hadamard weight transform, ported to MLX for Qwen3.5, with GPTQ 4-bit.

Importing this package registers the ``qwen3_5_macaq`` model type with mlx-lm, so ``mlx_lm.load`` can open
converted models without copying any file into the mlx-lm installation::

    import macaq_mlx  # noqa: F401  (registers model_type "qwen3_5_macaq")
    from mlx_lm import load
    model, tokenizer = load("path/to/Qwen3.5-9B-macaq4")
"""
import os
import sys

from . import qwen3_5_macaq

__version__ = "0.1.0"
MODEL_TYPE = "qwen3_5_macaq"
LOADER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_5_macaq.py")


def register():
    """Make mlx-lm resolve model_type "qwen3_5_macaq" to this package's loader.

    mlx-lm imports ``mlx_lm.models.<model_type>``; an entry in sys.modules satisfies that import. If a copy of the
    loader was already imported from the mlx-lm installation, that copy is kept and returned.
    """
    return sys.modules.setdefault(f"mlx_lm.models.{MODEL_TYPE}", qwen3_5_macaq)


register()
