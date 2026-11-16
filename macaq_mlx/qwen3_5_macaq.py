# Copyright 2026 the macaq-mlx authors.
# Derived from MACaQ (https://github.com/huawei-csl/MACaQ, commit e9d4f89), Copyright (c) 2026 the MACAQ authors,
# licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
#
# Changes from MACaQ: the weight-only activation transform that MACaQ's llama.cpp fork applies in
# llama-graph.cpp (build_macaq_pre) is re-implemented here as an mlx-lm model type for Qwen3.5, on top of MLX
# affine quantization. MACaQ's codebook formats (IQ4_XS + vector-codebook refinement) are not ported.
# The structure follows mlx-lm's prism_hadamard_qwen35.py (MIT License, Copyright © 2026 Apple Inc.; see
# licenses/mlx-lm-MIT.txt).
"""mlx-lm model type ``qwen3_5_macaq``: Qwen3.5 with the MACaQ input transform on selected linears.

Each listed linear stores M_W = (W / p) . B, quantized with MLX affine quantization, where p is a per-input-channel
pre-scale and B a block-diagonal normalized Sylvester Hadamard matrix. At run time the input enters the same basis,
M_A = (x . p) . B, so M_W M_A^T = W x^T.

Use it in one of two ways:
  * ``import macaq_mlx`` before ``mlx_lm.load(...)`` (registers this module as ``mlx_lm.models.qwen3_5_macaq``);
  * copy this single file into ``<site-packages>/mlx_lm/models/`` (for engines that load mlx-lm models themselves).
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.models import qwen3_5


@dataclass
class ModelArgs(qwen3_5.ModelArgs):
    macaq: Dict[str, Any] = field(default_factory=dict)


# The pack's config may set macaq.pre_dtype: the precision of the input transform (scale, then Hadamard).
# "float32" (the default, and what packs without the key get) converts each input to float32 first. "bfloat16"
# keeps the activations' 16-bit type: less transient memory and less time per layer. On the Qwen3.5-0.8B and 9B
# ports the quality was unchanged (KL between the two settings 0.0008 on the 9B; see docs/report.html).
PRE_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


def macaq_pre(x, pre_scale, block):
    """x . p, then the block Hadamard, computed in pre_scale's dtype; returns x's dtype."""
    shape, dtype = x.shape, x.dtype
    x = x.astype(pre_scale.dtype) * pre_scale
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block))
    return x.reshape(shape).astype(dtype)


class MacaqQuantizedLinear(nn.Module):
    """A quantized MACaQ linear: the input transform, then MLX affine quantized_matmul.

    Deliberately not an nn.QuantizedLinear subclass. Engines that route QuantizedLinear layers to their own kernels
    (oMLX 0.7.2's Qwen3.5 prefill patches check isinstance and call qmm on the raw input) would skip the transform
    and silently compute a different function. The parameters keep QuantizedLinear's names, so packs load as before.
    """

    def __init__(self, input_dims, output_dims, group_size=64, bits=4, mode="affine", block=64, pre_dtype=mx.float32):
        super().__init__()
        if mode != "affine":
            raise ValueError("MACaQ packs use MLX affine quantization")
        self.group_size, self.bits, self.mode, self.block = group_size, bits, mode, block
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size), dtype=mx.float16)
        self.biases = mx.zeros((output_dims, input_dims // group_size), dtype=mx.float16)
        self.pre_scale = mx.ones((input_dims,), dtype=pre_dtype)  # Model.sanitize casts the stored one to pre_dtype
        self.freeze()

    def _extra_repr(self):
        out_dims, packed = self.weight.shape
        return (f"input_dims={packed * 32 // self.bits}, output_dims={out_dims}, group_size={self.group_size}, "
                f"bits={self.bits}, mode={self.mode}, block={self.block}, pre_dtype={self.pre_scale.dtype}")

    def __call__(self, x):
        return mx.quantized_matmul(macaq_pre(x, self.pre_scale, self.block), self["weight"], scales=self["scales"],
                                   biases=self["biases"], transpose=True, group_size=self.group_size, bits=self.bits,
                                   mode=self.mode)


class MacaqLinear(nn.Linear):
    """An unquantized MACaQ linear (used while converting): the input transform, then a plain Linear."""

    def __init__(self, input_dims, output_dims, bias=False, block=64, pre_dtype=mx.float32):
        super().__init__(input_dims, output_dims, bias=bias)
        self.pre_scale = mx.ones((input_dims,), dtype=pre_dtype)
        self.block = block

    def __call__(self, x):
        return super().__call__(macaq_pre(x, self.pre_scale, self.block))

    def to_quantized(self, group_size=None, bits=None, mode="affine", quantize_input=False):
        if quantize_input:
            raise ValueError("MACaQ linears are weight-only")
        group_size, bits = group_size or 64, bits or 4
        out_d, in_d = self.weight.shape
        q = MacaqQuantizedLinear(in_d, out_d, group_size, bits, mode, self.block, self.pre_scale.dtype)
        q.weight, q.scales, q.biases = mx.quantize(self.weight, group_size, bits, mode=mode)
        q.pre_scale = self.pre_scale
        return q


class Model(qwen3_5.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        cfg = args.macaq
        block = int(cfg.get("hadamard_block", 0))
        if block < 2 or block & (block - 1):
            raise ValueError(f"MACaQ pack needs a power-of-two hadamard_block, got {block}")
        if cfg.get("pre_dtype", "float32") not in PRE_DTYPES:
            raise ValueError(f"MACaQ pre_dtype must be one of {sorted(PRE_DTYPES)}, got {cfg['pre_dtype']}")
        self.pre_dtype = PRE_DTYPES[cfg.get("pre_dtype", "float32")]
        modules = dict(self.language_model.named_modules())
        repl = []
        for path in cfg.get("linears", []):
            m = modules.get(path)
            if type(m) is not nn.Linear or "bias" in m:
                raise ValueError(f"MACaQ linear is not a plain bias-free Linear: {path}")
            out_d, in_d = m.weight.shape
            if in_d % block:
                raise ValueError(f"MACaQ linear width {in_d} is not a multiple of {block}: {path}")
            repl.append((path, MacaqLinear(in_d, out_d, False, block, self.pre_dtype)))
        if not repl:
            raise ValueError("MACaQ pack lists no linears")
        self.language_model.update_modules(tree_unflatten(repl))

    def sanitize(self, weights):
        # packs store pre_scale in float32; the transform runs in pre_scale's dtype
        return {k: v.astype(self.pre_dtype) if k.endswith(".pre_scale") else v
                for k, v in super().sanitize(weights).items()}
