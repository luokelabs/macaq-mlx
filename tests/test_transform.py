# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Unit checks of the transform on random data (no model download). Run: python -m pytest tests"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

import macaq_mlx
from macaq_mlx.port_convert import check_hadamard, sylvester
from macaq_mlx.qwen3_5_macaq import MacaqLinear, MacaqQuantizedLinear, macaq_pre


def test_registered_with_mlx_lm():
    import importlib
    mod = importlib.import_module("mlx_lm.models.qwen3_5_macaq")
    assert mod.Model.__name__ == "Model" and hasattr(mod, "MacaqQuantizedLinear")
    assert macaq_mlx.register() is mod


def test_hadamard_is_sylvester_order():
    for block in (16, 64, 128):
        H = check_hadamard(block)
        assert mx.allclose(H @ H.T, mx.eye(block), atol=1e-5)


def _transformed(width=256, rows=96, block=64, seed=0):
    mx.random.seed(seed)
    W = mx.random.normal((rows, width))
    p = mx.exp(mx.random.normal((width,)) * 0.5).astype(mx.bfloat16).astype(mx.float32)
    H = sylvester(block)
    Wt = ((W / p).reshape(rows, width // block, block) @ H).reshape(rows, width)
    return W, p, Wt


def test_transform_is_exact_in_float32():
    W, p, Wt = _transformed()
    x = mx.random.normal((8, W.shape[1]))
    ref = x @ W.T
    got = macaq_pre(x, p, 64) @ Wt.T
    assert (mx.abs(got - ref).max() / mx.abs(ref).max()).item() < 1e-5


def test_macaq_linear_matches_plain_linear():
    W, p, Wt = _transformed()
    lin = MacaqLinear(W.shape[1], W.shape[0], False, 64)
    lin.weight, lin.pre_scale = Wt, p
    plain = nn.Linear(W.shape[1], W.shape[0], bias=False)
    plain.weight = W
    x = mx.random.normal((2, 5, W.shape[1]))
    assert (mx.abs(lin(x) - plain(x)).max() / mx.abs(plain(x)).max()).item() < 1e-5


def test_quantized_layer_is_not_a_quantized_linear_and_matches():
    W, p, Wt = _transformed()
    lin = MacaqLinear(W.shape[1], W.shape[0], False, 64)
    lin.weight, lin.pre_scale = Wt, p
    q = lin.to_quantized(group_size=64, bits=4)
    # engines that swap nn.QuantizedLinear for their own kernels must not catch this layer (see qwen3_5_macaq.py)
    assert isinstance(q, MacaqQuantizedLinear) and not isinstance(q, nn.QuantizedLinear)
    assert sorted(k for k, _ in tree_flatten(q.parameters())) == ["biases", "pre_scale", "scales", "weight"]
    x = mx.random.normal((3, W.shape[1]))
    deq = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4)
    ref = macaq_pre(x, p, 64) @ deq.T
    assert (mx.abs(q(x) - ref).max() / mx.abs(ref).max()).item() < 1e-4


def test_bf16_transform_close_to_float32():
    W, p, Wt = _transformed()
    lin = MacaqLinear(W.shape[1], W.shape[0], False, 64)
    lin.weight, lin.pre_scale = Wt, p
    q32 = lin.to_quantized(group_size=64, bits=4)
    q16 = lin.to_quantized(group_size=64, bits=4)
    q16.pre_scale = p.astype(mx.bfloat16)          # exact: p is rounded to bf16 by the converter
    assert np.array_equal(np.array(q16.pre_scale.astype(mx.float32)), np.array(p))
    x = mx.random.normal((4, W.shape[1])).astype(mx.bfloat16)
    a, b = q32(x).astype(mx.float32), q16(x).astype(mx.float32)
    assert (mx.abs(a - b).max() / mx.abs(a).max()).item() < 2e-2
    assert math.isfinite(mx.abs(b).max().item())
