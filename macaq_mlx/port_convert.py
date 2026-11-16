# Copyright 2026 the macaq-mlx authors.
# Derived from MACaQ (https://github.com/huawei-csl/MACaQ, commit e9d4f89), Copyright (c) 2026 the MACAQ authors,
# licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
#
# Changes from MACaQ: the D1 pre-scale (quantize/macaq_balance.py) and the block-Hadamard weight transform
# (llamacpp/llama.cpp/macaq_math.py) are re-implemented with MLX on mlx-lm's Qwen3.5 model; the activation
# statistics are collected with the bf16 MLX model instead of llama-imatrix; the result is saved as an mlx-lm model
# (model_type qwen3_5_macaq) with MLX affine quantization instead of a GGUF with MACaQ codebooks.
"""MACaQ -> MLX, step 1: the MACaQ weight transform (D1 pre-scale + block Hadamard), then either plain MLX affine
quantization (round to nearest) or no quantization (--no-quant, the input for macaq-gptq).

Same formulas as huawei-csl/MACaQ e9d4f89:
  quantize/macaq_balance.py    p = clamp(std_over_rows(W)^a / rms(x)^(1-a), 1e-4, 1e4), rounded to bf16
  macaq_math.py                M_W = (W / p) reshaped to 64-wide input blocks @ normalized Sylvester Hadamard
  prepare_weight_only_gguf.py  every linear of the main decoder layers; statistics = mean of x^2 per input channel
The statistics are collected with the bf16 MLX model on --calib (default 128 chunks x 512 tokens).

usage: macaq-convert --hf BF16_DIR --out DIR --calib TEXT [--stats FILE] [--bits 4] [--group 64] [--emb-bits 4]
                     [--no-quant] [--pre-dtype bfloat16]
"""
import argparse
import math
import os
import re
import shutil
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path, tree_unflatten
from mlx_lm.utils import load, quantize_model, save

import macaq_mlx
from macaq_mlx.qwen3_5_macaq import PRE_DTYPES, MacaqLinear, macaq_pre

# every linear of the Qwen3.5 decoder layers (linear-attention and full-attention layers, and the MLP)
TARGET = re.compile(r"model\.layers\.(\d+)\.(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
                    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
SOURCE = "huawei-csl/MACaQ e9d4f89 D1 pre-scale + block Hadamard"


def sylvester(b):
    h = np.array([[(-1) ** bin(i & j).count("1") for j in range(b)] for i in range(b)], dtype=np.float32)
    return mx.array(h / math.sqrt(b))


def check_hadamard(block):
    """The transform must use exactly MACaQ's Hadamard: check MLX's transform against the Sylvester matrix."""
    H = sylvester(block)
    got = mx.hadamard_transform(mx.eye(block), scale=1 / math.sqrt(block))
    assert mx.abs(got - H).max().item() < 1e-6, "mx.hadamard_transform is not the Sylvester ordering"
    return H


class StatLinear(nn.Module):
    def __init__(self, lin, acc, path):
        super().__init__()
        self.lin = lin
        object.__setattr__(self, "_acc", acc)     # plain attributes, not part of the module tree
        object.__setattr__(self, "_path", path)

    def __call__(self, x):
        flat = x.astype(mx.float32).reshape(-1, x.shape[-1])
        s, n = self._acc[self._path]
        self._acc[self._path] = (s + flat.square().sum(0), n + flat.shape[0])
        return self.lin(x)


def collect_stats(model, tok, targets, calib, chunks, ctx):
    """Sum of x^2 per input channel of every target linear, over the first `chunks` ctx-token chunks of calib."""
    lm = model.language_model
    mods = dict(lm.named_modules())
    acc = {p: (mx.zeros((mods[p].weight.shape[1],), dtype=mx.float32), 0) for p in targets}
    lm.update_modules(tree_unflatten([(p, StatLinear(mods[p], acc, p)) for p in targets]))
    ids = tok.encode(open(calib, encoding="utf-8").read())
    n = min(chunks, len(ids) // ctx)
    if n == 0:
        raise SystemExit(f"{calib} has fewer than {ctx} tokens")
    t0 = time.time()
    for c in range(n):
        model(mx.array(ids[c * ctx:(c + 1) * ctx])[None])
        mx.eval([a[0] for a in acc.values()])
        if c % 32 == 0:
            print(f"stats chunk {c + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    lm.update_modules(tree_unflatten([(p, mods[p]) for p in targets]))
    return {p: (acc[p][0], acc[p][1]) for p in targets}, n * ctx


def load_bf16(path):
    """Load an HF checkpoint lazily and cast it to bf16, as mlx_lm.convert does."""
    model, tok, config = load(path, return_config=True, lazy=True)
    cast = getattr(model, "cast_predicate", lambda _: True)
    model.update(tree_map_with_path(
        lambda k, v: v.astype(mx.bfloat16) if cast(k) and mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters()))
    return model, tok, config


def find_targets(model, expect=None):
    lm = model.language_model
    targets = [p for p, m in lm.named_modules() if TARGET.fullmatch(p) and type(m) is nn.Linear]
    print("transformed linears:", len(targets), flush=True)
    if not targets:
        raise SystemExit("no Qwen3.5 decoder linears found: is this a dense Qwen3.5 checkpoint?")
    if expect is not None and len(targets) != expect:
        raise SystemExit(f"expected {expect} linears, found {len(targets)}")
    return targets


def get_stats(model, tok, targets, calib, stats_file, chunks, ctx):
    """Activation statistics, from the cache file when it exists, else collected (and cached if a file is given)."""
    if stats_file and os.path.exists(stats_file):
        raw = mx.load(stats_file)
        stats = {p: (raw[p + ".sum2"], int(raw[p + ".count"].item())) for p in targets}
        return stats, int(raw["_tokens"].item())
    if not calib:
        raise SystemExit("--calib is required unless --stats points to an existing statistics file")
    stats, tokens = collect_stats(model, tok, targets, calib, chunks, ctx)
    if stats_file:
        out = {"_tokens": mx.array([tokens])}
        for p, (s, n) in stats.items():
            out[p + ".sum2"], out[p + ".count"] = s, mx.array([n])
        mx.save_safetensors(stats_file, out)
    return stats, tokens


def transform(model, targets, stats, alpha, block, H):
    """Replace every target linear by a MacaqLinear holding M_W = (W / p) . B (bf16) and p (float32).

    Checks the equivalence (W/p)B . (x p)B == W x in float32 on random inputs; returns the worst relative error."""
    lm = model.language_model
    mods = dict(lm.named_modules())
    repl, worst = [], 0.0
    for p in targets:
        W = mods[p].weight.astype(mx.float32)
        rows, width = W.shape
        sum2, count = stats[p]
        w_std = mx.maximum(mx.sqrt(((W - W.mean(0)) ** 2).sum(0) / (rows - 1)), 1e-8)
        act_rms = mx.sqrt(mx.maximum(sum2 / max(count, 1), 1e-12))
        pre = mx.clip(w_std ** alpha / act_rms ** (1 - alpha), 1e-4, 1e4)
        pre = pre.astype(mx.bfloat16).astype(mx.float32)
        Wt = ((W / pre).reshape(rows, width // block, block) @ H).reshape(rows, width)
        new = MacaqLinear(width, rows, False, block)
        new.weight = Wt.astype(mx.bfloat16)
        new.pre_scale = pre
        x = mx.random.normal((4, width))
        ref = x @ W.T
        err = mx.abs(macaq_pre(x, pre, block) @ Wt.T - ref).max() / mx.abs(ref).max()
        mx.eval(new.parameters(), err)
        worst = max(worst, err.item())
        repl.append((p, new))
    lm.update_modules(tree_unflatten(repl))
    print(f"float32 equivalence, worst relative error: {worst:.2e}", flush=True)
    if worst >= 1e-3:
        raise SystemExit("the transformed linears do not reproduce the original ones")
    return worst


def save_pack(out, src, model, tok, config, **kw):
    """mlx-lm save, plus a copy of the loader so the folder is self-contained."""
    save(out, src, model, tok, config, **kw)
    shutil.copy(macaq_mlx.LOADER_FILE, os.path.join(out, "qwen3_5_macaq.py"))


def macaq_config(config, targets, alpha, block, calib, tokens):
    config["model_type"] = macaq_mlx.MODEL_TYPE
    if config.get("architectures"):
        # engines that pick a vision path by architecture name (e.g. oMLX) must see a text-only model
        config["architectures"] = ["Qwen3_5MacaqForCausalLM"]
    config["macaq"] = {"hadamard_block": block, "alpha": alpha, "linears": targets, "source": SOURCE,
                       "calibration": {"file": os.path.basename(calib) if calib else None, "tokens": tokens}}
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="original checkpoint (HF folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", help="calibration text file (UTF-8)")
    ap.add_argument("--stats", help="cache file for the activation statistics")
    ap.add_argument("--expect", type=int, help="number of linears to transform (sanity check)")
    ap.add_argument("--chunks", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--emb-bits", type=int, default=4)
    ap.add_argument("--no-quant", action="store_true", help="save the transformed bf16 model (input for macaq-gptq)")
    ap.add_argument("--pre-dtype", default="bfloat16", choices=sorted(PRE_DTYPES),
                    help="precision of the run-time input transform written to the config (quantized output only)")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    mx.random.seed(a.seed)
    H = check_hadamard(a.block)

    model, tok, config = load_bf16(a.hf)
    targets = find_targets(model, a.expect)
    stats, tokens = get_stats(model, tok, targets, a.calib, a.stats, a.chunks, a.ctx)
    print("calibration tokens:", tokens, flush=True)
    transform(model, targets, stats, a.alpha, a.block, H)

    config = macaq_config(config, targets, a.alpha, a.block, a.calib, tokens)
    if a.no_quant:
        # keep the float32 transform: macaq-gptq takes its Hessians in exactly the basis the weights live in
        config["macaq"]["source"] += "; not quantized"
    else:
        def pred(path, module):
            if path.endswith("embed_tokens") and a.emb_bits != a.bits:
                return {"group_size": a.group, "bits": a.emb_bits, "mode": "affine"}
            return True
        config["macaq"]["source"] += "; MLX affine RTN, no vector-codebook refinement"
        config["macaq"]["pre_dtype"] = a.pre_dtype
        model, config = quantize_model(model, config, a.group, a.bits, mode="affine", quant_predicate=pred)
    save_pack(a.out, a.hf, model, tok, config)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
