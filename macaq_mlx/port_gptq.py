# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
#
# The GPTQ procedure below (inverse_hessian, gptq_layer, and the packing via mlx_lm.quant.gptq.quantize) follows
# mlx-lm 0.32.2 mlx_lm/quant/gptq.py, Copyright © 2025 Apple Inc., MIT License (see licenses/mlx-lm-MIT.txt).
# Changes from mlx-lm: the Hessian is taken on the input each MACaQ linear really sees after its transform,
# M_A = (x . p) . B, in float32 (mlx-lm's Catcher only takes plain nn.Linear and sees x before the transform);
# the embedding is quantized afterwards with plain MLX affine quantization, once per --emb-bits value.
"""MACaQ transform + GPTQ in MLX: start from the transformed, unquantized model (macaq-convert --no-quant), take
each linear's Hessian on the input it really sees after the transform, M_A = (x . p) . B, in float32, then run the
same column-by-column GPTQ as mlx-lm 0.32.2 quant/gptq.py (4-bit, group 64, affine, damp 1e-2) on M_W.
Calibration: every --ctx-token chunk of --calib. The embedding is quantized last, once per --emb-bits value, and
each variant is saved to OUT-e<bits>.

usage: macaq-gptq --src TRANSFORMED_BF16_DIR --out OUT_PREFIX --calib TEXT [--emb-bits 4] [--pre-dtype bfloat16]
"""
import argparse
import copy
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.quant.gptq import quantize as pack
from mlx_lm.utils import compute_bits_per_weight, load

import macaq_mlx
from macaq_mlx.port_convert import save_pack
from macaq_mlx.qwen3_5_macaq import PRE_DTYPES, MacaqLinear, macaq_pre

GPTQ_SOURCE = "GPTQ (mlx-lm 0.32.2 gptq.py procedure) with the Hessian of the transformed input; no codebook"


class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        w = module.weight.shape[1]
        self.H = mx.zeros((w, w), dtype=mx.float32)

    def __call__(self, x, *args, **kwargs):
        m = self.module
        xt = x.astype(mx.float32)
        if isinstance(m, MacaqLinear):
            xt = macaq_pre(xt, m.pre_scale, m.block)      # the basis the quantized weight lives in
        xf = xt.reshape(-1, xt.shape[-1])
        self.H = self.H + xf.T @ xf
        return m(x, *args, **kwargs)


def inverse_hessian(H, damp=1e-2):
    with mx.stream(mx.cpu):
        d = damp * mx.mean(mx.diag(H))
        diag = mx.arange(H.shape[0])
        H[diag, diag] += d
        H = mx.linalg.cholesky(H)
        H = mx.linalg.cholesky_inv(H)
        return mx.linalg.cholesky(H, upper=True)


def gptq_layer(W, Hinv, bits, group):
    n_bins = 2 ** bits - 1

    @mx.compile
    def err_of(w, d, scales, biases):
        q = scales * mx.clip(mx.round((w - biases) / scales), 0.0, n_bins) + biases
        return (w - q) / d

    all_s, all_b = [], []
    for i in range(0, W.shape[-1], group):
        j = i + group
        err = mx.zeros_like(W[..., i:j])
        _, s, b = mx.quantize(W[..., i:j], bits=bits, group_size=group)
        all_s.append(s)
        all_b.append(b)
        for k in range(i, j):
            e = err_of(W[..., k:k + 1], Hinv[k, k], s, b)
            W[..., k:j] -= e @ Hinv[k:k + 1, k:j]
            err[..., k - i:k - i + 1] = e
            mx.eval(err, W)
        W[..., j:] -= err @ Hinv[i:j, j:]
    s, b = mx.concatenate(all_s, -1), mx.concatenate(all_b, -1)
    return pack(W, bits, s, b), s, b


def calibration_ids(tok, calib, ctx):
    ids = mx.array(tok.encode(open(calib, encoding="utf-8").read()))
    if ids.size < ctx:
        raise SystemExit(f"{calib} has fewer than {ctx} tokens")
    return ids[: (ids.size // ctx) * ctx].reshape(-1, ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="output of macaq-convert --no-quant")
    ap.add_argument("--out", required=True, help="output prefix; each variant goes to OUT-e<emb bits>")
    ap.add_argument("--calib", required=True, help="calibration text file (UTF-8)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--emb-bits", default="4", help="comma-separated embedding bit widths, e.g. 4,6")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--damp", type=float, default=1e-2)
    ap.add_argument("--pre-dtype", default="bfloat16", choices=sorted(PRE_DTYPES),
                    help="precision of the run-time input transform written to the config")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    embs = [int(e) for e in a.emb_bits.split(",")]
    for e in embs:
        if os.path.exists(f"{a.out}-e{e}"):
            raise SystemExit(f"{a.out}-e{e} exists")
    mx.random.seed(a.seed)

    model, tok, config = load(a.src, return_config=True)
    if config.get("model_type") != macaq_mlx.MODEL_TYPE or config["macaq"].get("pre_dtype", "float32") != "float32":
        raise SystemExit("--src must be the unquantized output of macaq-convert --no-quant")
    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    layers = [(k, Catcher(m)) for k, m in leaves if type(m) in (nn.Linear, MacaqLinear)]
    n_macaq = sum(isinstance(c.module, MacaqLinear) for _, c in layers)
    print(f"GPTQ linears: {len(layers)} ({n_macaq} with the MACaQ transform)", flush=True)
    model.update_modules(tree_unflatten(layers))

    ids = calibration_ids(tok, a.calib, a.ctx)
    print(f"calibration: {ids.shape[0]} x {a.ctx} tokens", flush=True)
    t0 = time.time()
    for s in range(0, ids.shape[0], a.batch):
        model(ids[s:s + a.batch])
        mx.eval([c.H for _, c in layers])
        if (s // a.batch) % 16 == 0:
            print(f"hessians {min(s + a.batch, ids.shape[0])}/{ids.shape[0]} {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    done = []
    for n, (key, c) in enumerate(layers):
        Hinv = inverse_hessian(c.H, a.damp)
        c.H = mx.array(0.0)
        mx.eval(Hinv)
        m = c.module
        dtype = m.weight.dtype
        Wq, s, b = gptq_layer(m.weight.astype(mx.float32), Hinv, a.bits, a.group)
        q = m.to_quantized(group_size=a.group, bits=a.bits)   # MacaqQuantizedLinear keeps pre_scale (float32)
        q.weight, q.scales, q.biases = Wq, s.astype(dtype), b.astype(dtype)
        mx.eval(q.parameters())
        done.append((key, q))
        if n % 20 == 0:
            print(f"quantized {n + 1}/{len(layers)} {time.time() - t0:.0f}s", flush=True)
    model.update_modules(tree_unflatten(done))

    rest = [k for k, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
            if hasattr(m, "to_quantized") and not isinstance(m, nn.QuantizedLinear)]
    print("left for plain quantization:", rest, flush=True)
    if rest != ["language_model.model.embed_tokens"]:
        raise SystemExit(f"unexpected unquantized modules: {rest}")
    emb = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))[rest[0]]
    for e in embs:
        model.update_modules(tree_unflatten([(rest[0], emb.to_quantized(group_size=a.group, bits=e))]))
        cfg = copy.deepcopy(config)
        cfg["quantization"] = {"group_size": a.group, "bits": a.bits, "mode": "affine"}
        if e != a.bits:
            cfg["quantization"][rest[0]] = {"group_size": a.group, "bits": e}
        cfg["quantization_config"] = cfg["quantization"]
        cfg["macaq"]["source"] = f"huawei-csl/MACaQ e9d4f89 D1 pre-scale + block Hadamard; {GPTQ_SOURCE}"
        cfg["macaq"]["gptq_calibration"] = {"file": os.path.basename(a.calib), "tokens": int(ids.size),
                                            "damp": a.damp}
        cfg["macaq"]["pre_dtype"] = a.pre_dtype
        print(f"emb {e}-bit: {compute_bits_per_weight(model):.3f} bits per weight", flush=True)
        save_pack(f"{a.out}-e{e}", a.src, model, tok, cfg, donate_model=False)
        print("saved", f"{a.out}-e{e}", flush=True)


if __name__ == "__main__":
    main()
