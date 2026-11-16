# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
# Uses the MACaQ-derived transform in port_convert.py and the mlx-lm-derived GPTQ in port_gptq.py (see both).
"""MACaQ -> MLX in one process, for any size (used for Qwen3.5-9B): the same two steps as macaq-convert --no-quant
followed by macaq-gptq, without writing the transformed bf16 model to disk.

  1. MACaQ D1 pre-scale + 64-wide Sylvester Hadamard on every linear of the decoder layers, statistics from
     --calib (default 128 chunks x 512 tokens) on the bf16 model;
  2. GPTQ (mlx-lm 0.32.2 gptq.py procedure: 4-bit, group 64, affine, damp 1e-2) with the Hessian of the input each
     linear really sees after the transform, every 512-token chunk of --calib.

An untied output layer (lm_head, e.g. Qwen3.5-9B) is not one of MACaQ's main-layer linears, so it gets plain GPTQ
(Hessian of its own input). The embedding gets plain MLX affine quantization (--emb-bits, default 4).

Memory: the Hessians are collected --group-layers decoder layers at a time (each pass stops the forward after its
last layer), and every linear keeps its bf16 weight until all passes are done, so each Hessian is still taken on
the activations of the unquantized model, as in the one-pass macaq-gptq. Each finished pass is saved to --ckpt, so a
run that was stopped (for example by scripts/watchdog.sh) resumes at the next pass. Qwen3.5-9B with 8 layers per
pass peaked at about 33 GB.

usage: macaq-port --hf BF16_DIR --out DIR --calib TEXT [--stats FILE] [--group-layers 8] [--pre-dtype bfloat16]
"""
import argparse
import os
import re
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import compute_bits_per_weight

from macaq_mlx import port_convert as pc
from macaq_mlx import port_gptq as pg
from macaq_mlx.qwen3_5_macaq import PRE_DTYPES, MacaqLinear

LM_HEAD = "language_model.lm_head"
EMBED = "language_model.model.embed_tokens"


class StopForward(Exception):
    pass


class Raiser(nn.Module):
    def __call__(self, *args, **kwargs):
        raise StopForward


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="original checkpoint (HF folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True, help="calibration text file (UTF-8)")
    ap.add_argument("--stats", help="cache file for the activation statistics")
    ap.add_argument("--expect", type=int, help="number of linears to transform (sanity check)")
    ap.add_argument("--chunks", type=int, default=128, help="calibration chunks for the statistics")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--emb-bits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--damp", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--group-layers", type=int, default=8, help="decoder layers per Hessian pass")
    ap.add_argument("--cache-gb", type=int, default=4, help="MLX buffer cache limit")
    ap.add_argument("--pre-dtype", default="bfloat16", choices=sorted(PRE_DTYPES),
                    help="precision of the run-time input transform written to the config")
    ap.add_argument("--ckpt", help="directory for the per-pass GPTQ results (default OUT.ckpt)")
    ap.add_argument("--keep-ckpt", action="store_true", help="keep the checkpoints after a successful save")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    ckpt_dir = a.ckpt or a.out.rstrip("/") + ".ckpt"
    mx.random.seed(a.seed)
    mx.set_cache_limit(a.cache_gb << 30)
    T0 = time.time()
    H = pc.check_hadamard(a.block)

    # ---- step 1: statistics and transform (as macaq-convert, without saving the transformed bf16 model)
    model, tok, config = pc.load_bf16(a.hf)
    lm = model.language_model
    targets = pc.find_targets(model, a.expect)
    stats, tokens = pc.get_stats(model, tok, targets, a.calib, a.stats, a.chunks, a.ctx)
    print("calibration tokens:", tokens, f"{time.time() - T0:.0f}s", flush=True)
    pc.transform(model, targets, stats, a.alpha, a.block, H)
    del stats
    mx.clear_cache()
    config = pc.macaq_config(config, targets, a.alpha, a.block, a.calib, tokens)

    # ---- step 2: GPTQ in the transformed basis, Hessians collected a group of layers at a time
    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    order = [k for k, m in leaves.items() if type(m) in (nn.Linear, MacaqLinear)]
    n_layers = len(lm.model.layers)

    def layer_of(k):
        m = re.match(r"language_model\.model\.layers\.(\d+)\.", k)
        return int(m.group(1)) if m else n_layers          # lm_head goes with the last group

    n_macaq = sum(isinstance(leaves[k], MacaqLinear) for k in order)
    plain = [k for k in order if not isinstance(leaves[k], MacaqLinear)]
    print(f"GPTQ linears: {len(order)} ({n_macaq} with the MACaQ transform; plain: {plain})", flush=True)
    if n_macaq != len(targets) or plain not in ([], [LM_HEAD]):
        raise SystemExit(f"unexpected linears outside the decoder layers: {plain}")

    ids = pg.calibration_ids(tok, a.calib, a.ctx)
    print(f"calibration: {ids.shape[0]} x {a.ctx} tokens, {a.group_layers} layers per pass", flush=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    n_done, t0 = 0, time.time()
    for g0 in range(0, n_layers, a.group_layers):
        g1 = min(g0 + a.group_layers, n_layers)
        keys = [k for k in order if g0 <= layer_of(k) < g1 or (g1 == n_layers and layer_of(k) == n_layers)]
        ck = os.path.join(ckpt_dir, f"pass-{g0:02d}-{g1 - 1:02d}.safetensors")
        if os.path.exists(ck):
            n_done += len(keys)
            print(f"pass layers {g0}-{g1 - 1}: from checkpoint {ck}", flush=True)
            continue
        cat = [(k, pg.Catcher(leaves[k])) for k in keys]
        stop = [(k, Raiser()) for k in order if layer_of(k) == g1] if g1 < n_layers else []
        model.update_modules(tree_unflatten(cat + stop))
        for s in range(0, ids.shape[0], a.batch):
            try:
                model(ids[s:s + a.batch])
            except StopForward:
                pass
            mx.eval([c.H for _, c in cat])
        restore = [(k, leaves[k]) for k in keys] + [(k, leaves[k]) for k, _ in stop]
        model.update_modules(tree_unflatten(restore))
        print(f"pass layers {g0}-{g1 - 1}: {len(cat)} hessians {time.time() - t0:.0f}s", flush=True)
        out = {}
        for i in range(len(cat)):
            key, c = cat[i]
            cat[i] = None
            Hinv = pg.inverse_hessian(c.H, a.damp)
            c.H = mx.array(0.0)
            mx.eval(Hinv)
            dtype = c.module.weight.dtype
            Wq, sc, b = pg.gptq_layer(c.module.weight.astype(mx.float32), Hinv, a.bits, a.group)
            out[key + ".weight"], out[key + ".scales"], out[key + ".biases"] = Wq, sc.astype(dtype), b.astype(dtype)
            mx.eval(out[key + ".weight"], out[key + ".scales"], out[key + ".biases"])
            del c, Hinv, Wq, sc, b
            n_done += 1
            if n_done % 10 == 1 or n_done == len(order):
                print(f"quantized {n_done}/{len(order)} {key} {time.time() - t0:.0f}s "
                      f"active {mx.get_active_memory() / 2**30:.1f} GB peak {mx.get_peak_memory() / 2**30:.1f} GB",
                      flush=True)
        mx.save_safetensors(ck + ".tmp.safetensors", out)
        os.replace(ck + ".tmp.safetensors", ck)
        print(f"checkpoint {ck}", flush=True)
        del cat, out
        mx.clear_cache()

    # all passes done: swap every linear for its quantized form from the checkpoints
    q_arrays = {}
    for f in sorted(os.listdir(ckpt_dir)):
        if f.startswith("pass-") and f.endswith(".safetensors") and ".tmp." not in f:
            q_arrays.update(mx.load(os.path.join(ckpt_dir, f)))
    if len(q_arrays) != 3 * len(order):
        raise SystemExit(f"checkpoints in {ckpt_dir} hold {len(q_arrays)} arrays, expected {3 * len(order)}")
    done = []
    for key in order:
        q = leaves[key].to_quantized(group_size=a.group, bits=a.bits)   # MacaqQuantizedLinear keeps pre_scale
        q.weight, q.scales, q.biases = q_arrays[key + ".weight"], q_arrays[key + ".scales"], q_arrays[key + ".biases"]
        done.append((key, q))
    model.update_modules(tree_unflatten(done))
    del done, leaves, q_arrays
    mx.clear_cache()

    rest = [k for k, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
            if hasattr(m, "to_quantized") and not isinstance(m, nn.QuantizedLinear)]
    print("left for plain quantization:", rest, flush=True)
    if rest != [EMBED]:
        raise SystemExit(f"unexpected unquantized modules: {rest}")
    emb = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))[EMBED]
    model.update_modules(tree_unflatten([(EMBED, emb.to_quantized(group_size=a.group, bits=a.emb_bits))]))
    config["quantization"] = {"group_size": a.group, "bits": a.bits, "mode": "affine"}
    if a.emb_bits != a.bits:
        config["quantization"][EMBED] = {"group_size": a.group, "bits": a.emb_bits}
    config["quantization_config"] = config["quantization"]
    config["macaq"]["source"] = f"{pc.SOURCE}; {pg.GPTQ_SOURCE}"
    config["macaq"]["gptq_calibration"] = {"file": os.path.basename(a.calib), "tokens": int(ids.size),
                                           "damp": a.damp}
    if plain:
        config["macaq"]["lm_head"] = "plain GPTQ (not a MACaQ main-layer linear)"
    config["macaq"]["pre_dtype"] = a.pre_dtype
    print(f"{compute_bits_per_weight(model):.3f} bits per weight", flush=True)
    pc.save_pack(a.out, a.hf, model, tok, config)
    print("saved", a.out, f"{time.time() - T0:.0f}s", flush=True)
    if not a.keep_ckpt:
        shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    main()
