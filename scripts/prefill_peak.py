# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Peak and transient memory of one long prefill (no generation).

usage: python scripts/prefill_peak.py MODEL_DIR TOKENS OUT_JSON [--text data/eval.txt]
transient_gb = peak during the prefill minus the memory held before it (weights), in GiB.
"""
import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load

import macaq_mlx  # noqa: F401  (registers model_type qwen3_5_macaq)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("tokens", type=int)
    ap.add_argument("out")
    ap.add_argument("--text", default="data/eval.txt")
    a = ap.parse_args()

    model, tok = load(a.model)
    ids = tok.encode(open(a.text, encoding="utf-8").read())[:a.tokens]
    mx.eval(model(mx.array(ids[:64])[None]))  # warm-up
    mx.synchronize()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    t = time.perf_counter()
    out = model(mx.array(ids)[None])
    mx.eval(out)
    dt = time.perf_counter() - t
    res = {"tokens": len(ids), "weights_gb": round(base / 2**30, 3),
           "peak_gb": round(mx.get_peak_memory() / 2**30, 3),
           "transient_gb": round((mx.get_peak_memory() - base) / 2**30, 3),
           "seconds": round(dt, 3), "tok_per_s": round(len(ids) / dt, 1)}
    json.dump(res, open(a.out, "w"))
    print(res)


if __name__ == "__main__":
    main()
