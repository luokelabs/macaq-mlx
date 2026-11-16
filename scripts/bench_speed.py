# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""MLX speed, matched to llama-bench -p 512 -n 128 -r 5: a prompt of 512 tokens from the evaluation text, 128
generated tokens (greedy), 5 runs after one warm-up; medians of prompt and generation tokens/s, peak memory.

usage: python scripts/bench_speed.py MODEL_DIR OUT_JSON [--text data/eval.txt]
"""
import argparse
import json
import statistics

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

import macaq_mlx  # noqa: F401  (registers model_type qwen3_5_macaq)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--text", default="data/eval.txt")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()

    model, tok = load(a.model)
    ids = tok.encode(open(a.text, encoding="utf-8").read())[:a.prompt_tokens]
    greedy = make_sampler(temp=0.0)
    runs = []
    for i in range(a.runs + 1):
        mx.clear_cache()
        last = None
        for last in stream_generate(model, tok, ids, max_tokens=a.gen_tokens, sampler=greedy):
            pass
        if i:  # the first run is a warm-up
            runs.append({"prompt_tps": last.prompt_tps, "gen_tps": last.generation_tps,
                         "gen_tokens": last.generation_tokens, "peak_gb": last.peak_memory})
    res = {"prompt_tokens": len(ids),
           "pp_tps_median": statistics.median(r["prompt_tps"] for r in runs),
           "tg_tps_median": statistics.median(r["gen_tps"] for r in runs),
           "peak_gb": max(r["peak_gb"] for r in runs), "runs": runs}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
