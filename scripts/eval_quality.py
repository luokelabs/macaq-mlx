# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Quantization fidelity of an MLX model against a reference (normally the bf16 original): perplexity, mean KL
divergence and same-top-1 share, scored like llama-perplexity (CTX-token chunks of the evaluation text, the second
half of each chunk scored), plus an optional homophone candidate-ranking task scored like llama-perplexity
--hellaswag (mean log-prob of the tokens after the candidates' common prefix).

usage: python scripts/eval_quality.py REF_DIR TEST_DIR OUT_JSON [--text data/eval.txt] [--chunks 100]
                                      [--homophone data/homophone-clean.jsonl]
"""
import argparse
import json
import math
import time

import mlx.core as mx
from mlx_lm import load

import macaq_mlx  # noqa: F401  (registers model_type qwen3_5_macaq)

CTX = 512


def logprobs(model, toks):
    logits = model(mx.array(toks)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--text", default="data/eval.txt")
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--homophone", help="homophone task file (jsonl); skipped when not given")
    a = ap.parse_args()

    ref, tok = load(a.ref)
    test, _ = load(a.test)
    ids = tok.encode(open(a.text, encoding="utf-8").read())
    n_chunks = min(a.chunks, len(ids) // CTX)
    nll_ref = nll_test = kl = 0.0
    top1_same = count = 0
    t0 = time.time()
    for c in range(n_chunks):
        toks = ids[c * CTX:(c + 1) * CTX]
        lp_r, lp_t = logprobs(ref, toks), logprobs(test, toks)
        first = CTX // 2
        sl = slice(first - 1, CTX - 1)          # predictions at first-1 .. CTX-2 score tokens first .. CTX-1
        tgt = mx.array(toks[first:])
        r, t = lp_r[sl], lp_t[sl]
        nll_ref += -mx.take_along_axis(r, tgt[:, None], axis=-1).sum().item()
        nll_test += -mx.take_along_axis(t, tgt[:, None], axis=-1).sum().item()
        kl += (mx.exp(r) * (r - t)).sum().item()
        top1_same += (mx.argmax(r, -1) == mx.argmax(t, -1)).sum().item()
        count += CTX - first
    res = {"chunks": n_chunks, "tokens_scored": count,
           "ppl_ref": math.exp(nll_ref / count), "ppl_test": math.exp(nll_test / count),
           "kl_mean": kl / count, "same_top1_pct": 100 * top1_same / count, "seconds": time.time() - t0}

    if a.homophone:
        items = [json.loads(line) for line in open(a.homophone, encoding="utf-8")]

        def pick(model, it):
            seqs = [tok.encode(it["context"] + c) for c in it["candidates"]]
            common = 0
            while all(len(s) > common for s in seqs) and len({s[common] for s in seqs}) == 1:
                common += 1
            scores = []
            for s in seqs:
                lp = logprobs(model, s)
                vals = [lp[i - 1, s[i]].item() for i in range(max(common, 1), len(s))]
                scores.append(sum(vals) / len(vals))
            return max(range(len(scores)), key=lambda i: scores[i])

        right_ref = right_test = agree = 0
        for it in items:
            pr, pt = pick(ref, it), pick(test, it)
            right_ref += pr == it["label"]
            right_test += pt == it["label"]
            agree += pr == pt
        res.update({"homophone_n": len(items), "homophone_acc_ref": 100 * right_ref / len(items),
                    "homophone_acc_test": 100 * right_test / len(items),
                    "homophone_same_pick_pct": 100 * agree / len(items)})
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
