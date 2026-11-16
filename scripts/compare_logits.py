# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Check that two loaders (or two pre_dtype settings) compute the same function: dump the logits of a model on a
fixed ~700-token text, then compare two dumps.

usage: python scripts/compare_logits.py dump MODEL_DIR TEXT_FILE OUT.npz
       python scripts/compare_logits.py cmp A.npz B.npz
"""
import sys

import numpy as np


def dump(model_dir, text_file, out):
    import mlx.core as mx
    from mlx_lm import load

    import macaq_mlx  # noqa: F401  (registers model_type qwen3_5_macaq)

    model, tok = load(model_dir)
    ids = tok.encode(open(text_file, encoding="utf-8").read()[:2500])[:700]
    logits = model(mx.array(ids)[None]).astype(mx.float32)
    mx.eval(logits)
    np.savez(out, logits=np.array(logits[0]), ids=np.array(ids))
    print("tokens", len(ids), "classes",
          sorted({type(m).__name__ for _, m in model.named_modules() if "Macaq" in type(m).__name__}))


def cmp(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    assert (a["ids"] == b["ids"]).all(), "the two dumps are not on the same tokens"
    la, lb = a["logits"], b["logits"]
    d = np.abs(la - lb)
    print(f"identical {np.array_equal(la, lb)}  max |diff| {d.max():.4g}  mean |diff| {d.mean():.4g}  "
          f"same top-1 {100 * (la.argmax(-1) == lb.argmax(-1)).mean():.2f}% of {len(la)} positions")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "dump":
        dump(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "cmp":
        cmp(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
