# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Public benchmarks for a base MLX model through lm-evaluation-harness (mlx-lm's MLXLM wrapper), zero-shot, no chat
template, keeping every question's result so two models can be compared item by item (compare_lmeval.py).

usage: python bench/lmeval/lmeval_run.py MODEL_DIR OUT.json TASK[,TASK...] [LIMIT]
  LIMIT: N questions, or a share like 0.4, per task (per subject for MMLU, C-Eval, CMMLU); lm-eval takes the first ones
  LMEVAL_CACHE=path keeps every answer in lm-eval's sqlite cache, so a run that was stopped picks up where it was.
Tasks under ./tasks next to this file are found too (cmmlu_local: CMMLU from its CSV zip, see make_cmmlu_tasks.py).
"""
import json
import os
import sys
import time

import lm_eval
from lm_eval.tasks import TaskManager
from mlx_lm.evaluate import MLXLM

import macaq_mlx  # noqa: F401  (registers model_type qwen3_5_macaq)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    model_dir, out, tasks = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    limit = (float(sys.argv[4]) if "." in sys.argv[4] else int(sys.argv[4])) if len(sys.argv) > 4 else None
    t0 = time.time()
    lm = MLXLM(model_dir, batch_size=16, use_chat_template=False)
    tm = TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks"))
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, task_manager=tm, num_fewshot=0,
                                  use_cache=os.environ.get("LMEVAL_CACHE"), limit=limit, log_samples=True,
                                  apply_chat_template=False, random_seed=123, numpy_random_seed=123,
                                  torch_random_seed=123, fewshot_random_seed=123)
    keep = ("acc", "acc_norm", "word_perplexity", "byte_perplexity", "bits_per_byte", "perplexity")
    samples = {t: [{"doc_id": s["doc_id"], **{k: s[k] for k in keep if k in s}} for s in lst]
               for t, lst in res["samples"].items()}
    with open(out, "w") as f:
        json.dump({"model": os.path.basename(os.path.normpath(model_dir)), "tasks": tasks, "limit": limit,
                   "seconds": round(time.time() - t0, 1), "results": res["results"], "groups": res.get("groups"),
                   "n-samples": res.get("n-samples"), "versions": res.get("versions"), "samples": samples},
                  f, default=str)
    for t, r in res["results"].items():
        print(t, {k: v for k, v in r.items() if isinstance(v, float) and "stderr" not in k})
    print(f"seconds {time.time() - t0:.0f}")


if __name__ == "__main__":     # datasets starts worker processes that re-import this file
    main()
