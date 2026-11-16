# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Compare lm-eval results of several variants task by task, with an exact McNemar test on the questions where two
variants differ.

usage: python bench/lmeval/compare_lmeval.py RESULTS_DIR NAME... [--json OUT]
  RESULTS_DIR/<name>.json (or <name>.json.gz) as written by lmeval_run.py.
Metric per task follows common practice: acc_norm for ARC-Challenge, HellaSwag, PIQA; acc elsewhere; word perplexity
for WikiText. Group tasks (MMLU, C-Eval, CMMLU) pool every subject's questions.
"""
import gzip
import json
import math
import os
import sys

TASKS = [  # (task, metric, label, language)
    ("mmlu", "acc", "MMLU (57 subjects)", "en"),
    ("arc_easy", "acc", "ARC-Easy", "en"),
    ("arc_challenge", "acc_norm", "ARC-Challenge", "en"),
    ("hellaswag", "acc_norm", "HellaSwag", "en"),
    ("piqa", "acc_norm", "PIQA", "en"),
    ("winogrande", "acc", "WinoGrande", "en"),
    ("boolq", "acc", "BoolQ", "en"),
    ("lambada_openai", "acc", "LAMBADA (OpenAI)", "en"),
    ("ceval-valid", "acc", "C-Eval valid (52 subj.)", "zh"),
    ("cmmlu_local", "acc", "CMMLU (67 subjects)", "zh"),
    ("xstorycloze_zh", "acc", "XStoryCloze zh", "zh"),
    ("xcopa_zh", "acc", "XCOPA zh", "zh"),
    ("xwinograd_zh", "acc", "XWinograd zh", "zh"),
]
GROUP_PREFIX = {"mmlu": "mmlu_", "ceval-valid": "ceval-valid_", "cmmlu_local": "cmmlu_local_"}


def p_exact(b, c):
    """Two-sided exact McNemar (binomial) p-value for b vs c discordant pairs."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def read(R, name):
    for path, opener in ((os.path.join(R, name + ".json"), open), (os.path.join(R, name + ".json.gz"), gzip.open)):
        if os.path.exists(path):
            with opener(path, "rt", encoding="utf-8") as f:
                return json.load(f)
    return None


def main():
    argv = sys.argv[1:]
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) < 2:
        raise SystemExit(__doc__)
    R, names = argv[0], argv[1:]
    runs = {n: r for n in names if (r := read(R, n)) is not None}
    names = [n for n in names if n in runs]

    def items(n, task, metric):
        s = runs[n]["samples"]
        keys = [k for k in s if k.startswith(GROUP_PREFIX[task])] if task in GROUP_PREFIX else [task]
        return {(k, it["doc_id"]): bool(it[metric]) for k in keys for it in s[k]}

    def score(n, task, metric):
        if task in GROUP_PREFIX and runs[n].get("groups"):
            r = runs[n]["groups"].get(task)
        else:
            r = runs[n]["results"].get(task)
        return r[metric + ",none"]

    summary = {"tasks": {}, "pairs": {}, "seconds": {n: runs[n]["seconds"] for n in names}}
    print("task".ljust(26) + "".join(n.rjust(9) for n in names) + "       n")
    for task, metric, label, lang in TASKS:
        its = {n: items(n, task, metric) for n in names}
        n_items = len(next(iter(its.values())))
        assert all(len(v) == n_items and set(v) == set(its[names[0]]) for v in its.values()), task
        sc = {n: score(n, task, metric) for n in names}
        for n in names:  # the pooled item mean equals the group's size-weighted score
            assert abs(sum(its[n].values()) / n_items - sc[n]) < 1e-9, (task, n)
        summary["tasks"][task] = {"label": label, "metric": metric, "lang": lang, "n": n_items,
                                  **{n: {"score": sc[n], "right": sum(its[n].values())} for n in names}}
        print(f"{label:26s}" + "".join(f"{100 * sc[n]:9.2f}" for n in names) + f"{n_items:8d}")
    w = {n: runs[n]["results"]["wikitext"]["word_perplexity,none"] for n in names}
    lp = {n: runs[n]["results"]["lambada_openai"]["perplexity,none"] for n in names}
    summary["wikitext_word_ppl"], summary["lambada_ppl"] = w, lp
    print(f"{'WikiText word ppl (low)':26s}" + "".join(f"{w[n]:9.3f}" for n in names))
    print(f"{'LAMBADA ppl (low)':26s}" + "".join(f"{lp[n]:9.3f}" for n in names))

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    print("\n== per-question differences (right only in A / only in B, exact McNemar p)")
    for a, b in pairs:
        tot_x = tot_y = 0
        lines = []
        for task, metric, label, lang in TASKS:
            ia, ib = items(a, task, metric), items(b, task, metric)
            x = sum(ia[k] and not ib[k] for k in ia)
            y = sum(ib[k] and not ia[k] for k in ia)
            tot_x, tot_y = tot_x + x, tot_y + y
            p = p_exact(x, y)
            summary["pairs"].setdefault(f"{a}|{b}", {})[task] = {"only_a": x, "only_b": y, "p": p}
            lines.append(f"  {label:26s} {x:5d} / {y:<5d} p={p:.3g}" + ("  *" if p < 0.05 else ""))
        p = p_exact(tot_x, tot_y)
        summary["pairs"][f"{a}|{b}"]["all"] = {"only_a": tot_x, "only_b": tot_y, "p": p}
        print(f"{a} vs {b}:")
        print("\n".join(lines))
        print(f"  {'all 13 tasks pooled':26s} {tot_x:5d} / {tot_y:<5d} p={p:.3g}")
    print("\nseconds:", summary["seconds"])
    if out_json:
        with open(out_json, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
