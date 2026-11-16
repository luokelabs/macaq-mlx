# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Public text for calibration and evaluation, from disjoint Wikipedia articles (zh + en), plus a homophone
candidate-ranking task built from the Chinese evaluation articles (IME-style: same toneless pinyin, pick the word).

Writes into OUT_DIR (default ./data):
  calib.txt                calibration text (zh + en paragraphs, shuffled; ~420k characters)
  eval.txt                 evaluation text (different articles; ~210k characters)
  homophone.jsonl          {"context", "candidates", "label", "pinyin"}, candidates share toneless pinyin
  homophone-clean.jsonl    the same without items whose context contains a newline (the set the results use)

Source: wikimedia/wikipedia 20231101.zh and 20231101.en (CC BY-SA 4.0), streamed, first 200 qualifying articles
skipped. The results in this repository used the files whose sha256 is listed in data/SHA256SUMS; check yours with
`shasum -a 256 -c data/SHA256SUMS` (from the repository root, after writing into ./data).

usage: python scripts/make_data.py [OUT_DIR]      (needs: pip install datasets jieba pypinyin)
"""
import json
import os
import random
import re
import sys
from itertools import islice

from datasets import load_dataset

D = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(D, exist_ok=True)
random.seed(1112)
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"   # the dataset revision the files were made from

CALIB_CHARS = {"zh": 120_000, "en": 300_000}
EVAL_CHARS = {"zh": 60_000, "en": 150_000}


def articles(lang):
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True,
                      revision=WIKI_REVISION)
    for row in ds:
        text = row["text"].strip()
        if len(text) < 1500:
            continue
        yield row["title"], text[:6000]


def take(gen, limit):
    out, n = [], 0
    for title, text in gen:
        out.append((title, text))
        n += len(text)
        if n >= limit:
            break
    return out


calib, evals, zh_eval = [], [], []
for lang in ("zh", "en"):
    gen = articles(lang)
    for _ in islice(gen, 200):   # skip a stretch so we do not take the very first articles
        pass
    c = take(gen, CALIB_CHARS[lang])
    e = take(gen, EVAL_CHARS[lang])
    calib += [t for _, t in c]
    evals += [t for _, t in e]
    if lang == "zh":
        zh_eval = [t for _, t in e]
    print(lang, "calib articles", len(c), "eval articles", len(e))

# shuffle paragraphs, not articles, so the first calibration chunks already mix zh and en and many topics
calib = [p for t in calib for p in t.split("\n") if len(p.strip()) > 40]
random.shuffle(calib)
random.shuffle(evals)
open(os.path.join(D, "calib.txt"), "w").write("\n\n".join(calib))
open(os.path.join(D, "eval.txt"), "w").write("\n\n".join(evals))
print("calib chars", sum(map(len, calib)), "eval chars", sum(map(len, evals)))

# ---- homophone task -------------------------------------------------------------------------------------------
import jieba  # noqa: E402
from pypinyin import lazy_pinyin  # noqa: E402

jieba.setLogLevel(60)
freq = {}
for raw in jieba.get_dict_file():
    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    parts = line.split()
    if len(parts) >= 2:
        w, f = parts[0], int(parts[1])
        if len(w) == 2 and re.fullmatch(r"[一-鿿]{2}", w):
            freq[w] = f
by_py = {}
for w, f in freq.items():
    by_py.setdefault(" ".join(lazy_pinyin(w)), []).append((f, w))
for k in by_py:
    by_py[k].sort(reverse=True)

items, seen = [], set()
sentences = []
for art in zh_eval:
    for s in re.split(r"(?<=[。！？；])", art):
        s = s.strip()
        if 14 <= len(s) <= 80 and re.search(r"[一-鿿]{8}", s):
            sentences.append(s)
random.shuffle(sentences)
for s in sentences:
    words = list(jieba.cut(s))
    pos = 0
    spots = []
    for w in words:
        if pos >= 8 and w in freq and freq[w] >= 300:
            group = [x for f, x in by_py.get(" ".join(lazy_pinyin(w)), []) if x != w and f >= 300]
            if len(group) >= 3:
                spots.append((pos, w, group[:3]))
        pos += len(w)
    if not spots:
        continue
    pos, w, others = random.choice(spots)
    ctx = s[:pos]
    if ctx in seen:
        continue
    seen.add(ctx)
    cands = [w] + others
    random.shuffle(cands)
    items.append({"context": ctx, "candidates": cands, "label": cands.index(w), "pinyin": " ".join(lazy_pinyin(w))})
    if len(items) >= 400:
        break
with open(os.path.join(D, "homophone.jsonl"), "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
# items with a newline in the context are dropped: they broke llama-perplexity's line-based hellaswag format, which
# the comparison with the upstream llama.cpp build used
keep = [it for it in items if "\n" not in it["context"] and all("\n" not in c for c in it["candidates"])]
with open(os.path.join(D, "homophone-clean.jsonl"), "w") as f:
    for it in keep:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("homophone items", len(items), "kept", len(keep))
