# Data

All calibration and evaluation text is public Wikipedia (`wikimedia/wikipedia`, dumps `20231101.zh` and
`20231101.en`, dataset revision `b04c8d1c`), licensed CC BY-SA 4.0.

| File | In git | What |
|---|---|---|
| `calib.txt` | no, made by `scripts/make_data.py` | ~420k characters of zh + en paragraphs, shuffled. The first 128 × 512 tokens give the MACaQ statistics; every 512-token chunk (261 for the Qwen3.5 tokenizer) is the GPTQ calibration set. |
| `eval.txt` | no, made by `scripts/make_data.py` | ~210k characters from other articles; never used for calibration. `scripts/eval_quality.py` scores its first 100 × 512 tokens. |
| `homophone.jsonl` | no, made by `scripts/make_data.py` | Candidate-ranking items: a sentence prefix from the Chinese evaluation articles and four common two-character words with the same toneless pinyin. |
| `homophone-clean.jsonl` | **yes** | The 233 items used for the results (items whose context contains a newline dropped). Contains short excerpts of Wikipedia: CC BY-SA 4.0, attribution to the Wikipedia contributors. |
| `SHA256SUMS` | yes | Hashes of the files the published results used. |

Regenerate and check (from the repository root):

```bash
pip install -e ".[data]"
python scripts/make_data.py data
shasum -a 256 -c data/SHA256SUMS
```

The streamed dataset order is fixed by the revision, and the shuffles use a fixed seed, so the text files should
come out identical. The homophone items also depend on jieba's dictionary and pypinyin (pinned in
`requirements.txt`); if `homophone.jsonl` differs, use the committed `homophone-clean.jsonl`.
