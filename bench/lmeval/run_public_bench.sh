#!/bin/bash
# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
#
# The 13 public benchmarks of docs/report.html (plus WikiText perplexity), zero-shot, no chat template, for each
# NAME=MODEL_DIR given; then the item-by-item comparison. Results go to RESULTS_DIR/<name>.json.
#
# usage: bench/lmeval/run_public_bench.sh RESULTS_DIR LIMIT NAME=MODEL_DIR [NAME=MODEL_DIR ...]
#   LIMIT: "" for every question (used for Qwen3.5-0.8B), 0.4 for the first 40% of each task / subject (Qwen3.5-9B)
# example:
#   bench/lmeval/run_public_bench.sh results/lmeval-0.8b "" \
#       bf16=models/Qwen3.5-0.8B-Base q4=models/Qwen3.5-0.8B-Base-4bit port=models/Qwen3.5-0.8B-Base-macaq4
# Needs bench/lmeval/tasks/cmmlu_local (make_cmmlu_tasks.py). Answers are cached per model under RESULTS_DIR/cache,
# so a stopped run resumes. On a small Mac wrap each run in scripts/watchdog.sh.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 3 ] || { sed -n '4,13p' "$0"; exit 2; }
R=$1; LIMIT=$2; shift 2
TASKS=arc_easy,arc_challenge,boolq,hellaswag,piqa,winogrande,lambada_openai,wikitext,mmlu,ceval-valid,cmmlu_local,xstorycloze_zh,xcopa_zh,xwinograd_zh
[ -f "$here/tasks/cmmlu_local/_cmmlu_local.yaml" ] || { echo "run make_cmmlu_tasks.py first" >&2; exit 1; }
mkdir -p "$R/cache"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
names=()
for nm in "$@"; do
  name=${nm%%=*}; dir=${nm#*=}; names+=("$name")
  if [ -f "$R/$name.json" ]; then echo "== $name already done"; continue; fi
  echo "== $name ($dir) $(date +%H:%M:%S)"
  LMEVAL_CACHE="$R/cache/$name" python "$here/lmeval_run.py" "$dir" "$R/$name.json" "$TASKS" $LIMIT
done
python "$here/compare_lmeval.py" "$R" "${names[@]}" --json "$R/summary.json"
