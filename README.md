# macaq-mlx

A port of the weight transform of [MACaQ](https://github.com/huawei-csl/MACaQ) to [MLX](https://github.com/ml-explore/mlx),
for Qwen3.5 models on Apple silicon. Each linear layer gets MACaQ's per-channel pre-scale and 64-wide block Hadamard
transform; the transformed weights are then quantized to standard MLX affine 4-bit with GPTQ. The result loads with
[mlx-lm](https://github.com/ml-explore/mlx-lm) through one extra model file.

On Qwen3.5-0.8B-Base and Qwen3.5-9B, the converted 4-bit models are about twice as close to the bf16 original as
MLX's default 4-bit at the same size (mean KL 0.074 vs 0.158 and 0.040 vs 0.094). The cost is 10–15% slower
generation. Full method and numbers: [docs/report.html](docs/report.html).

This is research code, tested on two models with one version of MLX. It is not affiliated with the MACaQ authors.

## Relation to MACaQ

MACaQ (Apache-2.0; paper [arXiv:2611.10649](https://arxiv.org/abs/2611.10649)) quantizes weights in two steps, and
ships backends for its own llama.cpp fork, vLLM, Diffusers and PyTorch, but not MLX:

1. **Transform.** For every linear layer, compute a per-input-channel scale
   `p = clamp(std(W)^0.75 / rms(x)^0.25, 1e-4, 1e4)` (rounded to bf16), store `M_W = (W / p) · B` where `B` is a
   block-diagonal normalized Sylvester Hadamard matrix (blocks of 64), and at run time feed the layer
   `M_A = (x · p) · B`. The product is unchanged (`M_W M_Aᵀ = W xᵀ`), but outlier weights are spread over their block,
   so a 4-bit grid fits them better.
2. **Codebook quantization.** Quantize `M_W` to llama.cpp's IQ4_XS codebook with a vector-codebook refinement.

This repository ports step 1 exactly (same formulas, exponent, block width and Hadamard ordering as MACaQ commit
`e9d4f89`; `mx.hadamard_transform` is checked against the Sylvester matrix on every run). Step 2 is replaced by GPTQ,
run in the transformed basis: each layer's Hessian is taken on `M_A`, the input the quantized weight actually sees.
MLX has no IQ4_XS format, and adding one would need new Metal kernels. The output is an ordinary MLX affine 4-bit
model (group 64) plus one `p` vector per transformed layer (about 1 MB for the 0.8B).

## Results

Measured against the bf16 original on 100 × 512 tokens of Wikipedia text (zh + en, the second half of each chunk
scored, as llama-perplexity does). "Default 4-bit" is what `mlx_lm.convert -q` produces (4-bit, group 64); for the
9B it is `mlx-community/Qwen3.5-9B-4bit`.

| | Qwen3.5-0.8B-Base, default 4-bit | 0.8B, this port | Qwen3.5-9B, default 4-bit | 9B, this port |
|---|---|---|---|---|
| File size | 424 MB | 425 MB | 5.95 GB (incl. ~0.9 GB vision tower) | 5.04 GB (text only) |
| Mean KL vs bf16 (lower is better) | 0.158 | **0.074** | 0.094 | **0.040** |
| Perplexity (bf16: 16.95 / 8.35) | 19.63 (+15.8%) | **18.18 (+7.2%)** | 9.14 (+9.5%) | **8.67 (+3.8%)** |
| Same top-1 token as bf16 | 78.0% | **83.8%** | 86.1% | **91.2%** |
| Generation, tokens/s | 146.7 | 124.6 (float32 transform) / 131.3 (bf16) | 114.9 | 96.7 (float32) / 104.0 (bf16) |

Speed: 512-token prompt, 128 generated tokens, median of 5; the 0.8B on a Mac mini M2 (16 GB), the 9B on a
Mac Studio M3 Ultra. Converting the 9B took 32 minutes on the M3 Ultra with a 33 GB memory peak.

Public benchmarks (lm-evaluation-harness 0.4.13, zero-shot, 13 tasks; questions compared one by one with an exact
McNemar test):

- **0.8B** (all 54,603 questions): the port is significantly better than default 4-bit on 6 of 13 tasks (MMLU
  46.7 → 49.0, ARC-Easy, ARC-Challenge, HellaSwag, BoolQ, XStoryCloze-zh) and significantly worse on none. On the
  English tasks it loses 1.1 points against bf16 on average, default 4-bit 2.9. Neither 4-bit model recovers the
  5–8.5 points lost on C-Eval and CMMLU. (The BoolQ gain mostly reflects fewer "yes" answers; see the report.)
- **9B** (first 40% of each task, 21,912 questions): the port is significantly better on LAMBADA and C-Eval and
  worse on none; pooled over all tasks it answers 865 questions right that default 4-bit gets wrong, against 730 the
  other way (p < 0.001). The ~2-point MMLU loss is not recovered.

Results files: [results/](results/) (quality and speed JSON, and every lm-eval question's result, gzipped). The port
was also run through internal, task-specific evaluations on private test sets; those are not published here.

### For context: other options on the 0.8B

Same evaluation, all on the Mac mini M2. The upstream row is MACaQ's own pipeline in its llama.cpp fork, compared
with an f16 GGUF instead of MLX bf16.

| Variant | Size | KL | Perplexity |
|---|---|---|---|
| MLX default 4-bit (group 64) | 424 MB | 0.158 | +15.8% |
| MLX 4-bit, group 32 | 471 MB | 0.118 | +12.1% |
| MLX 4-bit, embedding 6-bit | 488 MB | 0.132 | +12.9% |
| mlx-lm GPTQ 4-bit, embedding 6-bit | 488 MB | 0.060 | +5.6% |
| MLX 5-bit (group 64) | 518 MB | **0.036** | +3.4% |
| This port, transform + round-to-nearest 4-bit | 425 MB | 0.108 | +11.1% |
| **This port, transform + GPTQ 4-bit** | **425 MB** | **0.074** | +7.2% |
| This port, transform + GPTQ, embedding 6-bit | 489 MB | 0.047 | +4.3% |
| Upstream MACaQ IQ4_XS (llama.cpp fork, ~90 tok/s on the M2) | 498 MB | 0.034 | +3.5% |

If file size is not the constraint, plain MLX 5-bit is closer to the original and needs no extra code. The port is
the most accurate option we found at the size of default 4-bit.

## Install

Python 3.10+ on Apple silicon. The results were produced with Python 3.12, mlx 0.32.4 and mlx-lm 0.32.2.

```bash
git clone https://github.com/luokelabs/macaq-mlx.git && cd macaq-mlx
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the results
pip install -e .                  # the macaq_mlx package and the macaq-* commands
```

## Convert a model

Weights are not in this repository. Download the originals (pinned to the revisions used for the results):

```bash
hf download Qwen/Qwen3.5-0.8B-Base --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --local-dir models/Qwen3.5-0.8B-Base
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
```

Make the calibration text (public Wikipedia; see [data/README.md](data/README.md)):

```bash
python scripts/make_data.py data
shasum -a 256 -c data/SHA256SUMS
```

Convert in one process (transform, then GPTQ a group of layers at a time, with a checkpoint after each group):

```bash
macaq-port --hf models/Qwen3.5-0.8B-Base --out models/Qwen3.5-0.8B-Base-macaq4 --calib data/calib.txt
scripts/watchdog.sh port-9b.log macaq-port --hf models/Qwen3.5-9B --out models/Qwen3.5-9B-macaq4 --calib data/calib.txt
```

- Statistics: the first 128 × 512 tokens of `--calib`. GPTQ: every 512-token chunk (261 chunks here), 4-bit,
  group 64, damping 0.01. The embedding gets plain 4-bit (`--emb-bits 6` for 6-bit); an untied `lm_head` (the 9B)
  gets plain GPTQ, since MACaQ only transforms the decoder layers.
- Memory: the 9B needs a Mac with about 48 GB or more (33 GB peak with `--group-layers 8`). `scripts/watchdog.sh`
  stops the run when macOS reports memory pressure; run the same command again and it resumes from the last
  finished group (`OUT.ckpt/`, deleted after a successful save).
- The 0.8B results in the report were made in two steps, with the transformed bf16 model written to disk in between.
  On the 0.8B the two paths give bit-identical weights:
  ```bash
  macaq-convert --hf models/Qwen3.5-0.8B-Base --out models/0.8b-transformed --calib data/calib.txt --no-quant
  macaq-gptq --src models/0.8b-transformed --out models/Qwen3.5-0.8B-Base-macaq4 --calib data/calib.txt --emb-bits 4
  # writes models/Qwen3.5-0.8B-Base-macaq4-e4
  ```
- `macaq-convert` without `--no-quant` gives the transform with round-to-nearest 4-bit (KL 0.108 on the 0.8B).

Each output folder contains `qwen3_5_macaq.py`, the model file mlx-lm needs, and a `config.json` with
`model_type: qwen3_5_macaq` and a `macaq` section (transformed layers, block width, calibration, `pre_dtype`).

## Use a converted model

From Python, import the package first; it registers the model type with mlx-lm:

```python
import macaq_mlx  # noqa: F401  (registers model_type "qwen3_5_macaq")
from mlx_lm import load, generate

model, tokenizer = load("models/Qwen3.5-9B-macaq4")
print(generate(model, tokenizer, "The capital of France is", max_tokens=20))
```

For mlx-lm's command-line tools, `mlx_lm.server`, or engines built on mlx-lm, copy the model file into the mlx-lm
installation once:

```bash
cp models/Qwen3.5-9B-macaq4/qwen3_5_macaq.py "$(python -c 'import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))')/models/"
mlx_lm.generate --model models/Qwen3.5-9B-macaq4 --prompt "The capital of France is"
```

**Transform precision (`pre_dtype`).** The run-time transform (scale, then Hadamard) runs in the precision set by
`macaq.pre_dtype` in `config.json`: `bfloat16` (what the converters write), `float32` (the default when the key is
missing), or `float16` (untested). On the 9B, bf16 and float32 differ by KL 0.0008; bf16 generates about 7.5%
faster and needs less memory for long prompts. Edit the key to switch; the weights do not change.

**Serving engines.** The quantized MACaQ layer is deliberately not a subclass of `nn.QuantizedLinear`. oMLX 0.7.2
replaces every `QuantizedLinear` with its own kernels for prompts over 2048 tokens; with a subclass, that skipped the
input transform and produced garbage. With the current loader, oMLX 0.7.2 and mlx-lm gave identical outputs on
short and 4,080-token prompts. Any engine that swaps layers by type could have the same problem: compare its output
with mlx-lm's (`scripts/compare_logits.py`). The converters also set `architectures` to
`Qwen3_5MacaqForCausalLM`, so engines that choose a vision path by architecture treat the model as text-only.

## Evaluate

```bash
# fidelity vs the bf16 original: KL, perplexity, top-1 agreement, homophone ranking (233 items)
python scripts/eval_quality.py models/Qwen3.5-0.8B-Base models/Qwen3.5-0.8B-Base-macaq4 eval-port.json \
    --homophone data/homophone-clean.jsonl
# speed (512-token prompt + 128 tokens) and the memory of one long prefill
python scripts/bench_speed.py models/Qwen3.5-0.8B-Base-macaq4 speed-port.json
python scripts/prefill_peak.py models/Qwen3.5-0.8B-Base-macaq4 4096 prefill-port.json
# baseline: MLX default 4-bit
mlx_lm.convert --hf-path models/Qwen3.5-0.8B-Base --mlx-path models/Qwen3.5-0.8B-Base-4bit -q
```

Public benchmarks (needs `pip install -e ".[bench]"`):

```bash
# CMMLU from its official CSV release (the hub copy needs a loading script): download cmmlu_v1_0_1.zip from the
# Hugging Face dataset lmlmcat/cmmlu, unzip it, then
python bench/lmeval/make_cmmlu_tasks.py /abs/path/to/cmmlu
# all questions (0.8B); use 0.4 instead of "" for the first 40% of each task (9B)
bench/lmeval/run_public_bench.sh results/my-0.8b "" bf16=models/Qwen3.5-0.8B-Base \
    q4=models/Qwen3.5-0.8B-Base-4bit port=models/Qwen3.5-0.8B-Base-macaq4
# compare existing results, e.g. the published ones
python bench/lmeval/compare_lmeval.py results/lmeval-9b bf16 q4 port
```

## Repository layout

| Path | What |
|---|---|
| `macaq_mlx/qwen3_5_macaq.py` | The mlx-lm model type `qwen3_5_macaq` (loader). Shared by the 0.8B and the 9B. |
| `macaq_mlx/port_convert.py` | `macaq-convert`: statistics, MACaQ transform, optional round-to-nearest quantization. |
| `macaq_mlx/port_gptq.py` | `macaq-gptq`: GPTQ on a transformed model, Hessians in the transformed basis. |
| `macaq_mlx/port_layerwise.py` | `macaq-port`: both steps in one process, layer groups, checkpoints. |
| `scripts/` | Data preparation, fidelity, speed, prefill memory, logit comparison, memory watchdog. |
| `bench/lmeval/` | lm-eval runner, CMMLU task generator, item-by-item comparison with McNemar tests. |
| `results/` | The numbers behind the report. |
| `docs/report.html` | Full write-up: method, fidelity, public benchmarks, costs, limitations. Regenerated from `results/` by `python docs/build_report.py`. |
| `tests/` | Unit checks of the transform on random data (`python -m pytest tests`). |

## Known limitations

- Dense Qwen3.5 only (tested: Qwen3.5-0.8B-Base, Qwen3.5-9B). Other architectures need the list of layers in
  `port_convert.py` and the loader adapted.
- MACaQ's codebook step is not ported. Upstream, on llama.cpp, reaches KL 0.034 on the 0.8B (498 MB) against 0.074
  here (425 MB).
- Generation is 10–15% slower than default 4-bit: two extra GPU operations per transformed layer. A fused Metal kernel
  (transform + quantized matmul) would remove most of this; none is written.
- The 9B port is text-only; the vision tower of Qwen3.5-9B is dropped.
- Calibration text is Wikipedia (zh + en). Other domains and languages were not tried.
- Each benchmark ran once; the 9B public benchmarks used the first 40% of each task. Speed numbers come from shared
  machines and are indicative.
- Tested with mlx 0.32.4 and mlx-lm 0.32.2 only.

## License

The code is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). It is derived from MACaQ (Apache-2.0) and
adapts parts of mlx-lm (MIT, [licenses/mlx-lm-MIT.txt](licenses/mlx-lm-MIT.txt)); [NOTICE](NOTICE) lists what came
from where and what changed. `data/homophone-clean.jsonl` contains Wikipedia excerpts under CC BY-SA 4.0. Models you
convert are derivatives of the Qwen3.5 weights and keep their license (Apache-2.0).

## Acknowledgements

- MACaQ by the MACAQ authors: <https://github.com/huawei-csl/MACaQ>, arXiv:2611.10649.
- GPTQ: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
  arXiv:2210.17323; the implementation follows mlx-lm's `mlx_lm/quant/gptq.py`.
- MLX and mlx-lm by Apple; Qwen3.5 by the Qwen team.
