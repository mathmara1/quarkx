# Adaptive per-block grid selection (research prototype)

For each block of 32 weight values, this scheme quantize-dequantizes under
several candidate OCP MX element formats (e.g. `fp6_e2m3` and `fp6_e3m2`)
and keeps whichever format minimizes per-block MSE. Concept is from the
IF4 paper ([arXiv:2603.28765](https://arxiv.org/abs/2603.28765)) and the
Grid Games paper ([arXiv:2605.12327](https://arxiv.org/abs/2605.12327)),
generalized to 6-bit formats.

Built on Quark's existing `fake_quantize_fp4_fp6_per_group_with_scale`
primitive. Pure weight-only PTQ via in-place `nn.Linear.weight` rewrite;
activations stay BF16. The eval pipeline is Quark's standard
`quark.contrib.llm_eval.eval_model` (same one `quantize_quark.py` uses),
so results format-match Track 1 outputs by construction.

## Files

- `adaptive_grid.py` — core: `adaptive_quantize_dequantize` (single tensor)
  and `quantize_model_weights_inplace` (whole model walker).
- `run_adaptive.py` — runner: load HF model → apply adaptive quant → hand
  off to Quark's `eval_model` for lm-evaluation-harness tasks and/or PPL.

## Usage

The flags mirror `quantize_quark.py`'s eval subset, so any task list you've
used for Track 1 baselines works here verbatim.

**BF16 baseline (sanity check — should match Track 1's BF16 run):**

```bash
python run_adaptive.py \
    --model_dir Qwen/Qwen3-0.6B \
    --skip_quantization \
    --tasks piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval \
    --eval_batch_size 16
```

**Fixed-format baseline — single format in `--formats` makes the "selection"
trivial and produces the standard per-format reconstruction:**

```bash
python run_adaptive.py \
    --model_dir Qwen/Qwen3-0.6B \
    --formats fp6_e2m3 \
    --tasks piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval \
    --eval_batch_size 16
```

**Adaptive `fp6_e2m3 ⊕ fp6_e3m2` — the headline run:**

```bash
python run_adaptive.py \
    --model_dir Qwen/Qwen3-0.6B \
    --formats fp6_e2m3 fp6_e3m2 \
    --tasks piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval \
    --eval_batch_size 16
```

## Validation plan

The runner uses Quark's standard eval, so the natural comparison is against
Track 1's Quark baselines on the same model + same tasks:

| Run | Where it comes from |
|---|---|
| BF16 baseline | Track 1 `01_bf16_baseline.log` (done) |
| Fixed MXFP6_e2m3 | Track 1 `02_mxfp6_e2m3_rtn.log` (in progress) |
| Fixed MXFP6_e3m2 | Track 1 `03_mxfp6_e3m2_rtn.log` (queued) |
| Adaptive(e2m3, e3m2) | This script |

All four rows use the same `eval_model` call, so per-task numbers are
directly comparable.

**Acceptance criteria** (what "correct" looks like):
1. With `--skip_quantization`, this script reproduces Track 1's BF16
   numbers (within lm-eval run-to-run noise, typically ~0.001 on accuracy).
2. With `--formats fp6_e2m3`, this script reproduces Track 1's
   `--quant_scheme mxfp6_e2m3` numbers (same arithmetic, same eval).
3. With `--formats fp6_e2m3 fp6_e3m2`, the adaptive run gives accuracy
   *between* the two fixed runs and never strictly worse than the better one
   on most tasks. Per-block MSE selection is provably optimal at the block
   level, but per-block MSE composes non-linearly through the network — the
   network-level metric is empirical, not guaranteed.

If criterion 1 fails: there's a load/eval bug in this script.
If criterion 2 fails: Quark's `fake_quantize_fp4_fp6_per_group_with_scale`
behaves differently when called from outside vs. from inside the standard
quantization pipeline. Worth investigating.
If criterion 3 fails: the per-block selection logic in `adaptive_grid.py`
has a bug.

## Cross-validation against the adaptive-mxfp6 library

Optional, but useful when the goal is "Quark backend matches microxcaling
backend on the same algorithm":

```powershell
# On laptop CPU, run the existing library through the SAME lm-eval tasks
# (requires adding lm-eval support to that library, which it doesn't have
# currently — its eval is custom-PPL only).
```

That cross-validation isn't built yet — the user's library does sliding-
window WikiText-2 PPL, not lm-eval tasks. If we ever need direct comparison
across libraries, the easier path is: have the user's library save HF-format
quantized model dirs, then point `quantize_quark.py --skip_quantization` at
each to eval through Quark's pipeline.

## Open items

- **Scheme registration**: this still bypasses `--quant_scheme`. Once
  validation lands, the natural next step is registering as
  `mxfp_adaptive` and wiring through `quantize_quark.py`.
- **Wider format families**: `[fp6_e2m3, fp6_e3m2, int6, nf6]` needs INT6/NF6
  in Quark's per-group path (not currently supported there).
- **vLLM deployment**: per-block format selection needs runtime dispatch
  (Track 4: MI355X kernel work).
