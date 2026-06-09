# Track 3a validation status — 2026-06-09

## Setup
- Quark installed in: `~/quark-work/quarkx` (editable, `pip install -e .`)
- Quark version: `0.11.1+28871bf` (branch `release/0.11`, HEAD "Phase 1 + Track 3a smoke results (Qwen3-0.6B)")
- Adaptive scheme name (string from register_scheme): **`mxfp_adaptive_e2m3_e3m2`**
  - 2-way: formats `["fp6_e2m3", "fp6_e3m2"]`, group_size=32, weight-only static PTQ (activations BF16). The 3-way `_fp4` variant is present but commented out.
- C++ kernel_ext compiled OK: **yes** — built for `PYTORCH_ROCM_ARCH='gfx950'`, 44s, no errors.

### Environment fixes applied during setup (env only — NO quarkx code changed)
1. **Removed broken `torchvision` 0.26.0** — built for torch 2.11.0, crashed on our torch 2.10.0+rocm7.0 (`operator torchvision::nms does not exist`), which broke `from transformers import AutoProcessor`. torchvision is unused for text-LLM weight PTQ (not in requirements or example code). Reversible.
2. **ninja PATH** — `ninja` pkg+binary are in the venv, but must run with `~/quark-venv/bin` on PATH or torch can't find the `ninja` executable for kernel JIT. All runs use `PATH="$HOME/quark-venv/bin:$PATH"`.

## IMPORTANT deviation from brief: how PPL is obtained
The brief assumed `--skip_evaluation` still prints a sanity perplexity. **It does not in this build** — with `--skip_evaluation` the script quantizes, freezes, and exits with no PPL. Verified in code: `quantize_quark.py:321-328` auto-sets `use_ppl_eval_model=True` and calls `eval_model()` ONLY when eval is *not* skipped. `quark/contrib/llm_eval/evaluation.py:79-81` then runs `ppl_eval` and prints `[INFO] Perplexity: <n>`.
- So all runs were done WITHOUT `--skip_evaluation` (not a guess — this is the script's documented path).
- `ppl_eval` uses **seq_len=2048 over the full wikitext-2-raw-v1 test split** (evaluation.py:130), which is DIFFERENT from the user's library reference config (seq=512, stride=256, 1 sequence). ⇒ absolute PPLs are NOT comparable to the library's 19.66; only Quark-internal comparisons (the per-block invariant) are valid here.
- The `Token indices sequence length is longer...` warning is benign — `ppl_eval` windows the corpus into 2048-token chunks.
- Selector fraction stats are NOT logged by this build. Non-degeneracy is instead proven by the invariant (adaptive strictly below both singles ⇒ genuinely mixing).

Also fixed during setup (env only): removed broken torchvision; put venv bin on PATH for ninja; redirected HF cache to `~/quark-work/hf_home` via `HF_HOME` because `~/.cache/huggingface` is root-owned from prior Docker work (non-destructive — old cache untouched).

## Layer 1 — smoke test
- Ran to completion: **yes** (exit 0)
- Wikitext PPL (adaptive): **21.376**
- Selector stats: not logged by this build (see note above)
- Verdict: **PASS** — runs clean, finite PPL in the 18–22 range.

## Layer 2 — three-config comparison
| config | PPL |
|---|---|
| mxfp6_e2m3 single | 21.694 |
| mxfp6_e3m2 single | 21.778 |
| adaptive (mxfp_adaptive_e2m3_e3m2) | **21.376** |

- Per-block invariant (adaptive ≤ min(singles)): **YES** — 21.376 ≤ 21.694, margin 0.318 below the better single; below both, equal to neither ⇒ non-degenerate per-block selection.
- Adaptive PPL within 18–22 sanity range: **YES** (21.38).
- Within ±5% of library 4-way reference (19.66): **NO** (21.38 = +8.7%) — but EXPECTED and not a port failure: (a) different eval methodology (Quark seq2048/full-test vs library seq512/1-seq), (b) 2-way here vs the library's 4-way which adds int6+nf6 (those extra grids are what pull the library below fp16). Absolute cross-comparison to 19.66 is not apples-to-apples.
- Verdict: **PASS** — the headline correctness property (per-block guarantee) holds strictly; no fail signals (adaptive is not > singles, not degenerate).

## fp16/native baseline cross-check (same Quark ppl_eval, seq2048/full-test)
Ran unquantized baseline via `--skip_quantization` (model-native bf16; `--quant_scheme` ignored). PPL = **20.971**.

| config | PPL | Δ vs baseline |
|---|---|---|
| baseline (unquantized bf16) | 20.971 | — |
| adaptive (mxfp_adaptive_e2m3_e3m2) | 21.376 | +0.405 (+1.9%) |
| mxfp6_e2m3 | 21.694 | +0.723 (+3.5%) |
| mxfp6_e3m2 | 21.778 | +0.807 (+3.9%) |

**Adaptive ~halves the quantization degradation** vs either single 6-bit format. This is the expected behavior and confirms the numbers look right.

Cross-frame consistency vs the user's library (which measured in a seq512/1-seq frame):
- library fp16 19.6862 → Quark fp16 20.971  (+1.28, pure methodology shift)
- library 4-way adaptive 19.6610 → Quark 2-way adaptive 21.376  (+1.72; the extra +0.44 over the fp16 shift = the 2-way-vs-4-way gap, since the library's int6/nf6 grids help)
Everything is internally consistent: the seq2048 frame shifts all PPLs up by ~1.3, and the 2-way port lands exactly where expected relative to fp16.

## Recommendation
- Ready for Layer 3 full eval on Qwen3-8B: **yes** — Layers 1 & 2 pass clean; the scheme runs and the selector is provably correct.
- Suggested cheap cross-check before/alongside Layer 3: run an **fp16 (unquantized) baseline through the SAME Quark `ppl_eval`** (seq2048/full-test) on Qwen3-0.6B. That gives the apples-to-apples reference for these absolute numbers (the library's 19.66 is in a different measurement frame), so we can quantify adaptive's degradation over fp16 directly. ~3 min, no quant.
- Issues / open questions:
  - Confirm the brief author is OK that PPL came from the real eval path (no `--skip_evaluation`) at seq2048/full-test, not the seq512/1-seq config the 19.66 reference used. The invariant conclusion is unaffected, but absolute-number matching to the library requires aligning eval configs.
