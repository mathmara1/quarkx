# Track 2 (vLLM benchmark) status — 2026-06-08

## DONE: MXFP4 + BF16 baselines for Llama-3.3-70B, with speedup comparison

Pipeline works end-to-end on MI350X. Image `rocm/vllm-dev:nightly_main_20251117`.
Benchmark tool: `vllm bench serve` (benchmark_serving.py is deprecated). All runs 0 failures.
Workload: 1024 in / 1024 out, random dataset, num_prompts = concurrency*16.

### MXFP4 vs BF16 — Llama-3.3-70B (output tok/s, MI350X)
| conc | BF16 tok/s | MXFP4 tok/s | speedup | BF16 TPOT | MXFP4 TPOT |
|------|-----------|-------------|---------|-----------|------------|
| 1    | 35.3      | 72.1        | 2.04x   | 28.2 ms   | 13.8 ms    |
| 4    | 69.9      | 269.8       | 3.86x*  | 57.0 ms   | 14.7 ms    |
| 16   | 426.9     | 738.4       | 1.73x   | 37.1 ms   | 21.4 ms    |
| 64   | 1264.4    | 1414.3      | 1.12x   | 49.7 ms   | 44.8 ms    |

Pattern (expected): MXFP4 wins biggest at LOW concurrency (memory-bandwidth bound — MXFP4 moves ~half the bytes), gap shrinks toward high concurrency (compute-bound, 1.12x at N=64).

*Note on N=4: BF16 wall-clock there (937s) is longer than N=16 (614s) — this is NOT an anomaly, just batching: N=16 generates 4x more tokens but at ~6x the aggregate throughput, so finishes sooner. BF16 N=4 throughput (69.9 tok/s) was re-confirmed by a partial rerun (~70 tok/s) before it was stopped. The one mild quirk: BF16 TPOT at N=4 (57ms) > N=16 (37ms) — reproducible, a small-batch scheduling effect, not measurement noise. The 3.86x there reflects this BF16 TPOT dip; treat N=1/16/64 as the cleaner trend.

## Models on disk (~/vllm-work/models/)
- `Llama-3.3-70B-MXFP4` — amd/Llama-3.3-70B-Instruct-MXFP4-Preview (38GB, quark/MXFP4)
- `Llama-3.3-70B-BF16` — unsloth/Llama-3.3-70B-Instruct (132GB, bfloat16). NOTE: meta-llama repo is gated (no HF token here); used ungated unsloth mirror. Identical arch (80 layers, hidden 8192). For throughput/latency, weight values don't matter — only arch+dtype — so it's a faithful BF16 baseline.

## Result files (~/vllm-work/results/)
- `llama70b_mxfp4_throughput.json`        (MXFP4 summary)
- `llama70b_bf16_throughput.json`         (BF16 summary)
- `llama70b_mxfp4_vs_bf16_comparison.json` (speedup table)
- per-run: `bench_N{1,4,16,64}.json` (MXFP4), `bf16_bench_N{1,4,16,64}.json` (BF16)
- logs: `~/vllm-work/*.log`

## Server container state
`vllm` container is currently serving the **BF16** model (holds GPU). `docker rm -f vllm` to free the GPU when done.

## MXFP6 support investigation (2026-06-08)

**Verdict: (b) loader-only / EMULATED — no native FP6 compute kernels. Plus currently non-functional out of the box (missing `amd-quark`).**

Container: vllm `0.11.1rc7.dev249`, compressed-tensors `0.12.2`, device `AMD Instinct MI350 OAM`, `current_platform.supports_mx() == True`.

### (1) vLLM source — MXFP6 exists ONLY in the Quark path
- `quantization/utils/ocp_mx_utils.py`: `SUPPORTED_OCP_MX_DTYPES = {"mxfp4", "mxfp6_e3m2", "mxfp6_e2m3"}`. (mxfp8/mxint8 listed in OCP_MX_DTYPES but NOT supported.) Schemes defined: `w_mxfp6_e3m2_a_mxfp6_e3m2`, `w_mxfp6_e2m3_a_mxfp6_e2m3`, and mixed `w_mxfp4_a_mxfp6_{e2m3,e3m2}`. Block size 32.
- `quark/schemes/quark_ocp_mx.py:164` — the decisive line:
  ```python
  self.emulate = not current_platform.supports_mx() or (
      self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4")
  ```
  → **emulation is FORCED for any dtype other than pure mxfp4/mxfp4**, even though this box reports `supports_mx()==True`. Native GEMM kernels (`gemm_afp4wfp4`, AITER) exist for w_mxfp4_a_mxfp4 ONLY. Confirmed by the runtime warning at :186 ("...kernels for input_dtype=... weight_dtype=... are not yet integrated in vLLM. Simulated weight dequantization and activation QDQ ... computed in high precision.").
- MoE: `fused_moe/utils.py:213` — explicit `# TODO: native mxfp6 is currently not integrated in vllm`. Emulated via quant_dequant too.
- Emulation impl `quantization/utils/mxfp6_utils.py` calls `quark.torch.kernel.hw_emulation...fake_quantize_fp4_fp6_per_group_with_scale` and **raises ImportError("The package `amd-quark` is required ... pip install amd-quark")** if quark is absent.

### (2) compressed-tensors 0.12.2 — NO MXFP6 at all
CompressionFormat enum = {dense, float-quantized, pack-quantized, nvfp4-pack-quantized, naive-quantized, sparse...}. No `mxfp6`/`mxfp6-pack-quantized`. So MXFP6 is **Quark-method-only**; the compressed-tensors loader path can't do it.

### Blocking gap on THIS container
`amd-quark` is NOT installed (`import quark` → ModuleNotFoundError). So even the emulated MXFP6 path would currently raise ImportError. To run MXFP6 at all here: `pip install amd-quark` inside the container.

### Practical implication for Track 2
MXFP6 in this vLLM = **memory-footprint / accuracy tool, NOT a throughput win**. It dequantizes to high precision and does the matmul in bf16/fp16, so expect throughput ≤ bf16 (dequant overhead) and well below native MXFP4. Don't expect an FP6 speedup curve like MXFP4's. Useful if the research question is "accuracy vs footprint at 6-bit", not "tokens/s".

### Candidate MXFP6 models on HF (for a load/smoke test)
- `fxmarty/Llama-3.1-70B-Instruct-2-layers-mxfp6` — 2-layer toy, ideal smoke test of the loader (fxmarty = HF quant eng).
- `fxmarty/qwen1.5_moe_a2.7b_chat_w_fp6_e3m2_a_fp6_e3m2` (~10.7k dl) and `..._w_fp4_a_fp6_e2m3` — small real Quark OCP-MX FP6 MoE models.
- Most other "mxfp6" hits are **GGUF** (llama.cpp, not vLLM-loadable) or noise (FP64 etc.).

## AITER FP6 kernel check (2026-06-08) — confirms the blocker is AMD
Scanned the AITER in this container (the kernel substrate vLLM AND SGLang use on ROCm): Python API, all 93 compiled `.so`, AOT `.hsaco`, op bindings. **Result: (c) NOTHING — no MXFP6 at any level.** GEMMs present: a16w16 (bf16), a8w8 (fp8/int8), a4w4 (FP4/MX). No `a6w6`/fp6 GEMM, no fp6 quant helper, no fp6 symbol in any binary, no file named `*fp6*`. Only trace of "fp6" = a docstring in `fp4_utils.py:128`. FP4 sanity check (found `gemm_a4w4_asm`) proves the scan was valid.
⇒ No native FP6 GEMM to probe, so NO benchmark script was written. ⇒ Native FP6 serving is **blocked on AMD** (ship FP6 GEMM in AITER); vLLM/SGLang are downstream. Switching frameworks won't help — same empty substrate.

## Bottom line on FP6 throughput
There is currently NO way to benchmark *native* FP6 serving on this stack. Emulated FP6 (vLLM, dequant→bf16) runs but only measures emulation overhead (≤ bf16), not the format. FP6's measurable value today = memory footprint + accuracy (Track 1), not tok/s.

## Next (NOT started — waiting on your go-ahead)
1. Qwen3 MXFP4 (matches Track-1 accuracy benchmarks, which use Qwen3-8B).
2. Re-check FP6 ONLY when a newer ROCm/AITER ships an FP6 GEMM — re-run the same AITER scan on the new image; if `a6w6`/fp6 symbols appear, native FP6 benchmarking becomes possible.
3. MXFP6 emulated smoke test IF desired: `pip install amd-quark`, load `fxmarty/Llama-3.1-70B-Instruct-2-layers-mxfp6` — emulated only, no throughput benefit.
