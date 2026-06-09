"""
Track 2 throughput chart in the style of AMD Blog 4 Figure 6.

Two side-by-side panels:
  Left: output tokens/sec vs concurrency (BF16 line vs MXFP4 line)
  Right: mean TPOT (ms) vs concurrency (BF16 line vs MXFP4 line)

Reads llama70b_mxfp4_vs_bf16_comparison.json sitting next to this script.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "results" / "llama70b_mxfp4_vs_bf16_comparison.json").read_text())

rows = sorted(DATA["rows"], key=lambda r: r["max_concurrency"])
conc = [r["max_concurrency"] for r in rows]
bf16_tps = [r["bf16_output_tok_s"] for r in rows]
mxfp4_tps = [r["mxfp4_output_tok_s"] for r in rows]
bf16_tpot = [r["bf16_mean_tpot_ms"] for r in rows]
mxfp4_tpot = [r["mxfp4_mean_tpot_ms"] for r in rows]
speedups = [r["throughput_speedup_x"] for r in rows]

fig, (ax_tps, ax_tpot) = plt.subplots(1, 2, figsize=(13, 5))

# --- Throughput panel ---
ax_tps.plot(conc, bf16_tps, marker="o", linewidth=2, color="#3b82f6", label="BF16")
ax_tps.plot(conc, mxfp4_tps, marker="s", linewidth=2, color="#10b981", label="MXFP4")
for x, y in zip(conc, bf16_tps):
    ax_tps.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -16),
                    fontsize=8, ha="center", color="#1e3a8a")
for x, y, s in zip(conc, mxfp4_tps, speedups):
    ax_tps.annotate(f"{y:.0f}  ({s:.2f}x)", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center", color="#065f46")
ax_tps.set_xscale("log", base=2)
ax_tps.set_xticks(conc)
ax_tps.set_xticklabels(conc)
ax_tps.set_xlabel("max concurrency")
ax_tps.set_ylabel("output throughput (tokens/sec)")
ax_tps.set_title("Total output throughput")
ax_tps.grid(True, linestyle="--", alpha=0.4)
ax_tps.legend()
ax_tps.spines["top"].set_visible(False)
ax_tps.spines["right"].set_visible(False)

# --- TPOT panel ---
ax_tpot.plot(conc, bf16_tpot, marker="o", linewidth=2, color="#3b82f6", label="BF16")
ax_tpot.plot(conc, mxfp4_tpot, marker="s", linewidth=2, color="#10b981", label="MXFP4")
for x, y in zip(conc, bf16_tpot):
    ax_tpot.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     fontsize=8, ha="center", color="#1e3a8a")
for x, y in zip(conc, mxfp4_tpot):
    ax_tpot.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -16),
                     fontsize=8, ha="center", color="#065f46")
ax_tpot.set_xscale("log", base=2)
ax_tpot.set_xticks(conc)
ax_tpot.set_xticklabels(conc)
ax_tpot.set_xlabel("max concurrency")
ax_tpot.set_ylabel("mean TPOT (ms)  -- lower is better")
ax_tpot.set_title("Time per output token (decode latency)")
ax_tpot.grid(True, linestyle="--", alpha=0.4)
ax_tpot.legend()
ax_tpot.spines["top"].set_visible(False)
ax_tpot.spines["right"].set_visible(False)

fig.suptitle(
    "Llama-3.3-70B on AMD Instinct MI350X (1024 in / 1024 out, random workload)\n"
    "vLLM, rocm/vllm-dev:nightly_main_20251117",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_png = HERE / "throughput_chart.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")

print("\nThroughput speedup summary:")
print(f"{'conc':>6s} {'BF16 tok/s':>12s} {'MXFP4 tok/s':>12s} {'speedup':>10s} "
      f"{'BF16 TPOT':>10s} {'MXFP4 TPOT':>11s}")
for r in rows:
    print(f"{r['max_concurrency']:>6d} "
          f"{r['bf16_output_tok_s']:>12.1f} "
          f"{r['mxfp4_output_tok_s']:>12.1f} "
          f"{r['throughput_speedup_x']:>9.2f}x "
          f"{r['bf16_mean_tpot_ms']:>9.1f}ms "
          f"{r['mxfp4_mean_tpot_ms']:>10.1f}ms")
