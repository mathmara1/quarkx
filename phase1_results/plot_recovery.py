"""
Recovery bar chart in the style of AMD Blog 4 Figure 9.

Plots each quantized run's accuracy as a percentage of the BF16 baseline,
laid out in a 2-row bar chart over the 9 evaluation tasks. Right now we
only have one quantized run (MXFP6_e2m3 RTN) so each task shows one bar;
when more runs are added (e3m2, adaptive grid, scaling-formula variants,
etc.) the script auto-extends to a grouped bar chart per task.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BASELINE_LABEL = "01_bf16_baseline"

# Top row, bottom row -- AMD Blog 4 Fig 9 layout (omit ifeval to match the look;
# ifeval has too many sub-metrics for a clean bar). "mean" is the unweighted
# average of all per-task recoveries on each row of the figure.
TOP_TASKS = ["mean", "arc_challenge", "arc_easy", "hellaswag", "leaderboard_mmlu_pro"]
BOTTOM_TASKS = ["piqa", "winogrande", "gsm8k_platinum", "lambada_standard", "ifeval"]

# Which metric to pull for each task -- prefer acc_norm > acc > exact_match.
METRIC_PRIORITY = ["acc_norm", "acc", "exact_match"]
# Special-case ifeval: use inst_level_strict_acc (most cited).
TASK_METRIC_OVERRIDE = {"ifeval": "inst_level_strict_acc"}


def pick_metric(task_data: dict[str, float], task_name: str) -> tuple[str, float] | None:
    if task_name in TASK_METRIC_OVERRIDE:
        m = TASK_METRIC_OVERRIDE[task_name]
        if m in task_data:
            return m, task_data[m]
    for m in METRIC_PRIORITY:
        if m in task_data:
            return m, task_data[m]
    return None


def main() -> None:
    combined = json.loads((HERE / "all_results.json").read_text())
    if BASELINE_LABEL not in combined:
        raise SystemExit(
            f"baseline run {BASELINE_LABEL!r} not found in all_results.json; "
            f"have: {list(combined.keys())}"
        )

    runs = [r for r in combined.keys() if r != BASELINE_LABEL]
    if not runs:
        raise SystemExit("no quantized runs to compare against the baseline")

    baseline = combined[BASELINE_LABEL]

    # recovery[run][task] = quant_value / baseline_value * 100
    recovery: dict[str, dict[str, float]] = {run: {} for run in runs}
    metric_used: dict[str, str] = {}
    for task in TOP_TASKS + BOTTOM_TASKS:
        if task == "mean":
            continue
        if task not in baseline:
            continue
        base_metric = pick_metric(baseline[task], task)
        if base_metric is None:
            continue
        metric_name, base_val = base_metric
        metric_used[task] = metric_name
        for run in runs:
            run_task = combined[run].get(task, {})
            if metric_name in run_task and base_val != 0:
                recovery[run][task] = run_task[metric_name] / base_val * 100

    # Compute "mean" as the unweighted average of the row's task recoveries
    # for each run (one mean column per row pair would be misleading; we
    # use a single overall mean across all tasks shown, matching the blog
    # figure where "mean n-shot" sits on the top row as an aggregate).
    for run in runs:
        all_task_recoveries = [
            recovery[run][t]
            for t in (TOP_TASKS + BOTTOM_TASKS)
            if t in recovery[run]
        ]
        if all_task_recoveries:
            recovery[run]["mean"] = float(np.mean(all_task_recoveries))

    # Color palette like the AMD figure: blue, orange, green, red, purple
    colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#a855f7"]
    bar_width = 0.8 / max(len(runs), 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharey=True)
    for ax, row_tasks in zip(axes, [TOP_TASKS, BOTTOM_TASKS]):
        x = np.arange(len(row_tasks))
        ax.axhline(y=100, linestyle="--", color="gray", linewidth=1, label="no_quant")
        for i, run in enumerate(runs):
            heights = [recovery[run].get(t, 0) for t in row_tasks]
            offset = (i - (len(runs) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, heights, bar_width,
                label=run, color=colors[i % len(colors)],
                edgecolor="black", linewidth=0.5,
            )
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.3,
                        f"{h:.1f}",
                        ha="center", va="bottom", fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(row_tasks, fontsize=10)
        ax.set_ylim(min(75, min(
            h for run in runs for h in recovery[run].values() if h > 0
        ) - 5) if any(recovery[run] for run in runs) else 75, 102)
        ax.set_ylabel("Recovery vs. non-quantized (100%)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend at the bottom (combine no_quant line + run labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    fig.suptitle(
        "Qwen/Qwen3-8B MXFP6 evaluation\n"
        "Recovery vs. non-quantized baseline (BF16)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_png = HERE / "recovery_chart.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # Also print the underlying numbers as a quick text summary
    print("\nRecovery (% of BF16 baseline):")
    print(f"{'task':25s} {'metric':25s} " + " ".join(f"{r:>20s}" for r in runs))
    for task in TOP_TASKS + BOTTOM_TASKS:
        metric = metric_used.get(task, "(mean)" if task == "mean" else "(n/a)")
        cells = " ".join(
            f"{recovery[r].get(task, float('nan')):>20.2f}" for r in runs
        )
        print(f"{task:25s} {metric:25s} {cells}")


if __name__ == "__main__":
    main()
