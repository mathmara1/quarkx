"""
Run per-block adaptive grid selection on an HF model, then evaluate with
the SAME pipeline `quantize_quark.py` uses (lm-evaluation-harness via
`quark.contrib.llm_eval.eval_model`).

Using the standard eval pipeline means the results land in the same
markdown table format as Track 1's BF16/MXFP6_e2m3/MXFP6_e3m2 runs, so
adaptive-vs-baseline comparisons are apples-to-apples within Quark's
own measurement methodology.

Usage:
    python run_adaptive.py --model_dir Qwen/Qwen3-0.6B \\
        --formats fp6_e2m3 fp6_e3m2 \\
        --tasks piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval \\
        --eval_batch_size 16 \\
        --output_path adaptive_results

The acceptance test: results should be close to the (separately-run) Quark
baselines on the same model and same task list, with the adaptive scheme
expected to recover some of the gap between BF16 and the worse of the two
fixed-format runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Make adaptive_grid importable when run from its own dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_grid import (  # noqa: E402
    parse_formats,
    quantize_model_weights_inplace,
)

# Quark's eval entry point — same one quantize_quark.py uses.
from quark.contrib.llm_eval import eval_model  # noqa: E402


def build_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Construct the argparse.Namespace `eval_model` expects.

    `eval_model` was designed to be called from `quantize_quark.py` and reads
    ~15 fields off the args namespace. We mirror its defaults exactly so the
    eval behaves identically to a Track 1 run on the same tasks.
    """
    return argparse.Namespace(
        # required by eval_model for tokenizer loading
        model_dir=args.model_dir,
        # WikiText-2 ppl: True = also compute it alongside the task evals
        use_ppl_eval_model=args.use_ppl_eval_model,
        # subset size for ppl_eval / mlperf (-1 == full split)
        num_eval_data=args.num_eval_data,
        # KV-cache PPL is a separate code path; we leave it off
        use_ppl_eval_for_kv_cache=False,
        ppl_eval_for_kv_cache_context_size=1024,
        ppl_eval_for_kv_cache_sample_size=512,
        ppl_eval_for_kv_cache_patch_size=None,
        # MLPerf-ROUGE path off
        use_mlperf_rouge=False,
        eval_data_dir=None,
        # the actual lm-eval invocation params
        tasks=args.tasks,
        eval_batch_size=args.eval_batch_size,
        max_eval_batch_size=args.max_eval_batch_size,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=False,
        output_path=args.output_path,
        log_samples=False,
        # not used in our flow but read by eval_model when constructing model_args
        model_args="",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Adaptive per-block grid selection PTQ runner (uses Quark's eval).",
    )
    # Quant config
    p.add_argument(
        "--model_dir",
        required=True,
        help="HF model id or local path (e.g., Qwen/Qwen3-0.6B).",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["fp6_e2m3", "fp6_e3m2"],
        help="Candidate MX element formats to choose between per block.",
    )
    p.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="MX block size. OCP spec default is 32.",
    )
    p.add_argument(
        "--skip_quantization",
        action="store_true",
        help="Eval BF16 baseline without quantizing (sanity check / direct comparison).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )

    # Eval args (mirror the relevant subset from quantize_quark.py)
    p.add_argument(
        "--tasks",
        default=None,
        type=str,
        metavar="task1,task2",
        help=(
            "Comma-separated lm-eval tasks. Use the same list as Track 1 "
            "for direct comparison: "
            "piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,"
            "hellaswag,gsm8k_platinum,lambada_standard,ifeval"
        ),
    )
    p.add_argument("--use_ppl_eval_model", action="store_true",
                   help="Also compute WikiText-2 PPL in the standard Quark way.")
    p.add_argument("--eval_batch_size", type=str, default="16",
                   help="Eval batch size. 'auto', 'auto:N', or an int.")
    p.add_argument("--max_eval_batch_size", type=int, default=64,
                   help="Max batch size when --eval_batch_size auto.")
    p.add_argument("--num_eval_data", type=int, default=-1,
                   help="Number of eval samples. -1 = full dataset.")
    p.add_argument("--num_fewshot", type=int, default=None,
                   help="Few-shot count override. None = task default.")
    p.add_argument("--apply_chat_template", action="store_true")
    p.add_argument("--output_path", default=None,
                   help="lm-eval --output_path. Writes per-task JSON if set.")
    args = p.parse_args()

    if args.tasks is None and not args.use_ppl_eval_model:
        print(
            "[run] WARNING: neither --tasks nor --use_ppl_eval_model set. "
            "Nothing will be evaluated. Add --tasks <list> or --use_ppl_eval_model."
        )

    formats = parse_formats(args.formats)
    print(f"[run] formats={[f.value for f in formats]} block_size={args.block_size}")

    print(f"[run] loading model {args.model_dir} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    print(f"[run] model loaded in {time.time() - t0:.1f}s", flush=True)

    if not args.skip_quantization:
        print("[run] applying adaptive quantization to linear weights ...", flush=True)
        t0 = time.time()
        summary = quantize_model_weights_inplace(
            model, formats, block_size=args.block_size
        )
        elapsed = time.time() - t0
        print(
            f"[run] quantized {len(summary)} linear layers in {elapsed:.1f}s",
            flush=True,
        )
        # Aggregate format distribution across all layers.
        total = {f.value: 0.0 for f in formats}
        for per_layer in summary.values():
            for fmt_name, frac in per_layer.items():
                total[fmt_name] += frac
        n_layers = max(len(summary), 1)
        print("[run] avg format distribution across quantized layers:")
        for fmt_name, total_frac in total.items():
            print(f"  {fmt_name}: {total_frac / n_layers:.1%}")
    else:
        print("[run] --skip_quantization set; evaluating BF16 baseline", flush=True)

    # Hand off to Quark's eval — same code path as quantize_quark.py.
    print("[run] handing off to Quark eval pipeline ...", flush=True)
    eval_args = build_eval_args(args)
    eval_model(eval_args, model, args.device)


if __name__ == "__main__":
    main()
