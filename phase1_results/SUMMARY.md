# Phase 1: Qwen3-8B accuracy results

| Task | Metric | 01_bf16_baseline | 02_mxfp6_e2m3_rtn | diff |
|---|---|---|---|---|
| _wikitext | perplexity | 9.7299 | 9.7685 | +0.0386 |
| arc_challenge | acc | 0.5520 | 0.5461 | -0.0059 |
| arc_challenge | acc_norm | 0.5666 | 0.5563 | -0.0103 |
| arc_easy | acc | 0.8338 | 0.8295 | -0.0043 |
| arc_easy | acc_norm | 0.8085 | 0.8110 | +0.0025 |
| gsm8k_platinum | exact_match | 0.9007 | 0.8999 | -0.0008 |
| hellaswag | acc | 0.5702 | 0.5701 | -0.0001 |
| hellaswag | acc_norm | 0.7494 | 0.7475 | -0.0019 |
| ifeval | inst_level_loose_acc | 0.4137 | 0.4053 | -0.0084 |
| ifeval | inst_level_strict_acc | 0.3837 | 0.3777 | -0.0060 |
| ifeval | prompt_level_loose_acc | 0.2921 | 0.2754 | -0.0167 |
| ifeval | prompt_level_strict_acc | 0.2532 | 0.2348 | -0.0184 |
| lambada_standard | acc | 0.6097 | 0.6142 | +0.0045 |
| lambada_standard | perplexity | 6.1811 | 6.1231 | -0.0580 |
| leaderboard_mmlu_pro | acc | 0.4752 | 0.4694 | -0.0058 |
| piqa | acc | 0.7650 | 0.7704 | +0.0054 |
| piqa | acc_norm | 0.7748 | 0.7731 | -0.0017 |
| winogrande | acc | 0.6701 | 0.6811 | +0.0110 |