# Phase 1: Qwen3-8B accuracy results

| Task | Metric | 01_bf16_baseline | 01_bf16 | 02_mxfp6_e2m3_rtn | all |
|---|---|---|---|---|---|
| 01_bf16 | arc_challenge |  |  |  |  |
| 01_bf16 | arc_easy |  |  |  |  |
| 01_bf16 | gsm8k_platinum |  |  |  |  |
| 01_bf16 | hellaswag |  |  |  |  |
| 01_bf16 | ifeval |  |  |  |  |
| 01_bf16 | lambada_standard |  |  |  |  |
| 01_bf16 | leaderboard_mmlu_pro |  |  |  |  |
| 01_bf16 | piqa |  |  |  |  |
| 01_bf16 | winogrande |  |  |  |  |
| 01_bf16_baseline | _wikitext |  |  |  |  |
| 01_bf16_baseline | arc_challenge |  |  |  |  |
| 01_bf16_baseline | arc_easy |  |  |  |  |
| 01_bf16_baseline | gsm8k_platinum |  |  |  |  |
| 01_bf16_baseline | hellaswag |  |  |  |  |
| 01_bf16_baseline | ifeval |  |  |  |  |
| 01_bf16_baseline | lambada_standard |  |  |  |  |
| 01_bf16_baseline | leaderboard_mmlu_pro |  |  |  |  |
| 01_bf16_baseline | piqa |  |  |  |  |
| 01_bf16_baseline | winogrande |  |  |  |  |
| 02_mxfp6_e2m3_rtn | _wikitext |  |  |  |  |
| 02_mxfp6_e2m3_rtn | arc_challenge |  |  |  |  |
| 02_mxfp6_e2m3_rtn | arc_easy |  |  |  |  |
| 02_mxfp6_e2m3_rtn | gsm8k_platinum |  |  |  |  |
| 02_mxfp6_e2m3_rtn | hellaswag |  |  |  |  |
| 02_mxfp6_e2m3_rtn | ifeval |  |  |  |  |
| 02_mxfp6_e2m3_rtn | lambada_standard |  |  |  |  |
| 02_mxfp6_e2m3_rtn | leaderboard_mmlu_pro |  |  |  |  |
| 02_mxfp6_e2m3_rtn | piqa |  |  |  |  |
| 02_mxfp6_e2m3_rtn | winogrande |  |  |  |  |
| 03_mxfp6_e3m2_rtn | _wikitext |  |  |  |  |
| _wikitext | perplexity | 9.7299 |  | 9.7685 |  |
| arc_challenge | acc | 0.5520 | 0.5520 | 0.5461 |  |
| arc_challenge | acc_norm | 0.5666 | 0.5666 | 0.5563 |  |
| arc_easy | acc | 0.8338 | 0.8338 | 0.8295 |  |
| arc_easy | acc_norm | 0.8085 | 0.8085 | 0.8110 |  |
| gsm8k_platinum | exact_match | 0.9007 | 0.9007 | 0.8999 |  |
| hellaswag | acc | 0.5702 | 0.5702 | 0.5701 |  |
| hellaswag | acc_norm | 0.7494 | 0.7494 | 0.7475 |  |
| ifeval | inst_level_loose_acc | 0.4137 | 0.4137 | 0.4053 |  |
| ifeval | inst_level_strict_acc | 0.3837 | 0.3837 | 0.3777 |  |
| ifeval | prompt_level_loose_acc | 0.2921 | 0.2921 | 0.2754 |  |
| ifeval | prompt_level_strict_acc | 0.2532 | 0.2532 | 0.2348 |  |
| lambada_standard | acc | 0.6097 | 0.6097 | 0.6142 |  |
| lambada_standard | perplexity | 6.1811 | 6.1811 | 6.1231 |  |
| leaderboard_mmlu_pro | acc | 0.4752 | 0.4752 | 0.4694 |  |
| piqa | acc | 0.7650 | 0.7650 | 0.7704 |  |
| piqa | acc_norm | 0.7748 | 0.7748 | 0.7731 |  |
| winogrande | acc | 0.6701 | 0.6701 | 0.6811 |  |