# Statistical tests (ada results)

python eval/stats_ada.py --root results --out STATS_ADA.md

A. lm-eval tasks: paired bootstrap (items aligned across runs) of the score difference
   between model pairs, and memory intact vs zeroed within a model. Holm correction per table.
B. Fine-tuning: mean +- sd over seeds per (task, model, regime, lr); Welch t-tests for the
   key comparisons (memory vs FFN subset vs full, and the learning-rate controls).
C. CounterFact: recall accuracy with McNemar's exact test (memory intact vs zeroed) and a
   paired bootstrap between models.

## A. Benchmarks: paired bootstrap on per-item scores

Difference = first minus second (accuracy points x100). 95% CI from 2000 resamples; p two-sided from the bootstrap standard error, Holm-corrected within each table. `*` marks p_holm < 0.05. WikiText is a perplexity, not an accuracy, and is left out.

### zeroshot

| comparison | task | diff | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---:|
| R1 vs R4 | hellaswag | -0.15 | [-0.70, +0.43] | 0.612 | 1.000 |
| R1 vs R4 | piqa | +0.27 | [-1.20, +1.90] | 0.727 | 1.000 |
| R1 vs R4 | winogrande | -1.89 | [-5.37, +1.42] | 0.264 | 1.000 |
| R1 vs R4 | copa | -2.00 | [-7.00, +4.00] | 0.475 | 1.000 |
| R1 vs R4 | social_iqa | -0.56 | [-1.94, +0.87] | 0.437 | 1.000 |
| R1 vs R4 | arc_easy | +0.51 | [-0.97, +1.94] | 0.504 | 1.000 |
| R1 vs R4 | arc_challenge | +1.28 | [-0.68, +3.16] | 0.201 | 1.000 |
| R1 vs R4 | sciq | +1.10 | [-1.20, +3.30] | 0.340 | 1.000 |
| R1 vs R4 | openbookqa | +0.60 | [-2.00, +3.20] | 0.649 | 1.000 |
| R1 vs R4 | mmlu_continuation | +0.13 | [-0.38, +0.64] | 0.627 | 1.000 |
| R1 vs R4 | lambada_openai | +0.14 | [-0.80, +1.11] | 0.783 | 1.000 |
| R1 vs R4 | boolq | -5.20* | [-6.76, -3.61] | <0.001 | <0.001 |
| R1 vs R4 | race | +0.00 | [-2.11, +2.11] | 1.000 | 1.000 |
| R1 vs R4 | blimp | +1.29* | [+1.01, +1.57] | <0.001 | <0.001 |
| R2 vs R4 | hellaswag | +0.38 | [-0.22, +0.91] | 0.189 | 1.000 |
| R2 vs R4 | piqa | +0.16 | [-1.42, +1.69] | 0.841 | 1.000 |
| R2 vs R4 | winogrande | -0.95 | [-4.34, +2.21] | 0.570 | 1.000 |
| R2 vs R4 | copa | -9.00 | [-15.02, -3.00] | 0.005 | 0.874 |
| R2 vs R4 | arc_easy | -0.17 | [-1.73, +1.31] | 0.827 | 1.000 |
| R2 vs R4 | arc_challenge | -0.51 | [-2.56, +1.45] | 0.615 | 1.000 |
| R2 vs R4 | sciq | +4.00 | [+1.70, +6.30] | <0.001 | 0.101 |
| R2 vs R4 | openbookqa | +0.40 | [-2.20, +2.80] | 0.755 | 1.000 |
| R2 vs R4 | mmlu_continuation | -0.07 | [-0.58, +0.43] | 0.778 | 1.000 |
| R2 vs R4 | lambada_openai | +2.04* | [+1.09, +3.05] | <0.001 | 0.010 |
| R2 vs R4 | boolq | -10.37* | [-12.69, -7.95] | <0.001 | <0.001 |
| R2 vs R4 | race | -1.15 | [-3.16, +0.77] | 0.255 | 1.000 |
| R2 vs R4 | blimp | +1.12* | [+0.86, +1.39] | <0.001 | <0.001 |
| R2 vs R4 | social_iqa | +0.67 | [-0.77, +2.05] | 0.346 | 1.000 |
| R3 vs R4 | hellaswag | +0.34 | [-0.23, +0.93] | 0.250 | 1.000 |
| R3 vs R4 | piqa | +0.38 | [-1.09, +1.85] | 0.609 | 1.000 |
| R3 vs R4 | winogrande | +0.95 | [-2.53, +4.58] | 0.595 | 1.000 |
| R3 vs R4 | copa | -4.00 | [-11.00, +2.00] | 0.239 | 1.000 |
| R3 vs R4 | social_iqa | +1.38 | [-0.10, +2.87] | 0.064 | 1.000 |
| R3 vs R4 | arc_easy | +2.02 | [+0.51, +3.49] | 0.007 | 1.000 |
| R3 vs R4 | arc_challenge | +0.09 | [-1.88, +2.22] | 0.935 | 1.000 |
| R3 vs R4 | sciq | +0.60 | [-1.50, +2.70] | 0.590 | 1.000 |
| R3 vs R4 | openbookqa | +0.80 | [-1.80, +3.60] | 0.577 | 1.000 |
| R3 vs R4 | mmlu_continuation | +0.14 | [-0.36, +0.66] | 0.603 | 1.000 |
| R3 vs R4 | lambada_openai | +0.39 | [-0.58, +1.38] | 0.424 | 1.000 |
| R3 vs R4 | boolq | -3.49* | [-4.89, -2.05] | <0.001 | <0.001 |
| R3 vs R4 | race | -0.57 | [-2.58, +1.53] | 0.595 | 1.000 |
| R3 vs R4 | blimp | +0.90* | [+0.63, +1.21] | <0.001 | <0.001 |
| R5 vs R4 | hellaswag | +0.51 | [-0.08, +1.04] | 0.080 | 1.000 |
| R5 vs R4 | piqa | +0.82 | [-0.55, +2.23] | 0.256 | 1.000 |
| R5 vs R4 | winogrande | -2.60 | [-5.76, +0.47] | 0.112 | 1.000 |
| R5 vs R4 | copa | -4.00 | [-10.00, +2.00] | 0.190 | 1.000 |
| R5 vs R4 | arc_easy | +0.17 | [-1.26, +1.64] | 0.823 | 1.000 |
| R5 vs R4 | arc_challenge | +0.43 | [-1.54, +2.39] | 0.673 | 1.000 |
| R5 vs R4 | sciq | +4.30* | [+2.10, +6.50] | <0.001 | 0.031 |
| R5 vs R4 | openbookqa | +1.20 | [-1.20, +3.80] | 0.355 | 1.000 |
| R5 vs R4 | mmlu_continuation | +0.18 | [-0.31, +0.64] | 0.467 | 1.000 |
| R5 vs R4 | lambada_openai | +1.75* | [+0.81, +2.66] | <0.001 | 0.044 |
| R5 vs R4 | boolq | -2.94* | [-4.01, -1.77] | <0.001 | <0.001 |
| R5 vs R4 | race | +1.05 | [-1.05, +3.16] | 0.316 | 1.000 |
| R5 vs R4 | blimp | +0.21 | [-0.06, +0.49] | 0.134 | 1.000 |
| R5 vs R4 | social_iqa | +0.41 | [-0.92, +1.89] | 0.563 | 1.000 |
| A vs R4 | hellaswag | -0.35 | [-0.92, +0.20] | 0.227 | 1.000 |
| A vs R4 | piqa | +0.16 | [-1.25, +1.63] | 0.826 | 1.000 |
| A vs R4 | winogrande | -0.32 | [-3.71, +2.92] | 0.850 | 1.000 |
| A vs R4 | copa | +1.00 | [-4.00, +6.00] | 0.708 | 1.000 |
| A vs R4 | arc_easy | +0.84 | [-0.63, +2.19] | 0.263 | 1.000 |
| A vs R4 | arc_challenge | -1.02 | [-3.07, +1.02] | 0.325 | 1.000 |
| A vs R4 | sciq | +1.10 | [-1.00, +3.30] | 0.313 | 1.000 |
| A vs R4 | openbookqa | +1.20 | [-1.40, +4.00] | 0.390 | 1.000 |
| A vs R4 | mmlu_continuation | +0.06 | [-0.47, +0.54] | 0.825 | 1.000 |
| A vs R4 | lambada_openai | +0.06 | [-0.85, +0.99] | 0.904 | 1.000 |
| A vs R4 | boolq | -3.33* | [-4.65, -2.02] | <0.001 | <0.001 |
| A vs R4 | race | +0.38 | [-1.72, +2.58] | 0.724 | 1.000 |
| A vs R4 | blimp | +0.92* | [+0.63, +1.19] | <0.001 | <0.001 |
| A vs R4 | social_iqa | -0.56 | [-2.00, +0.82] | 0.430 | 1.000 |
| S1 vs R4 | hellaswag | -0.11 | [-0.68, +0.44] | 0.705 | 1.000 |
| S1 vs R4 | piqa | -0.11 | [-1.52, +1.31] | 0.883 | 1.000 |
| S1 vs R4 | winogrande | -1.42 | [-4.50, +1.90] | 0.388 | 1.000 |
| S1 vs R4 | copa | -10.00 | [-16.00, -4.00] | 0.002 | 0.330 |
| S1 vs R4 | social_iqa | -0.36 | [-1.74, +0.97] | 0.603 | 1.000 |
| S1 vs R4 | arc_easy | +0.29 | [-1.22, +1.85] | 0.707 | 1.000 |
| S1 vs R4 | arc_challenge | -0.17 | [-2.22, +1.71] | 0.864 | 1.000 |
| S1 vs R4 | sciq | +1.10 | [-1.10, +3.20] | 0.325 | 1.000 |
| S1 vs R4 | openbookqa | +1.00 | [-1.60, +3.60] | 0.466 | 1.000 |
| S1 vs R4 | mmlu_continuation | +0.14 | [-0.38, +0.63] | 0.594 | 1.000 |
| S1 vs R4 | lambada_openai | +0.56 | [-0.35, +1.51] | 0.224 | 1.000 |
| S1 vs R4 | boolq | -1.53 | [-2.54, -0.55] | 0.003 | 0.503 |
| S1 vs R4 | race | -0.57 | [-2.58, +1.44] | 0.574 | 1.000 |
| S1 vs R4 | blimp | +1.74* | [+1.48, +2.01] | <0.001 | <0.001 |
| R2 vs R5 | hellaswag | -0.13 | [-0.69, +0.46] | 0.656 | 1.000 |
| R2 vs R5 | piqa | -0.65 | [-2.18, +0.92] | 0.407 | 1.000 |
| R2 vs R5 | winogrande | +1.66 | [-1.74, +4.97] | 0.329 | 1.000 |
| R2 vs R5 | copa | -5.00 | [-12.00, +2.00] | 0.169 | 1.000 |
| R2 vs R5 | arc_easy | -0.34 | [-1.81, +1.14] | 0.657 | 1.000 |
| R2 vs R5 | arc_challenge | -0.94 | [-2.90, +1.02] | 0.362 | 1.000 |
| R2 vs R5 | sciq | -0.30 | [-2.40, +1.80] | 0.777 | 1.000 |
| R2 vs R5 | openbookqa | -0.80 | [-3.60, +2.00] | 0.568 | 1.000 |
| R2 vs R5 | mmlu_continuation | -0.25 | [-0.77, +0.24] | 0.341 | 1.000 |
| R2 vs R5 | lambada_openai | +0.29 | [-0.68, +1.22] | 0.549 | 1.000 |
| R2 vs R5 | boolq | -7.43* | [-9.72, -5.01] | <0.001 | <0.001 |
| R2 vs R5 | race | -2.20 | [-4.40, -0.10] | 0.039 | 1.000 |
| R2 vs R5 | blimp | +0.91* | [+0.63, +1.17] | <0.001 | <0.001 |
| R2 vs R5 | social_iqa | +0.26 | [-1.23, +1.84] | 0.736 | 1.000 |
| R3 vs R5 | hellaswag | -0.17 | [-0.74, +0.40] | 0.562 | 1.000 |
| R3 vs R5 | piqa | -0.44 | [-1.90, +1.04] | 0.569 | 1.000 |
| R3 vs R5 | winogrande | +3.55 | [+0.32, +6.87] | 0.036 | 1.000 |
| R3 vs R5 | copa | +0.00 | [-7.00, +7.00] | 1.000 | 1.000 |
| R3 vs R5 | social_iqa | +0.97 | [-0.46, +2.41] | 0.187 | 1.000 |
| R3 vs R5 | arc_easy | +1.85 | [+0.34, +3.45] | 0.024 | 1.000 |
| R3 vs R5 | arc_challenge | -0.34 | [-2.22, +1.62] | 0.732 | 1.000 |
| R3 vs R5 | sciq | -3.70 | [-6.00, -1.50] | 0.001 | 0.181 |
| R3 vs R5 | openbookqa | -0.40 | [-3.00, +2.40] | 0.771 | 1.000 |
| R3 vs R5 | mmlu_continuation | -0.04 | [-0.57, +0.48] | 0.873 | 1.000 |
| R3 vs R5 | lambada_openai | -1.36 | [-2.29, -0.41] | 0.004 | 0.625 |
| R3 vs R5 | boolq | -0.55 | [-2.08, +0.95] | 0.487 | 1.000 |
| R3 vs R5 | race | -1.63 | [-3.64, +0.38] | 0.122 | 1.000 |
| R3 vs R5 | blimp | +0.69* | [+0.41, +0.96] | <0.001 | <0.001 |
| R1 vs R2 | hellaswag | -0.53 | [-1.10, +0.04] | 0.067 | 1.000 |
| R1 vs R2 | piqa | +0.11 | [-1.25, +1.42] | 0.878 | 1.000 |
| R1 vs R2 | winogrande | -0.95 | [-4.26, +2.37] | 0.578 | 1.000 |
| R1 vs R2 | copa | +7.00 | [+0.00, +14.00] | 0.045 | 1.000 |
| R1 vs R2 | social_iqa | -1.23 | [-2.76, +0.20] | 0.105 | 1.000 |
| R1 vs R2 | arc_easy | +0.67 | [-0.76, +2.19] | 0.376 | 1.000 |
| R1 vs R2 | arc_challenge | +1.79 | [-0.34, +3.92] | 0.091 | 1.000 |
| R1 vs R2 | sciq | -2.90 | [-5.10, -0.90] | 0.006 | 0.874 |
| R1 vs R2 | openbookqa | +0.20 | [-2.20, +2.60] | 0.875 | 1.000 |
| R1 vs R2 | mmlu_continuation | +0.20 | [-0.29, +0.69] | 0.426 | 1.000 |
| R1 vs R2 | lambada_openai | -1.90* | [-2.83, -0.97] | <0.001 | 0.013 |
| R1 vs R2 | boolq | +5.17* | [+2.97, +7.46] | <0.001 | 0.001 |
| R1 vs R2 | race | +1.15 | [-0.86, +3.35] | 0.285 | 1.000 |
| R1 vs R2 | blimp | +0.17 | [-0.11, +0.44] | 0.226 | 1.000 |
| S1 vs R2 | hellaswag | -0.49 | [-1.07, +0.09] | 0.099 | 1.000 |
| S1 vs R2 | piqa | -0.27 | [-1.74, +1.14] | 0.716 | 1.000 |
| S1 vs R2 | winogrande | -0.47 | [-3.63, +2.68] | 0.764 | 1.000 |
| S1 vs R2 | copa | -1.00 | [-6.00, +4.00] | 0.701 | 1.000 |
| S1 vs R2 | social_iqa | -1.02 | [-2.46, +0.46] | 0.165 | 1.000 |
| S1 vs R2 | arc_easy | +0.46 | [-0.97, +1.85] | 0.521 | 1.000 |
| S1 vs R2 | arc_challenge | +0.34 | [-1.62, +2.30] | 0.738 | 1.000 |
| S1 vs R2 | sciq | -2.90 | [-4.80, -1.10] | 0.003 | 0.491 |
| S1 vs R2 | openbookqa | +0.60 | [-2.00, +3.20] | 0.647 | 1.000 |
| S1 vs R2 | mmlu_continuation | +0.21 | [-0.27, +0.70] | 0.412 | 1.000 |
| S1 vs R2 | lambada_openai | -1.47 | [-2.43, -0.54] | 0.002 | 0.382 |
| S1 vs R2 | boolq | +8.84* | [+6.54, +11.04] | <0.001 | <0.001 |
| S1 vs R2 | race | +0.57 | [-1.44, +2.49] | 0.570 | 1.000 |
| S1 vs R2 | blimp | +0.62* | [+0.36, +0.88] | <0.001 | 0.001 |
| A vs R2 | hellaswag | -0.73 | [-1.27, -0.18] | 0.008 | 1.000 |
| A vs R2 | piqa | +0.00 | [-1.52, +1.58] | 1.000 | 1.000 |
| A vs R2 | winogrande | +0.63 | [-2.60, +3.79] | 0.698 | 1.000 |
| A vs R2 | copa | +10.00 | [+5.00, +16.00] | <0.001 | 0.108 |
| A vs R2 | arc_easy | +1.01 | [-0.42, +2.48] | 0.183 | 1.000 |
| A vs R2 | arc_challenge | -0.51 | [-2.56, +1.54] | 0.626 | 1.000 |
| A vs R2 | sciq | -2.90 | [-5.00, -0.80] | 0.006 | 0.904 |
| A vs R2 | openbookqa | +0.80 | [-2.00, +3.60] | 0.570 | 1.000 |
| A vs R2 | mmlu_continuation | +0.13 | [-0.37, +0.61] | 0.616 | 1.000 |
| A vs R2 | lambada_openai | -1.98* | [-2.91, -0.99] | <0.001 | 0.009 |
| A vs R2 | boolq | +7.03* | [+4.62, +9.30] | <0.001 | <0.001 |
| A vs R2 | race | +1.53 | [-0.48, +3.64] | 0.150 | 1.000 |
| A vs R2 | blimp | -0.20 | [-0.46, +0.07] | 0.145 | 1.000 |
| A vs R2 | social_iqa | -1.23 | [-2.66, +0.20] | 0.094 | 1.000 |
| R1 intact vs zeroed | hellaswag | +3.42* | [+2.65, +4.14] | <0.001 | <0.001 |
| R1 intact vs zeroed | piqa | +6.86* | [+5.01, +8.81] | <0.001 | <0.001 |
| R1 intact vs zeroed | winogrande | +1.26 | [-1.98, +4.58] | 0.454 | 1.000 |
| R1 intact vs zeroed | copa | +7.00 | [-1.00, +15.00] | 0.089 | 1.000 |
| R1 intact vs zeroed | social_iqa | +2.10 | [+0.51, +3.63] | 0.010 | 1.000 |
| R1 intact vs zeroed | arc_easy | +6.57* | [+4.92, +8.25] | <0.001 | <0.001 |
| R1 intact vs zeroed | arc_challenge | +2.22 | [-0.00, +4.52] | 0.055 | 1.000 |
| R1 intact vs zeroed | sciq | +9.30* | [+6.70, +11.80] | <0.001 | <0.001 |
| R1 intact vs zeroed | openbookqa | +2.80 | [-0.40, +6.00] | 0.091 | 1.000 |
| R1 intact vs zeroed | mmlu_continuation | +1.05* | [+0.48, +1.61] | <0.001 | 0.044 |
| R1 intact vs zeroed | lambada_openai | +20.16* | [+19.06, +21.33] | <0.001 | <0.001 |
| R1 intact vs zeroed | boolq | +3.55* | [+1.90, +5.35] | <0.001 | 0.010 |
| R1 intact vs zeroed | race | +1.63 | [-0.67, +4.02] | 0.169 | 1.000 |
| R1 intact vs zeroed | blimp | +17.54* | [+17.16, +17.93] | <0.001 | <0.001 |
| R2 intact vs zeroed | hellaswag | +1.32* | [+0.86, +1.81] | <0.001 | <0.001 |
| R2 intact vs zeroed | piqa | +2.01 | [+0.76, +3.32] | 0.002 | 0.334 |
| R2 intact vs zeroed | winogrande | +1.34 | [-1.26, +3.79] | 0.293 | 1.000 |
| R2 intact vs zeroed | copa | -2.00 | [-9.00, +6.00] | 0.597 | 1.000 |
| R2 intact vs zeroed | arc_easy | +1.73 | [+0.46, +2.95] | 0.006 | 0.997 |
| R2 intact vs zeroed | arc_challenge | +0.68 | [-1.02, +2.39] | 0.433 | 1.000 |
| R2 intact vs zeroed | sciq | +4.50* | [+2.80, +6.30] | <0.001 | <0.001 |
| R2 intact vs zeroed | openbookqa | -0.40 | [-2.80, +2.00] | 0.748 | 1.000 |
| R2 intact vs zeroed | mmlu_continuation | +0.13 | [-0.27, +0.53] | 0.531 | 1.000 |
| R2 intact vs zeroed | lambada_openai | +8.38* | [+7.55, +9.26] | <0.001 | <0.001 |
| R2 intact vs zeroed | boolq | -6.82* | [-8.78, -4.89] | <0.001 | <0.001 |
| R2 intact vs zeroed | race | -0.29 | [-2.01, +1.44] | 0.745 | 1.000 |
| R2 intact vs zeroed | blimp | +8.07* | [+7.77, +8.38] | <0.001 | <0.001 |
| R2 intact vs zeroed | social_iqa | +1.54 | [+0.51, +2.61] | 0.005 | 0.863 |
| R3 intact vs zeroed | hellaswag | +1.03* | [+0.53, +1.54] | <0.001 | 0.012 |
| R3 intact vs zeroed | piqa | +1.47 | [+0.11, +2.88] | 0.037 | 1.000 |
| R3 intact vs zeroed | winogrande | +0.95 | [-1.34, +3.24] | 0.421 | 1.000 |
| R3 intact vs zeroed | copa | +4.00 | [-2.00, +10.00] | 0.198 | 1.000 |
| R3 intact vs zeroed | social_iqa | +0.92 | [-0.20, +2.00] | 0.105 | 1.000 |
| R3 intact vs zeroed | arc_easy | +2.90* | [+1.64, +4.29] | <0.001 | 0.003 |
| R3 intact vs zeroed | arc_challenge | +2.05 | [+0.51, +3.67] | 0.013 | 1.000 |
| R3 intact vs zeroed | sciq | -0.40 | [-2.10, +1.60] | 0.676 | 1.000 |
| R3 intact vs zeroed | openbookqa | +0.80 | [-1.40, +3.00] | 0.490 | 1.000 |
| R3 intact vs zeroed | mmlu_continuation | +0.68 | [+0.21, +1.14] | 0.005 | 0.731 |
| R3 intact vs zeroed | lambada_openai | +7.39* | [+6.50, +8.31] | <0.001 | <0.001 |
| R3 intact vs zeroed | boolq | -3.24* | [-4.59, -1.90] | <0.001 | <0.001 |
| R3 intact vs zeroed | race | -0.77 | [-2.30, +0.77] | 0.328 | 1.000 |
| R3 intact vs zeroed | blimp | +6.61* | [+6.34, +6.90] | <0.001 | <0.001 |
| A intact vs zeroed | hellaswag | +3.08* | [+2.32, +3.85] | <0.001 | <0.001 |
| A intact vs zeroed | piqa | +6.64* | [+4.62, +8.60] | <0.001 | <0.001 |
| A intact vs zeroed | winogrande | +1.03 | [-2.29, +4.58] | 0.557 | 1.000 |
| A intact vs zeroed | copa | +11.00 | [+2.00, +20.00] | 0.020 | 1.000 |
| A intact vs zeroed | arc_easy | +8.63* | [+6.82, +10.27] | <0.001 | <0.001 |
| A intact vs zeroed | arc_challenge | +2.39 | [+0.17, +4.61] | 0.035 | 1.000 |
| A intact vs zeroed | sciq | +11.30* | [+8.70, +13.90] | <0.001 | <0.001 |
| A intact vs zeroed | openbookqa | +2.60 | [-0.80, +6.20] | 0.137 | 1.000 |
| A intact vs zeroed | mmlu_continuation | +0.80 | [+0.26, +1.33] | 0.004 | 0.595 |
| A intact vs zeroed | lambada_openai | +20.76* | [+19.66, +21.85] | <0.001 | <0.001 |
| A intact vs zeroed | boolq | +0.55 | [-0.86, +1.87] | 0.434 | 1.000 |
| A intact vs zeroed | race | +3.25 | [+1.05, +5.45] | 0.004 | 0.671 |
| A intact vs zeroed | blimp | +18.43* | [+18.04, +18.82] | <0.001 | <0.001 |
| A intact vs zeroed | social_iqa | +1.59 | [+0.05, +3.02] | 0.041 | 1.000 |
| S1 intact vs zeroed | hellaswag | +1.35* | [+0.70, +2.03] | <0.001 | 0.012 |
| S1 intact vs zeroed | piqa | +3.92* | [+2.29, +5.50] | <0.001 | <0.001 |
| S1 intact vs zeroed | winogrande | -0.32 | [-3.08, +2.53] | 0.824 | 1.000 |
| S1 intact vs zeroed | copa | -4.00 | [-11.00, +3.00] | 0.286 | 1.000 |
| S1 intact vs zeroed | social_iqa | +0.87 | [-0.36, +2.15] | 0.184 | 1.000 |
| S1 intact vs zeroed | arc_easy | +2.74* | [+1.43, +4.04] | <0.001 | 0.010 |
| S1 intact vs zeroed | arc_challenge | +0.34 | [-1.37, +2.13] | 0.702 | 1.000 |
| S1 intact vs zeroed | sciq | +1.70 | [-0.10, +3.40] | 0.053 | 1.000 |
| S1 intact vs zeroed | openbookqa | +4.60 | [+2.20, +7.20] | <0.001 | 0.051 |
| S1 intact vs zeroed | mmlu_continuation | +0.46 | [+0.04, +0.86] | 0.030 | 1.000 |
| S1 intact vs zeroed | lambada_openai | +11.10* | [+10.21, +12.05] | <0.001 | <0.001 |
| S1 intact vs zeroed | boolq | +0.12 | [-0.89, +1.10] | 0.803 | 1.000 |
| S1 intact vs zeroed | race | -0.86 | [-2.78, +1.15] | 0.382 | 1.000 |
| S1 intact vs zeroed | blimp | +10.65* | [+10.32, +10.98] | <0.001 | <0.001 |

Significant tasks per comparison (better / worse / total):

- R1 vs R4: 1 / 1 / 14
- R2 vs R4: 2 / 1 / 14
- R3 vs R4: 1 / 1 / 14
- R5 vs R4: 2 / 1 / 14
- A vs R4: 1 / 1 / 14
- S1 vs R4: 1 / 0 / 14
- R2 vs R5: 1 / 1 / 14
- R3 vs R5: 1 / 0 / 14
- R1 vs R2: 1 / 1 / 14
- S1 vs R2: 2 / 0 / 14
- A vs R2: 1 / 1 / 14
- R1 intact vs zeroed: 8 / 0 / 14
- R2 intact vs zeroed: 4 / 1 / 14
- R3 intact vs zeroed: 4 / 1 / 14
- A intact vs zeroed: 6 / 0 / 14
- S1 intact vs zeroed: 5 / 0 / 14

### paper

| comparison | task | diff | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---:|
| R1 vs R4 | mmlu | -1.84 | [-2.87, -0.78] | <0.001 | 0.084 |
| R1 vs R4 | arc_easy | -2.33 | [-6.67, +2.33] | 0.332 | 1.000 |
| R1 vs R4 | arc_challenge | +2.33 | [-1.00, +6.33] | 0.216 | 1.000 |
| R1 vs R4 | triviaqa | +0.67 | [-1.33, +2.67] | 0.536 | 1.000 |
| R1 vs R4 | bbh_fewshot | -0.94 | [-2.17, +0.23] | 0.123 | 1.000 |
| R1 vs R4 | winogrande | -4.00 | [-10.67, +2.67] | 0.229 | 1.000 |
| R1 vs R4 | drop | +0.13 | [-0.93, +1.11] | 0.799 | 1.000 |
| R1 vs R4 | race | -0.33 | [-4.00, +3.33] | 0.856 | 1.000 |
| R2 vs R4 | mmlu | -1.96* | [-2.95, -1.04] | <0.001 | 0.006 |
| R2 vs R4 | arc_easy | -3.00 | [-8.00, +2.00] | 0.237 | 1.000 |
| R2 vs R4 | arc_challenge | +0.00 | [-4.33, +4.33] | 1.000 | 1.000 |
| R2 vs R4 | triviaqa | -1.00 | [-3.33, +1.33] | 0.402 | 1.000 |
| R2 vs R4 | bbh_fewshot | +0.43 | [-0.75, +1.60] | 0.473 | 1.000 |
| R2 vs R4 | winogrande | -7.00 | [-13.33, -0.66] | 0.030 | 1.000 |
| R2 vs R4 | drop | -0.05 | [-1.35, +1.17] | 0.939 | 1.000 |
| R2 vs R4 | race | +0.67 | [-3.33, +4.33] | 0.738 | 1.000 |
| R3 vs R4 | mmlu | -0.94 | [-1.92, +0.10] | 0.075 | 1.000 |
| R3 vs R4 | arc_easy | -3.33 | [-8.33, +1.33] | 0.166 | 1.000 |
| R3 vs R4 | arc_challenge | +0.67 | [-3.00, +4.33] | 0.723 | 1.000 |
| R3 vs R4 | triviaqa | +1.00 | [-1.00, +3.00] | 0.326 | 1.000 |
| R3 vs R4 | bbh_fewshot | -1.84 | [-2.99, -0.65] | 0.002 | 0.254 |
| R3 vs R4 | winogrande | -4.67 | [-11.67, +2.00] | 0.181 | 1.000 |
| R3 vs R4 | drop | -0.59 | [-1.74, +0.40] | 0.283 | 1.000 |
| R3 vs R4 | race | +2.00 | [-1.67, +5.67] | 0.283 | 1.000 |
| R5 vs R4 | mmlu | -2.09* | [-3.00, -1.10] | <0.001 | 0.001 |
| R5 vs R4 | arc_easy | -0.33 | [-5.00, +4.33] | 0.891 | 1.000 |
| R5 vs R4 | arc_challenge | +3.33 | [-0.67, +7.33] | 0.105 | 1.000 |
| R5 vs R4 | triviaqa | +0.67 | [-1.67, +3.00] | 0.560 | 1.000 |
| R5 vs R4 | bbh_fewshot | +0.41 | [-1.00, +1.78] | 0.561 | 1.000 |
| R5 vs R4 | winogrande | -7.33 | [-13.67, -0.67] | 0.031 | 1.000 |
| R5 vs R4 | drop | +0.15 | [-0.75, +1.05] | 0.739 | 1.000 |
| R5 vs R4 | race | +3.00 | [-0.67, +6.67] | 0.101 | 1.000 |
| A vs R4 | mmlu | -1.17 | [-2.07, -0.28] | 0.012 | 1.000 |
| A vs R4 | arc_easy | -3.00 | [-8.00, +2.00] | 0.233 | 1.000 |
| A vs R4 | arc_challenge | +0.00 | [-4.33, +4.33] | 1.000 | 1.000 |
| A vs R4 | winogrande | -1.67 | [-7.67, +4.33] | 0.599 | 1.000 |
| A vs R4 | race | +3.00 | [-1.00, +7.00] | 0.149 | 1.000 |
| A vs R4 | triviaqa | +0.67 | [-1.33, +2.67] | 0.519 | 1.000 |
| A vs R4 | bbh_fewshot | +1.27 | [+0.12, +2.47] | 0.036 | 1.000 |
| A vs R4 | drop | +0.29 | [-0.80, +1.34] | 0.606 | 1.000 |
| S1 vs R4 | mmlu | -1.20 | [-2.27, -0.14] | 0.028 | 1.000 |
| S1 vs R4 | arc_easy | +0.00 | [-5.00, +4.67] | 1.000 | 1.000 |
| S1 vs R4 | arc_challenge | +1.00 | [-3.33, +5.00] | 0.642 | 1.000 |
| S1 vs R4 | triviaqa | +0.33 | [-1.67, +2.33] | 0.742 | 1.000 |
| S1 vs R4 | bbh_fewshot | -0.29 | [-1.47, +0.89] | 0.628 | 1.000 |
| S1 vs R4 | winogrande | -0.33 | [-7.00, +6.33] | 0.923 | 1.000 |
| S1 vs R4 | drop | +0.37 | [-0.71, +1.44] | 0.505 | 1.000 |
| S1 vs R4 | race | +1.33 | [-2.33, +5.00] | 0.461 | 1.000 |
| R2 vs R5 | mmlu | +0.13 | [-0.86, +1.04] | 0.793 | 1.000 |
| R2 vs R5 | arc_easy | -2.67 | [-7.33, +1.67] | 0.243 | 1.000 |
| R2 vs R5 | arc_challenge | -3.33 | [-7.33, +0.67] | 0.096 | 1.000 |
| R2 vs R5 | triviaqa | -1.67 | [-4.00, +0.67] | 0.177 | 1.000 |
| R2 vs R5 | bbh_fewshot | +0.02 | [-1.17, +1.12] | 0.978 | 1.000 |
| R2 vs R5 | winogrande | +0.33 | [-6.33, +6.67] | 0.921 | 1.000 |
| R2 vs R5 | drop | -0.20 | [-1.50, +0.97] | 0.752 | 1.000 |
| R2 vs R5 | race | -2.33 | [-6.33, +1.67] | 0.250 | 1.000 |
| R3 vs R5 | mmlu | +1.16 | [+0.17, +2.17] | 0.027 | 1.000 |
| R3 vs R5 | arc_easy | -3.00 | [-7.33, +1.67] | 0.203 | 1.000 |
| R3 vs R5 | arc_challenge | -2.67 | [-6.00, +1.00] | 0.135 | 1.000 |
| R3 vs R5 | triviaqa | +0.33 | [-2.33, +3.00] | 0.800 | 1.000 |
| R3 vs R5 | bbh_fewshot | -2.26* | [-3.23, -1.34] | <0.001 | <0.001 |
| R3 vs R5 | winogrande | +2.67 | [-4.00, +9.67] | 0.439 | 1.000 |
| R3 vs R5 | drop | -0.74 | [-1.76, +0.27] | 0.154 | 1.000 |
| R3 vs R5 | race | -1.00 | [-4.67, +2.33] | 0.575 | 1.000 |
| R1 vs R2 | mmlu | +0.12 | [-0.96, +1.21] | 0.832 | 1.000 |
| R1 vs R2 | arc_easy | +0.67 | [-4.00, +5.00] | 0.776 | 1.000 |
| R1 vs R2 | arc_challenge | +2.33 | [-1.67, +6.33] | 0.245 | 1.000 |
| R1 vs R2 | triviaqa | +1.67 | [+0.00, +3.67] | 0.094 | 1.000 |
| R1 vs R2 | bbh_fewshot | -1.37 | [-2.38, -0.34] | 0.009 | 0.997 |
| R1 vs R2 | winogrande | +3.00 | [-2.67, +8.67] | 0.310 | 1.000 |
| R1 vs R2 | drop | +0.18 | [-0.97, +1.21] | 0.748 | 1.000 |
| R1 vs R2 | race | -1.00 | [-5.00, +3.00] | 0.631 | 1.000 |
| S1 vs R2 | mmlu | +0.76 | [-0.37, +1.87] | 0.191 | 1.000 |
| S1 vs R2 | arc_easy | +3.00 | [-1.67, +8.00] | 0.234 | 1.000 |
| S1 vs R2 | arc_challenge | +1.00 | [-3.00, +5.00] | 0.608 | 1.000 |
| S1 vs R2 | triviaqa | +1.33 | [-1.00, +3.33] | 0.238 | 1.000 |
| S1 vs R2 | bbh_fewshot | -0.72 | [-1.64, +0.26] | 0.148 | 1.000 |
| S1 vs R2 | winogrande | +6.67 | [+0.00, +13.00] | 0.047 | 1.000 |
| S1 vs R2 | drop | +0.42 | [-0.60, +1.52] | 0.428 | 1.000 |
| S1 vs R2 | race | +0.67 | [-3.33, +4.67] | 0.749 | 1.000 |
| A vs R2 | mmlu | +0.79 | [-0.27, +1.80] | 0.135 | 1.000 |
| A vs R2 | arc_easy | +0.00 | [-5.00, +5.00] | 1.000 | 1.000 |
| A vs R2 | arc_challenge | +0.00 | [-4.33, +4.00] | 1.000 | 1.000 |
| A vs R2 | winogrande | +5.33 | [-0.67, +11.33] | 0.089 | 1.000 |
| A vs R2 | race | +2.33 | [-1.67, +6.33] | 0.243 | 1.000 |
| A vs R2 | triviaqa | +1.67 | [-0.33, +3.67] | 0.095 | 1.000 |
| A vs R2 | bbh_fewshot | +0.84 | [-0.31, +2.03] | 0.145 | 1.000 |
| A vs R2 | drop | +0.34 | [-0.76, +1.54] | 0.571 | 1.000 |
| R1 intact vs zeroed | mmlu | -0.08 | [-1.11, +0.94] | 0.875 | 1.000 |
| R1 intact vs zeroed | arc_easy | +9.00* | [+4.00, +14.00] | <0.001 | 0.035 |
| R1 intact vs zeroed | arc_challenge | +2.33 | [-1.33, +6.33] | 0.231 | 1.000 |
| R1 intact vs zeroed | triviaqa | +3.33 | [+1.33, +5.67] | 0.003 | 0.302 |
| R1 intact vs zeroed | bbh_fewshot | +16.60* | [+15.62, +17.62] | <0.001 | <0.001 |
| R1 intact vs zeroed | winogrande | +0.67 | [-5.00, +6.67] | 0.817 | 1.000 |
| R1 intact vs zeroed | drop | +2.72* | [+1.77, +3.71] | <0.001 | <0.001 |
| R1 intact vs zeroed | race | +3.33 | [-1.00, +7.67] | 0.132 | 1.000 |
| R2 intact vs zeroed | mmlu | -1.48 | [-2.48, -0.50] | 0.004 | 0.418 |
| R2 intact vs zeroed | arc_easy | +0.67 | [-3.00, +4.33] | 0.718 | 1.000 |
| R2 intact vs zeroed | arc_challenge | +0.67 | [-2.00, +3.33] | 0.616 | 1.000 |
| R2 intact vs zeroed | triviaqa | +0.00 | [-2.00, +2.00] | 1.000 | 1.000 |
| R2 intact vs zeroed | bbh_fewshot | +12.33* | [+11.30, +13.35] | <0.001 | <0.001 |
| R2 intact vs zeroed | winogrande | +1.00 | [-3.00, +5.33] | 0.636 | 1.000 |
| R2 intact vs zeroed | drop | -0.34 | [-1.60, +0.84] | 0.587 | 1.000 |
| R2 intact vs zeroed | race | +2.00 | [-1.33, +5.33] | 0.250 | 1.000 |
| R3 intact vs zeroed | mmlu | -0.24 | [-1.16, +0.64] | 0.597 | 1.000 |
| R3 intact vs zeroed | arc_easy | +2.33 | [-2.00, +6.33] | 0.263 | 1.000 |
| R3 intact vs zeroed | arc_challenge | +2.33 | [-1.00, +6.00] | 0.199 | 1.000 |
| R3 intact vs zeroed | triviaqa | +2.33 | [+0.67, +4.33] | 0.017 | 1.000 |
| R3 intact vs zeroed | bbh_fewshot | +23.48* | [+22.45, +24.54] | <0.001 | <0.001 |
| R3 intact vs zeroed | winogrande | -1.33 | [-5.67, +3.33] | 0.562 | 1.000 |
| R3 intact vs zeroed | drop | +1.01 | [+0.11, +2.02] | 0.043 | 1.000 |
| R3 intact vs zeroed | race | +1.67 | [-1.67, +5.33] | 0.359 | 1.000 |
| A intact vs zeroed | mmlu | -0.62 | [-1.60, +0.31] | 0.199 | 1.000 |
| A intact vs zeroed | arc_easy | +4.00 | [-0.67, +8.67] | 0.102 | 1.000 |
| A intact vs zeroed | arc_challenge | +0.33 | [-4.00, +4.67] | 0.878 | 1.000 |
| A intact vs zeroed | winogrande | +0.67 | [-5.33, +7.00] | 0.829 | 1.000 |
| A intact vs zeroed | race | +6.33 | [+1.67, +10.67] | 0.008 | 0.834 |
| A intact vs zeroed | triviaqa | +3.00 | [+1.00, +5.01] | 0.006 | 0.646 |
| A intact vs zeroed | bbh_fewshot | +17.77* | [+16.50, +19.01] | <0.001 | <0.001 |
| A intact vs zeroed | drop | +3.01* | [+2.02, +4.28] | <0.001 | <0.001 |
| S1 intact vs zeroed | mmlu | -0.76 | [-1.56, +0.02] | 0.058 | 1.000 |
| S1 intact vs zeroed | arc_easy | +2.67 | [-1.00, +6.67] | 0.160 | 1.000 |
| S1 intact vs zeroed | arc_challenge | +2.00 | [-1.00, +5.33] | 0.230 | 1.000 |
| S1 intact vs zeroed | triviaqa | +2.33 | [+0.33, +4.67] | 0.035 | 1.000 |
| S1 intact vs zeroed | bbh_fewshot | +23.50* | [+22.47, +24.54] | <0.001 | <0.001 |
| S1 intact vs zeroed | winogrande | +1.33 | [-4.33, +7.00] | 0.633 | 1.000 |
| S1 intact vs zeroed | drop | +2.63* | [+1.42, +4.03] | <0.001 | 0.010 |
| S1 intact vs zeroed | race | +3.67 | [-0.33, +7.67] | 0.071 | 1.000 |

Significant tasks per comparison (better / worse / total):

- R1 vs R4: 0 / 0 / 8
- R2 vs R4: 0 / 1 / 8
- R3 vs R4: 0 / 0 / 8
- R5 vs R4: 0 / 1 / 8
- A vs R4: 0 / 0 / 8
- S1 vs R4: 0 / 0 / 8
- R2 vs R5: 0 / 0 / 8
- R3 vs R5: 0 / 1 / 8
- R1 vs R2: 0 / 0 / 8
- S1 vs R2: 0 / 0 / 8
- A vs R2: 0 / 0 / 8
- R1 intact vs zeroed: 3 / 0 / 8
- R2 intact vs zeroed: 1 / 0 / 8
- R3 intact vs zeroed: 1 / 0 / 8
- A intact vs zeroed: 2 / 0 / 8
- S1 intact vs zeroed: 2 / 0 / 8

### fewshot

| comparison | task | diff | 95% CI | p | p_holm |
|---|---|---:|---:|---:|---:|
| R1 vs R4 | arc_easy | -4.33 | [-9.00, +0.33] | 0.070 | 1.000 |
| R1 vs R4 | hellaswag | +0.00 | [-3.33, +3.33] | 1.000 | 1.000 |
| R2 vs R4 | arc_easy | +1.00 | [-3.67, +5.33] | 0.667 | 1.000 |
| R2 vs R4 | hellaswag | +0.33 | [-3.00, +3.67] | 0.841 | 1.000 |
| R3 vs R4 | arc_easy | -4.67 | [-9.33, -0.33] | 0.041 | 0.860 |
| R3 vs R4 | hellaswag | +1.33 | [-2.33, +5.00] | 0.471 | 1.000 |
| R5 vs R4 | arc_easy | -1.67 | [-6.33, +3.00] | 0.486 | 1.000 |
| R5 vs R4 | hellaswag | -0.33 | [-4.01, +3.33] | 0.862 | 1.000 |
| A vs R4 | arc_easy | -3.00 | [-7.33, +1.33] | 0.170 | 1.000 |
| A vs R4 | hellaswag | +1.33 | [-2.33, +5.00] | 0.496 | 1.000 |
| S1 vs R4 | arc_easy | -2.00 | [-6.67, +2.33] | 0.393 | 1.000 |
| S1 vs R4 | hellaswag | +2.33 | [-1.00, +5.67] | 0.167 | 1.000 |
| R2 vs R5 | arc_easy | +2.67 | [-2.00, +7.33] | 0.262 | 1.000 |
| R2 vs R5 | hellaswag | +0.67 | [-3.33, +4.67] | 0.739 | 1.000 |
| R3 vs R5 | arc_easy | -3.00 | [-8.00, +1.67] | 0.213 | 1.000 |
| R3 vs R5 | hellaswag | +1.67 | [-2.00, +5.67] | 0.390 | 1.000 |
| R1 vs R2 | arc_easy | -5.33 | [-10.33, -0.33] | 0.038 | 0.840 |
| R1 vs R2 | hellaswag | -0.33 | [-3.33, +2.33] | 0.817 | 1.000 |
| S1 vs R2 | arc_easy | -3.00 | [-7.67, +2.00] | 0.212 | 1.000 |
| S1 vs R2 | hellaswag | +2.00 | [-1.00, +5.67] | 0.237 | 1.000 |
| A vs R2 | arc_easy | -4.00 | [-8.67, +0.67] | 0.096 | 1.000 |
| A vs R2 | hellaswag | +1.00 | [-2.33, +4.33] | 0.563 | 1.000 |

Significant tasks per comparison (better / worse / total):

- R1 vs R4: 0 / 0 / 2
- R2 vs R4: 0 / 0 / 2
- R3 vs R4: 0 / 0 / 2
- R5 vs R4: 0 / 0 / 2
- A vs R4: 0 / 0 / 2
- S1 vs R4: 0 / 0 / 2
- R2 vs R5: 0 / 0 / 2
- R3 vs R5: 0 / 0 / 2
- R1 vs R2: 0 / 0 / 2
- S1 vs R2: 0 / 0 / 2
- A vs R2: 0 / 0 / 2

## B. Fine-tuning over seeds

| task | model | regime | lr | n | mean | sd | seeds |
|---|---|---|---:|---:|---:|---:|---|
| boolq | R1 | all | 5e-05 | 3 | 0.702 | 0.007 | 0:0.694, 1:0.706, 2:0.706 |
| boolq | R1 | backbone | 5e-05 | 1 | 0.701 | nan | 0:0.701 |
| boolq | R1 | memory | 0.0005 | 3 | 0.672 | 0.007 | 0:0.678, 1:0.664, 2:0.673 |
| boolq | R1 | none | 5e-05 | 1 | 0.572 | nan | 0:0.572 |
| boolq | R2 | all | 5e-05 | 3 | 0.700 | 0.009 | 0:0.710, 1:0.697, 2:0.693 |
| boolq | R2 | backbone | 5e-05 | 1 | 0.710 | nan | 0:0.710 |
| boolq | R2 | ffn12 | 5e-05 | 3 | 0.708 | 0.005 | 0:0.713, 1:0.704, 2:0.708 |
| boolq | R2 | ffn12 | 0.0005 | 3 | 0.635 | 0.012 | 0:0.630, 1:0.648, 2:0.627 |
| boolq | R2 | memory | 5e-05 | 3 | 0.628 | 0.011 | 0:0.627, 1:0.617, 2:0.640 |
| boolq | R2 | memory | 0.0005 | 3 | 0.673 | 0.009 | 0:0.680, 1:0.664, 2:0.677 |
| boolq | R2 | none | 5e-05 | 1 | 0.529 | nan | 0:0.529 |
| boolq | R3 | all | 5e-05 | 1 | 0.704 | nan | 0:0.704 |
| boolq | R3 | backbone | 5e-05 | 1 | 0.705 | nan | 0:0.705 |
| boolq | R3 | memory | 0.0005 | 1 | 0.674 | nan | 0:0.674 |
| boolq | R3 | none | 5e-05 | 1 | 0.585 | nan | 0:0.585 |
| boolq | R4 | all | 5e-05 | 3 | 0.696 | 0.004 | 0:0.700, 1:0.697, 2:0.693 |
| boolq | R4 | ffn8 | 5e-05 | 3 | 0.686 | 0.005 | 0:0.691, 1:0.685, 2:0.681 |
| boolq | R4 | ffn8 | 0.0005 | 3 | 0.661 | 0.015 | 0:0.644, 1:0.667, 2:0.672 |
| boolq | R4 | none | 5e-05 | 1 | 0.622 | nan | 0:0.622 |
| boolq | R5 | all | 5e-05 | 1 | 0.687 | nan | 0:0.687 |
| boolq | R5 | none | 5e-05 | 1 | 0.597 | nan | 0:0.597 |
| boolq | S1 | all | 5e-05 | 3 | 0.695 | 0.004 | 0:0.700, 1:0.693, 2:0.692 |
| boolq | S1 | backbone | 5e-05 | 1 | 0.695 | nan | 0:0.695 |
| boolq | S1 | memory | 0.0005 | 3 | 0.671 | 0.005 | 0:0.671, 1:0.666, 2:0.677 |
| boolq | S1 | none | 5e-05 | 1 | 0.613 | nan | 0:0.613 |
| ner | R1 | all | 5e-05 | 3 | 0.716 | 0.003 | 0:0.713, 1:0.716, 2:0.719 |
| ner | R1 | backbone | 5e-05 | 1 | 0.712 | nan | 0:0.712 |
| ner | R1 | memory | 0.0005 | 3 | 0.682 | 0.002 | 0:0.680, 1:0.682, 2:0.684 |
| ner | R1 | none | 5e-05 | 1 | 0.628 | nan | 0:0.628 |
| ner | R2 | all | 5e-05 | 3 | 0.715 | 0.001 | 0:0.715, 1:0.715, 2:0.716 |
| ner | R2 | backbone | 5e-05 | 1 | 0.715 | nan | 0:0.715 |
| ner | R2 | ffn12 | 5e-05 | 3 | 0.716 | 0.001 | 0:0.715, 1:0.715, 2:0.717 |
| ner | R2 | ffn12 | 0.0005 | 3 | 0.697 | 0.001 | 0:0.697, 1:0.696, 2:0.697 |
| ner | R2 | memory | 5e-05 | 3 | 0.658 | 0.001 | 0:0.656, 1:0.659, 2:0.658 |
| ner | R2 | memory | 0.0005 | 3 | 0.679 | 0.002 | 0:0.677, 1:0.681, 2:0.677 |
| ner | R2 | none | 5e-05 | 1 | 0.635 | nan | 0:0.635 |
| ner | R3 | all | 5e-05 | 1 | 0.717 | nan | 0:0.717 |
| ner | R3 | backbone | 5e-05 | 1 | 0.718 | nan | 0:0.718 |
| ner | R3 | memory | 0.0005 | 1 | 0.685 | nan | 0:0.685 |
| ner | R3 | none | 5e-05 | 1 | 0.625 | nan | 0:0.625 |
| ner | R4 | all | 5e-05 | 3 | 0.719 | 0.003 | 0:0.721, 1:0.716, 2:0.722 |
| ner | R4 | ffn8 | 5e-05 | 3 | 0.711 | 0.003 | 0:0.711, 1:0.709, 2:0.714 |
| ner | R4 | ffn8 | 0.0005 | 3 | 0.705 | 0.002 | 0:0.705, 1:0.706, 2:0.703 |
| ner | R4 | none | 5e-05 | 1 | 0.630 | nan | 0:0.630 |
| ner | R5 | all | 5e-05 | 1 | 0.714 | nan | 0:0.714 |
| ner | R5 | none | 5e-05 | 1 | 0.631 | nan | 0:0.631 |
| ner | S1 | all | 5e-05 | 3 | 0.715 | 0.005 | 0:0.713, 1:0.711, 2:0.721 |
| ner | S1 | backbone | 5e-05 | 1 | 0.717 | nan | 0:0.717 |
| ner | S1 | memory | 0.0005 | 3 | 0.676 | 0.004 | 0:0.675, 1:0.672, 2:0.681 |
| ner | S1 | none | 5e-05 | 1 | 0.642 | nan | 0:0.642 |
| sst2 | R1 | all | 5e-05 | 3 | 0.893 | 0.003 | 0:0.890, 1:0.894, 2:0.894 |
| sst2 | R1 | backbone | 5e-05 | 1 | 0.900 | nan | 0:0.900 |
| sst2 | R1 | memory | 0.0005 | 3 | 0.876 | 0.006 | 0:0.869, 1:0.880, 2:0.878 |
| sst2 | R1 | none | 5e-05 | 1 | 0.662 | nan | 0:0.662 |
| sst2 | R2 | all | 5e-05 | 3 | 0.902 | 0.002 | 0:0.900, 1:0.901, 2:0.905 |
| sst2 | R2 | backbone | 5e-05 | 1 | 0.904 | nan | 0:0.904 |
| sst2 | R2 | ffn12 | 5e-05 | 3 | 0.911 | 0.010 | 0:0.915, 1:0.917, 2:0.899 |
| sst2 | R2 | ffn12 | 0.0005 | 3 | 0.827 | 0.005 | 0:0.827, 1:0.822, 2:0.831 |
| sst2 | R2 | memory | 5e-05 | 3 | 0.853 | 0.007 | 0:0.860, 1:0.853, 2:0.845 |
| sst2 | R2 | memory | 0.0005 | 3 | 0.883 | 0.006 | 0:0.885, 1:0.876, 2:0.886 |
| sst2 | R2 | none | 5e-05 | 1 | 0.560 | nan | 0:0.560 |
| sst2 | R3 | all | 5e-05 | 1 | 0.894 | nan | 0:0.894 |
| sst2 | R3 | backbone | 5e-05 | 1 | 0.899 | nan | 0:0.899 |
| sst2 | R3 | memory | 0.0005 | 1 | 0.876 | nan | 0:0.876 |
| sst2 | R3 | none | 5e-05 | 1 | 0.649 | nan | 0:0.649 |
| sst2 | R4 | all | 5e-05 | 3 | 0.903 | 0.007 | 0:0.896, 1:0.909, 2:0.905 |
| sst2 | R4 | ffn8 | 5e-05 | 3 | 0.899 | 0.006 | 0:0.893, 1:0.905, 2:0.899 |
| sst2 | R4 | ffn8 | 0.0005 | 3 | 0.869 | 0.005 | 0:0.864, 1:0.869, 2:0.874 |
| sst2 | R4 | none | 5e-05 | 1 | 0.524 | nan | 0:0.524 |
| sst2 | R5 | all | 5e-05 | 1 | 0.897 | nan | 0:0.897 |
| sst2 | R5 | none | 5e-05 | 1 | 0.615 | nan | 0:0.615 |
| sst2 | S1 | all | 5e-05 | 3 | 0.901 | 0.005 | 0:0.896, 1:0.905, 2:0.903 |
| sst2 | S1 | backbone | 5e-05 | 1 | 0.900 | nan | 0:0.900 |
| sst2 | S1 | memory | 0.0005 | 3 | 0.878 | 0.010 | 0:0.884, 1:0.867, 2:0.883 |
| sst2 | S1 | none | 5e-05 | 1 | 0.554 | nan | 0:0.554 |

### Key comparisons (Welch t-test over seeds)

| task | comparison | A mean (n) | B mean (n) | diff | p |
|---|---|---:|---:|---:|---:|
| boolq | memory vs dense FFN8, each at its default LR | 0.673 (3) | 0.686 (3) | -0.012 | 0.115 |
| boolq | memory vs dense FFN8, same LR 5e-4 | 0.673 (3) | 0.661 (3) | +0.012 | 0.300 |
| boolq | memory vs dense FFN8, same LR 5e-5 | 0.628 (3) | 0.686 (3) | -0.058 | 0.005 |
| boolq | memory: LR 5e-4 vs 5e-5 | 0.673 (3) | 0.628 (3) | +0.045 | 0.006 |
| boolq | FFN8: LR 5e-4 vs 5e-5 | 0.661 (3) | 0.686 (3) | -0.025 | 0.090 |
| boolq | R2 memory vs R2 FFN12 (similar size), LR 5e-4 | 0.673 (3) | 0.635 (3) | +0.038 | 0.012 |
| boolq | R2 memory vs R2 FFN12, LR 5e-5 | 0.628 (3) | 0.708 (3) | -0.080 | 0.002 |
| boolq | R2 memory-only vs full fine-tuning | 0.673 (3) | 0.700 (3) | -0.027 | 0.020 |
| boolq | full fine-tuning R2 vs R4 | 0.700 (3) | 0.696 (3) | +0.004 | 0.543 |
| boolq | memory-only S1 vs R2 | 0.671 (3) | 0.673 (3) | -0.002 | 0.734 |
| boolq | memory-only R1 vs R2 | 0.672 (3) | 0.673 (3) | -0.002 | 0.789 |
| ner | memory vs dense FFN8, each at its default LR | 0.679 (3) | 0.711 (3) | -0.033 | <0.001 |
| ner | memory vs dense FFN8, same LR 5e-4 | 0.679 (3) | 0.705 (3) | -0.026 | <0.001 |
| ner | memory vs dense FFN8, same LR 5e-5 | 0.658 (3) | 0.711 (3) | -0.054 | <0.001 |
| ner | memory: LR 5e-4 vs 5e-5 | 0.679 (3) | 0.658 (3) | +0.021 | <0.001 |
| ner | FFN8: LR 5e-4 vs 5e-5 | 0.705 (3) | 0.711 (3) | -0.007 | 0.031 |
| ner | R2 memory vs R2 FFN12 (similar size), LR 5e-4 | 0.679 (3) | 0.697 (3) | -0.018 | 0.004 |
| ner | R2 memory vs R2 FFN12, LR 5e-5 | 0.658 (3) | 0.716 (3) | -0.058 | <0.001 |
| ner | R2 memory-only vs full fine-tuning | 0.679 (3) | 0.715 (3) | -0.037 | <0.001 |
| ner | full fine-tuning R2 vs R4 | 0.715 (3) | 0.719 (3) | -0.004 | 0.152 |
| ner | memory-only S1 vs R2 | 0.676 (3) | 0.679 (3) | -0.003 | 0.426 |
| ner | memory-only R1 vs R2 | 0.682 (3) | 0.679 (3) | +0.004 | 0.127 |
| sst2 | memory vs dense FFN8, each at its default LR | 0.883 (3) | 0.899 (3) | -0.016 | 0.024 |
| sst2 | memory vs dense FFN8, same LR 5e-4 | 0.883 (3) | 0.869 (3) | +0.014 | 0.036 |
| sst2 | memory vs dense FFN8, same LR 5e-5 | 0.853 (3) | 0.899 (3) | -0.046 | 0.001 |
| sst2 | memory: LR 5e-4 vs 5e-5 | 0.883 (3) | 0.853 (3) | +0.030 | 0.006 |
| sst2 | FFN8: LR 5e-4 vs 5e-5 | 0.869 (3) | 0.899 (3) | -0.030 | 0.003 |
| sst2 | R2 memory vs R2 FFN12 (similar size), LR 5e-4 | 0.883 (3) | 0.827 (3) | +0.056 | <0.001 |
| sst2 | R2 memory vs R2 FFN12, LR 5e-5 | 0.853 (3) | 0.911 (3) | -0.058 | 0.002 |
| sst2 | R2 memory-only vs full fine-tuning | 0.883 (3) | 0.902 (3) | -0.019 | 0.016 |
| sst2 | full fine-tuning R2 vs R4 | 0.902 (3) | 0.903 (3) | -0.001 | 0.809 |
| sst2 | memory-only S1 vs R2 | 0.878 (3) | 0.883 (3) | -0.005 | 0.524 |
| sst2 | memory-only R1 vs R2 | 0.876 (3) | 0.883 (3) | -0.007 | 0.211 |

## C. CounterFact recall

| model | acc intact | acc zeroed | McNemar p (intact vs zeroed) | items lost / gained |
|---|---:|---:|---:|---:|
| R1 | 0.663 | 0.542 | <0.001 | 359 / 118 |
| R2 | 0.663 | 0.632 | <0.001 | 134 / 73 |
| R3 | 0.655 | 0.642 | 0.064 | 112 / 85 |
| R4 | 0.645 | — | — | — |
| R5 | 0.659 | — | — | — |
| A | 0.658 | 0.534 | <0.001 | 381 / 134 |
| S1 | 0.675 | 0.634 | <0.001 | 168 / 86 |

| comparison | acc diff | 95% CI | p |
|---|---:|---:|---:|
| R1 vs R4 | +1.75 | [+0.25, +3.35] | 0.026 |
| R2 vs R4 | +1.75 | [+0.15, +3.35] | 0.037 |
| R3 vs R4 | +1.00 | [-0.75, +2.65] | 0.244 |
| R5 vs R4 | +1.35 | [-0.25, +2.95] | 0.112 |
| A vs R4 | +1.25 | [-0.30, +2.80] | 0.117 |
| S1 vs R4 | +2.95 | [+1.35, +4.60] | <0.001 |
| R2 vs R5 | +0.40 | [-1.25, +2.10] | 0.635 |
| R3 vs R5 | -0.35 | [-2.00, +1.25] | 0.671 |
| R1 vs R2 | +0.00 | [-1.60, +1.60] | 1.000 |
| S1 vs R2 | +1.20 | [-0.45, +2.95] | 0.164 |
| A vs R2 | -0.50 | [-2.15, +1.20] | 0.564 |

