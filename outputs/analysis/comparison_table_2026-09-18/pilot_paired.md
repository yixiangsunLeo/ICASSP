Source: `outputs/analysis/snn_round/shu_pilot_lr1e4/per_model.csv` and `paired_tests.csv` next to it (analyze_snn_round.py).

| entry | n_subj | n_cells | splits | acc ± SD | balanced acc ± SD | macro-F1 | params | GPU-h |
|:---|---:|---:|:---|---:|---:|---:|---:|---:|
| `eegnex+zs` | 25 | 5 | [41] | 64.65 ± 10.46 | 64.65 ± 10.49 | 63.82 | 57106 | 0.59 |
| `seegnex@hplif+zs` | 25 | 5 | [41] | 64.17 ± 10.12 | 64.24 ± 10.09 | 62.97 | 57186 | 1.46 |
| `seegnex@s_hplif+zs` | 25 | 5 | [41] | 63.27 ± 11.01 | 63.29 ± 11.02 | 62.33 | 57186 | 3.44 |
| `eegnet_official+zs` | 25 | 5 | [41] | 61.50 ± 10.49 | 61.52 ± 10.46 | 61.17 | 2610 | 0.16 |
| `seegnet@plif+zs` | 25 | 5 | [41] | 60.56 ± 11.09 | 60.59 ± 11.05 | 59.85 | 2612 | 0.11 |

| family | baseline → candidate | Δacc (pp) | 95% CI | wins/ties/losses | p (Wilcoxon) |
|:---|:---|---:|:---|:---|---:|
| preregistered | `eegnex+zs` → `seegnex@s_hplif+zs` | -1.38 | [-2.96, +0.19] | 7/0/18 | 0.040 |
| exploratory | `eegnex+bench` → `eegnex+zs` | -0.83 | [-1.52, -0.15] | 7/2/16 | 0.021 |
| exploratory | `eegnex+zs` → `eegnex+zs+adabn` | -0.07 | [-0.79, +0.64] | 10/0/15 | 0.412 |
| exploratory | `eegnex+zs` → `seegnex@hplif+zs` | -0.48 | [-1.84, +0.88] | 10/1/14 | 0.162 |
| exploratory | `seegnex@s_hplif+zs` → `seegnex@s_hplif+zs+adabn` | +0.11 | [-0.55, +0.77] | 12/3/10 | 0.987 |
| exploratory | `seegnex@hplif+zs` → `seegnex@s_hplif+zs` | -0.90 | [-1.93, +0.13] | 8/1/16 | 0.070 |
| exploratory | `eegnet_official+zs` → `seegnet@plif+zs` | -0.94 | [-1.95, +0.07] | 10/0/15 | 0.078 |
