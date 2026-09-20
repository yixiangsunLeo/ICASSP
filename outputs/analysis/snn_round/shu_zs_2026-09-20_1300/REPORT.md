# Spiking round — shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

arms: role `zs` <- `zs=outputs/experiments/shu/snn_shu_zs_lr1e4_v1`; bench <- `results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1`

> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 (PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending (STATUS §3.3); treat p-values here as approximate.

## Per-entry table (mean over subjects)

| entry                     | model               |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:--------------------------|:--------------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| eegnex+bench              | eegnex              |           25 |        10 |     0.6549 |         0.0995 |     0.6478 | 0.7053 |         0.6614 |      57106 |             73.0000 |      1.6142 |
| eegnex+zs                 | eegnex              |           25 |        10 |     0.6549 |         0.0995 |     0.6478 | 0.7053 |         0.6614 |      57106 |             73.0000 |      1.4118 |
| eegnex+zs+adabn           | eegnex              |           25 |        10 |     0.6512 |         0.0991 |     0.6505 | 0.7034 |         0.6614 |      57106 |             73.0000 |      1.4118 |
| seegnex@s_plif+zs+adabn   | seegnex@s_plif      |           15 |         3 |     0.6474 |         0.1203 |     0.6466 | 0.6975 |         0.6796 |      57109 |             25.0000 |      2.3679 |
| cbramod+zs                | cbramod             |           25 |        10 |     0.6458 |         0.0997 |     0.6420 | 0.6950 |         0.6564 |    4935002 |             10.5000 |      0.5676 |
| seegnex@hplif+zs          | seegnex@hplif       |           25 |        10 |     0.6453 |         0.0989 |     0.6349 | 0.6950 |         0.6471 |      57186 |             22.0000 |      2.9190 |
| seegnex@hplif+zs+adabn    | seegnex@hplif       |           25 |        10 |     0.6453 |         0.1021 |     0.6447 | 0.6924 |         0.6471 |      57186 |             22.0000 |      2.9190 |
| seegnex@s_lif+zs+adabn    | seegnex@s_lif       |           25 |        10 |     0.6442 |         0.0989 |     0.6437 | 0.6895 |         0.6418 |      57106 |             29.0000 |      9.8689 |
| seegnex@s_plif+zs         | seegnex@s_plif      |           15 |         3 |     0.6436 |         0.1231 |     0.6239 | 0.6988 |         0.6796 |      57109 |             25.0000 |      2.3679 |
| seegnex@s_lif+zs          | seegnex@s_lif       |           25 |        10 |     0.6423 |         0.0998 |     0.6352 | 0.6913 |         0.6418 |      57106 |             29.0000 |      9.8689 |
| labram+zs                 | labram              |           25 |        10 |     0.6421 |         0.1014 |     0.6385 | 0.6858 |         0.6459 |    5876338 |             16.0000 |      0.7176 |
| seegnex@s_hplif+zs+adabn  | seegnex@s_hplif     |           25 |        10 |     0.6392 |         0.1047 |     0.6387 | 0.6847 |         0.6379 |      57186 |             15.5000 |      7.7273 |
| seegnex@s_hplif+zs        | seegnex@s_hplif     |           25 |        10 |     0.6360 |         0.1062 |     0.6272 | 0.6861 |         0.6379 |      57186 |             15.5000 |      7.7273 |
| atcnet+bench              | atcnet              |           25 |        10 |     0.6196 |         0.1123 |     0.6167 | 0.6628 |         0.6195 |     113722 |             55.5000 |      5.1860 |
| tcformer+zs               | tcformer            |           25 |        10 |     0.6172 |         0.0984 |     0.6115 | 0.6672 |         0.6261 |     128996 |             15.5000 |      3.5280 |
| tcformer+zs+adabn         | tcformer            |           25 |        10 |     0.6169 |         0.0934 |     0.6156 | 0.6618 |         0.6261 |     128996 |             15.5000 |      3.5280 |
| s4erp_dw@f16d8+bench      | s4erp_dw@f16d8      |           25 |        10 |     0.6168 |         0.0917 |     0.6142 | 0.6599 |         0.6190 |     421794 |             10.5000 |      0.5732 |
| atcnet@tcn8+bench         | atcnet@tcn8         |           25 |        10 |     0.6157 |         0.1103 |     0.6127 | 0.6560 |         0.6190 |      40402 |             54.0000 |      2.2831 |
| eegnet_official+bench     | eegnet_official     |           25 |        10 |     0.6152 |         0.1072 |     0.6126 | 0.6557 |         0.6160 |       2610 |             41.5000 |      0.4409 |
| eegnet_official+zs        | eegnet_official     |           25 |        10 |     0.6152 |         0.1072 |     0.6126 | 0.6557 |         0.6160 |       2610 |             41.5000 |      0.2976 |
| eegnet_official+zs+adabn  | eegnet_official     |           25 |        10 |     0.6152 |         0.1083 |     0.6136 | 0.6552 |         0.6160 |       2610 |             41.5000 |      0.2976 |
| atcnet@noattn+bench       | atcnet@noattn       |           25 |        10 |     0.6143 |         0.1094 |     0.6111 | 0.6569 |         0.6228 |     102762 |             34.0000 |      3.9707 |
| eeg_deformer+bench        | eeg_deformer        |           25 |        10 |     0.6143 |         0.0954 |     0.6081 | 0.6605 |         0.6292 |    1503570 |              7.5000 |      0.8386 |
| atcnet@nwin1+bench        | atcnet@nwin1        |           25 |        10 |     0.6135 |         0.1119 |     0.6103 | 0.6559 |         0.6140 |      37618 |             60.0000 |      1.2841 |
| s4erp+bench               | s4erp               |           25 |        10 |     0.6127 |         0.0952 |     0.6085 | 0.6589 |         0.6222 |     944898 |              7.0000 |      1.3191 |
| deepconvnet+bench         | deepconvnet         |           25 |        10 |     0.6096 |         0.1031 |     0.5977 | 0.6518 |         0.6070 |     286302 |              6.5000 |      0.3474 |
| dualcd_s4_timepatch+bench | dualcd_s4_timepatch |           25 |        10 |     0.6083 |         0.1064 |     0.6038 | 0.6483 |         0.6162 |    3628418 |              7.0000 |      7.7540 |
| dualcd_transformer+bench  | dualcd_transformer  |           25 |        10 |     0.6052 |         0.0877 |     0.6001 | 0.6464 |         0.6126 |   67048066 |              8.5000 |     23.1936 |
| s4erp_dw@f64d2+bench      | s4erp_dw@f64d2      |           25 |        10 |     0.6052 |         0.0913 |     0.6022 | 0.6476 |         0.6142 |     423042 |              2.5000 |      0.5443 |
| seegnet@plif+zs+adabn     | seegnet@plif        |           25 |        10 |     0.6047 |         0.1115 |     0.6034 | 0.6465 |         0.6073 |       2612 |             19.0000 |      0.1947 |
| seegnet@plif+zs           | seegnet@plif        |           25 |        10 |     0.6045 |         0.1098 |     0.5988 | 0.6458 |         0.6073 |       2612 |             19.0000 |      0.1947 |
| dualcd_s4_flatten+bench   | dualcd_s4_flatten   |           25 |        10 |     0.6039 |         0.0827 |     0.6023 | 0.6424 |         0.6162 |   66009218 |              5.5000 |     18.3543 |
| atcnet_s4@noattn+bench    | atcnet_s4@noattn    |           25 |        10 |     0.6014 |         0.0983 |     0.6001 | 0.6347 |         0.6049 |      24810 |             40.0000 |      0.5238 |
| atcnet_s4+bench           | atcnet_s4           |           25 |        10 |     0.5985 |         0.1006 |     0.5953 | 0.6349 |         0.6052 |      27002 |             50.5000 |      0.6103 |
| s4erp_dw@f16d2p+bench     | s4erp_dw@f16d2p     |           25 |        10 |     0.5974 |         0.0787 |     0.5920 | 0.6402 |         0.6158 |     406146 |              2.0000 |      0.4256 |
| fbcnet+bench              | fbcnet              |           25 |        10 |     0.5924 |         0.0667 |     0.5794 | 0.6333 |         0.5928 |      13223 |             87.0000 |      0.8103 |
| eegpt+zs                  | eegpt               |           25 |        10 |     0.5671 |         0.0573 |     0.5655 | 0.5970 |         0.5681 |   25353636 |             32.5000 |      4.1224 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline           | candidate                | question                                                       |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |      d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:-------------------|:-------------------------|:---------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|---------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | eegnex+zs          | seegnex@s_hplif+zs       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment    |  25 |          0.6549 |           0.6360 | -0.0188 |         -0.0355 |         -0.0022 |  -0.4662 |      8 |      0 |       17 |    0.0230 |   0.1839 | False                 |
|  1 | exploratory   | eegnet_official+zs | seegnet@plif+zs          | S-EEGNet vs EEGNet, no alignment                               |  25 |          0.6152 |           0.6045 | -0.0107 |         -0.0184 |         -0.0030 |  -0.5751 |      6 |      1 |       18 |    0.0177 | nan      | <NA>                  |
|  2 | exploratory   | eegnex+zs          | seegnex@hplif+zs         | direct (T=4) S-EEGNeX vs EEGNeX, no alignment                  |  25 |          0.6549 |           0.6453 | -0.0095 |         -0.0224 |          0.0034 |  -0.3048 |      8 |      0 |       17 |    0.0324 | nan      | <NA>                  |
|  3 | exploratory   | eegnex+zs          | eegnex+zs+adabn          | AdaBN alone on the ANN                                         |  25 |          0.6549 |           0.6512 | -0.0036 |         -0.0090 |          0.0017 |  -0.2812 |      8 |      0 |       17 |    0.0464 | nan      | <NA>                  |
|  4 | exploratory   | seegnex@hplif+zs   | seegnex@s_hplif+zs       | H2 (no EA): temporal-native vs direct (T=4) encoding           |  25 |          0.6453 |           0.6360 | -0.0093 |         -0.0187 |          0.0001 |  -0.4087 |      8 |      0 |       17 |    0.0673 | nan      | <NA>                  |
|  5 | exploratory   | seegnex@s_hplif+zs | seegnex@s_hplif+zs+adabn | AdaBN alone on the SNN                                         |  25 |          0.6360 |           0.6392 |  0.0032 |         -0.0012 |          0.0075 |   0.2985 |     15 |      1 |        9 |    0.1531 | nan      | <NA>                  |
|  6 | exploratory   | seegnex@s_plif+zs  | seegnex@s_hplif+zs       | H3 (no EA): heterogeneous vs per-layer tau                     |  15 |          0.6436 |           0.6429 | -0.0007 |         -0.0164 |          0.0150 |  -0.0247 |      9 |      1 |        5 |    0.7299 | nan      | <NA>                  |
|  7 | exploratory   | seegnex@s_lif+zs   | seegnex@s_plif+zs        | H3 (no EA): PLIF vs LIF                                        |  15 |          0.6490 |           0.6436 | -0.0053 |         -0.0161 |          0.0054 |  -0.2745 |      6 |      0 |        9 |    0.2560 | nan      | <NA>                  |
|  8 | exploratory   | eegnex+bench       | eegnex+zs                | sanity: re-run EEGNeX vs benchmark EEGNeX (same recipe/splits) |  25 |          0.6549 |           0.6549 |  0.0000 |          0.0000 |          0.0000 | nan      |      0 |     25 |        0 |  nan      | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

_(no data yet)_

## Energy / firing rates (median over subjects)

| entry                    |   rate_lif1 |      snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   rate_lif3 |   n_subject_rows |
|:-------------------------|------------:|--------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|------------:|-----------------:|
| seegnex@s_hplif+zs       |      0.1200 | 19457008.0000 |  64188067.3009 |        147.2715 | 551937008.0000 |       2538.9102 |                     17.2397 |      0.1552 |          50.0000 |
| seegnex@s_hplif+zs+adabn |      0.1207 | 19457008.0000 |  64435444.0947 |        147.4941 | 551937008.0000 |       2538.9102 |                     17.2136 |      0.1549 |          50.0000 |
| seegnex@s_plif+zs        |      0.1401 | 19457008.0000 |  75074821.5768 |        157.0696 | 551937008.0000 |       2538.9102 |                     16.1642 |      0.2320 |          15.0000 |
| seegnex@s_lif+zs         |      0.1405 | 19457008.0000 |  75450655.0755 |        157.4078 | 551937008.0000 |       2538.9102 |                     16.1295 |      0.2339 |          50.0000 |
| seegnex@s_lif+zs+adabn   |      0.1409 | 19457008.0000 |  75581951.5132 |        157.5260 | 551937008.0000 |       2538.9102 |                     16.1174 |      0.2321 |          50.0000 |
| seegnex@s_plif+zs+adabn  |      0.1414 | 19457008.0000 |  75651893.9162 |        157.5889 | 551937008.0000 |       2538.9102 |                     16.1110 |      0.2344 |          15.0000 |
| seegnex@hplif+zs         |      0.1179 | 28676032.0000 | 253400183.3713 |        359.9699 | 551937008.0000 |       2538.9102 |                      7.0531 |      0.1889 |          50.0000 |
| seegnex@hplif+zs+adabn   |      0.1197 | 28676032.0000 | 257221317.4391 |        363.4089 | 551937008.0000 |       2538.9102 |                      6.9864 |      0.1888 |          50.0000 |
| seegnet@plif+zs          |      0.2567 | 17155968.0000 |     65704.9617 |         78.9766 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |          50.0000 |
| seegnet@plif+zs+adabn    |      0.2622 | 17155968.0000 |     67132.9990 |         78.9779 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |          50.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 210.6093048038365,
 "friedman_p": 2.3770426847159046e-27,
 "k": 35,
 "N": 25,
 "avg_ranks": {
  "eegnex+bench": 8.84,
  "eegnex+zs": 8.84,
  "eegnex+zs+adabn": 10.68,
  "seegnex@s_lif+zs+adabn": 10.72,
  "seegnex@hplif+zs": 10.78,
  "seegnex@s_lif+zs": 10.88,
  "seegnex@hplif+zs+adabn": 11.2,
  "cbramod+zs": 11.44,
  "seegnex@s_hplif+zs+adabn": 11.94,
  "seegnex@s_hplif+zs": 13.48,
  "labram+zs": 13.6,
  "atcnet+bench": 17.64,
  "eeg_deformer+bench": 18.86,
  "s4erp+bench": 19.16,
  "atcnet@noattn+bench": 19.22,
  "eegnet_official+zs+adabn": 19.26,
  "atcnet@tcn8+bench": 19.44,
  "s4erp_dw@f16d8+bench": 19.54,
  "eegnet_official+bench": 19.56,
  "eegnet_official+zs": 19.56,
  "tcformer+zs+adabn": 19.58,
  "atcnet@nwin1+bench": 19.6,
  "tcformer+zs": 20.66,
  "dualcd_s4_flatten+bench": 21.22,
  "dualcd_s4_timepatch+bench": 21.46,
  "deepconvnet+bench": 21.66,
  "dualcd_transformer+bench": 21.72,
  "s4erp_dw@f64d2+bench": 22.12,
  "fbcnet+bench": 23.32,
  "s4erp_dw@f16d2p+bench": 23.34,
  "atcnet_s4@noattn+bench": 23.42,
  "seegnet@plif+zs+adabn": 23.68,
  "seegnet@plif+zs": 23.82,
  "eegpt+zs": 24.52,
  "atcnet_s4+bench": 25.24
 },
 "critical_difference": 11.082076092258477,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
