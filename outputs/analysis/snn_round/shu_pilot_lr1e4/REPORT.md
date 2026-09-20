# Spiking round — shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

arms: role `zs` <- `zs=outputs/experiments/shu/snn_shu_zs_lr1e4_v1`; bench <- `results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1`

> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 (PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending (STATUS §3.3); treat p-values here as approximate.

## Per-entry table (mean over subjects)

| entry                     | model               |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:--------------------------|:--------------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| eegnex+bench              | eegnex              |           25 |        10 |     0.6549 |         0.0995 |     0.6478 | 0.7053 |         0.6614 |      57106 |             73.0000 |      1.6142 |
| eegnex+zs                 | eegnex              |           25 |         5 |     0.6465 |         0.1046 |     0.6382 | 0.6956 |         0.6808 |      57106 |             72.0000 |      0.5947 |
| eegnex+zs+adabn           | eegnex              |           25 |         5 |     0.6458 |         0.1029 |     0.6450 | 0.6936 |         0.6808 |      57106 |             72.0000 |      0.5947 |
| seegnex@hplif+zs+adabn    | seegnex@hplif       |           25 |         5 |     0.6431 |         0.1020 |     0.6423 | 0.6884 |         0.6666 |      57186 |             20.0000 |      1.4573 |
| seegnex@hplif+zs          | seegnex@hplif       |           25 |         5 |     0.6417 |         0.1012 |     0.6297 | 0.6898 |         0.6666 |      57186 |             20.0000 |      1.4573 |
| seegnex@s_hplif+zs+adabn  | seegnex@s_hplif     |           25 |         5 |     0.6338 |         0.1064 |     0.6332 | 0.6789 |         0.6607 |      57186 |             13.0000 |      3.4437 |
| seegnex@s_hplif+zs        | seegnex@s_hplif     |           25 |         5 |     0.6327 |         0.1101 |     0.6233 | 0.6803 |         0.6607 |      57186 |             13.0000 |      3.4437 |
| atcnet+bench              | atcnet              |           25 |        10 |     0.6196 |         0.1123 |     0.6167 | 0.6628 |         0.6195 |     113722 |             55.5000 |      5.1860 |
| s4erp_dw@f16d8+bench      | s4erp_dw@f16d8      |           25 |        10 |     0.6168 |         0.0917 |     0.6142 | 0.6599 |         0.6190 |     421794 |             10.5000 |      0.5732 |
| eegnet_official+zs+adabn  | eegnet_official     |           25 |         5 |     0.6166 |         0.1051 |     0.6149 | 0.6571 |         0.6327 |       2610 |             45.0000 |      0.1631 |
| atcnet@tcn8+bench         | atcnet@tcn8         |           25 |        10 |     0.6157 |         0.1103 |     0.6127 | 0.6560 |         0.6190 |      40402 |             54.0000 |      2.2831 |
| eegnet_official+bench     | eegnet_official     |           25 |        10 |     0.6152 |         0.1072 |     0.6126 | 0.6557 |         0.6160 |       2610 |             41.5000 |      0.4409 |
| eegnet_official+zs        | eegnet_official     |           25 |         5 |     0.6150 |         0.1049 |     0.6117 | 0.6570 |         0.6327 |       2610 |             45.0000 |      0.1631 |
| atcnet@noattn+bench       | atcnet@noattn       |           25 |        10 |     0.6143 |         0.1094 |     0.6111 | 0.6569 |         0.6228 |     102762 |             34.0000 |      3.9707 |
| eeg_deformer+bench        | eeg_deformer        |           25 |        10 |     0.6143 |         0.0954 |     0.6081 | 0.6605 |         0.6292 |    1503570 |              7.5000 |      0.8386 |
| atcnet@nwin1+bench        | atcnet@nwin1        |           25 |        10 |     0.6135 |         0.1119 |     0.6103 | 0.6559 |         0.6140 |      37618 |             60.0000 |      1.2841 |
| s4erp+bench               | s4erp               |           25 |        10 |     0.6127 |         0.0952 |     0.6085 | 0.6589 |         0.6222 |     944898 |              7.0000 |      1.3191 |
| deepconvnet+bench         | deepconvnet         |           25 |        10 |     0.6096 |         0.1031 |     0.5977 | 0.6518 |         0.6070 |     286302 |              6.5000 |      0.3474 |
| dualcd_s4_timepatch+bench | dualcd_s4_timepatch |           25 |        10 |     0.6083 |         0.1064 |     0.6038 | 0.6483 |         0.6162 |    3628418 |              7.0000 |      7.7540 |
| seegnet@plif+zs+adabn     | seegnet@plif        |           25 |         5 |     0.6071 |         0.1130 |     0.6058 | 0.6503 |         0.6228 |       2612 |             20.0000 |      0.1102 |
| seegnet@plif+zs           | seegnet@plif        |           25 |         5 |     0.6056 |         0.1109 |     0.5985 | 0.6495 |         0.6228 |       2612 |             20.0000 |      0.1102 |
| dualcd_transformer+bench  | dualcd_transformer  |           25 |        10 |     0.6052 |         0.0877 |     0.6001 | 0.6464 |         0.6126 |   67048066 |              8.5000 |     23.1936 |
| s4erp_dw@f64d2+bench      | s4erp_dw@f64d2      |           25 |        10 |     0.6052 |         0.0913 |     0.6022 | 0.6476 |         0.6142 |     423042 |              2.5000 |      0.5443 |
| dualcd_s4_flatten+bench   | dualcd_s4_flatten   |           25 |        10 |     0.6039 |         0.0827 |     0.6023 | 0.6424 |         0.6162 |   66009218 |              5.5000 |     18.3543 |
| atcnet_s4@noattn+bench    | atcnet_s4@noattn    |           25 |        10 |     0.6014 |         0.0983 |     0.6001 | 0.6347 |         0.6049 |      24810 |             40.0000 |      0.5238 |
| atcnet_s4+bench           | atcnet_s4           |           25 |        10 |     0.5985 |         0.1006 |     0.5953 | 0.6349 |         0.6052 |      27002 |             50.5000 |      0.6103 |
| s4erp_dw@f16d2p+bench     | s4erp_dw@f16d2p     |           25 |        10 |     0.5974 |         0.0787 |     0.5920 | 0.6402 |         0.6158 |     406146 |              2.0000 |      0.4256 |
| fbcnet+bench              | fbcnet              |           25 |        10 |     0.5924 |         0.0667 |     0.5794 | 0.6333 |         0.5928 |      13223 |             87.0000 |      0.8103 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline           | candidate                | question                                                       |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |     d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:-------------------|:-------------------------|:---------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|--------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | eegnex+zs          | seegnex@s_hplif+zs       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment    |  25 |          0.6465 |           0.6327 | -0.0138 |         -0.0296 |          0.0019 | -0.3620 |      7 |      0 |       18 |    0.0396 |   0.3164 | False                 |
|  1 | exploratory   | eegnex+bench       | eegnex+zs                | sanity: re-run EEGNeX vs benchmark EEGNeX (same recipe/splits) |  25 |          0.6549 |           0.6465 | -0.0083 |         -0.0152 |         -0.0015 | -0.5030 |      7 |      2 |       16 |    0.0208 | nan      | <NA>                  |
|  2 | exploratory   | eegnex+zs          | eegnex+zs+adabn          | AdaBN alone on the ANN                                         |  25 |          0.6465 |           0.6458 | -0.0007 |         -0.0079 |          0.0064 | -0.0419 |     10 |      0 |       15 |    0.4118 | nan      | <NA>                  |
|  3 | exploratory   | eegnex+zs          | seegnex@hplif+zs         | direct (T=4) S-EEGNeX vs EEGNeX, no alignment                  |  25 |          0.6465 |           0.6417 | -0.0048 |         -0.0184 |          0.0088 | -0.1453 |     10 |      1 |       14 |    0.1615 | nan      | <NA>                  |
|  4 | exploratory   | seegnex@s_hplif+zs | seegnex@s_hplif+zs+adabn | AdaBN alone on the SNN                                         |  25 |          0.6327 |           0.6338 |  0.0011 |         -0.0055 |          0.0077 |  0.0687 |     12 |      3 |       10 |    0.9870 | nan      | <NA>                  |
|  5 | exploratory   | seegnex@hplif+zs   | seegnex@s_hplif+zs       | H2 (no EA): temporal-native vs direct (T=4) encoding           |  25 |          0.6417 |           0.6327 | -0.0090 |         -0.0193 |          0.0013 | -0.3625 |      8 |      1 |       16 |    0.0696 | nan      | <NA>                  |
|  6 | exploratory   | eegnet_official+zs | seegnet@plif+zs          | S-EEGNet vs EEGNet, no alignment                               |  25 |          0.6150 |           0.6056 | -0.0094 |         -0.0195 |          0.0007 | -0.3861 |     10 |      0 |       15 |    0.0780 | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

_(no data yet)_

## Energy / firing rates (median over subjects)

| entry                    |   rate_lif1 |      snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   rate_lif3 |   n_subject_rows |
|:-------------------------|------------:|--------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|------------:|-----------------:|
| seegnex@s_hplif+zs+adabn |      0.1208 | 19457008.0000 |  64555473.4887 |        147.6022 | 551937008.0000 |       2538.9102 |                     17.2010 |      0.1514 |          25.0000 |
| seegnex@s_hplif+zs       |      0.1207 | 19457008.0000 |  64585362.7277 |        147.6291 | 551937008.0000 |       2538.9102 |                     17.1979 |      0.1530 |          25.0000 |
| seegnex@hplif+zs         |      0.1171 | 28676032.0000 | 251528039.1121 |        358.2850 | 551937008.0000 |       2538.9102 |                      7.0863 |      0.1870 |          25.0000 |
| seegnex@hplif+zs+adabn   |      0.1193 | 28676032.0000 | 256526966.7031 |        362.7840 | 551937008.0000 |       2538.9102 |                      6.9984 |      0.1891 |          25.0000 |
| seegnet@plif+zs          |      0.2528 | 17155968.0000 |     64721.6270 |         78.9757 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |          25.0000 |
| seegnet@plif+zs+adabn    |      0.2608 | 17155968.0000 |     66762.3414 |         78.9775 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |          25.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 132.01147862695007,
 "friedman_p": 8.6115095574784e-16,
 "k": 28,
 "N": 25,
 "avg_ranks": {
  "eegnex+bench": 6.88,
  "eegnex+zs": 8.32,
  "seegnex@hplif+zs": 8.34,
  "seegnex@hplif+zs+adabn": 8.34,
  "eegnex+zs+adabn": 9.26,
  "seegnex@s_hplif+zs+adabn": 10.38,
  "seegnex@s_hplif+zs": 11.32,
  "eeg_deformer+bench": 13.96,
  "atcnet+bench": 13.98,
  "eegnet_official+zs+adabn": 14.48,
  "s4erp_dw@f16d8+bench": 14.7,
  "eegnet_official+zs": 14.86,
  "atcnet@noattn+bench": 14.94,
  "s4erp+bench": 14.96,
  "atcnet@tcn8+bench": 15.0,
  "eegnet_official+bench": 15.58,
  "atcnet@nwin1+bench": 15.8,
  "dualcd_s4_flatten+bench": 16.12,
  "dualcd_transformer+bench": 16.36,
  "dualcd_s4_timepatch+bench": 16.86,
  "s4erp_dw@f64d2+bench": 17.0,
  "deepconvnet+bench": 17.42,
  "fbcnet+bench": 17.8,
  "seegnet@plif+zs+adabn": 18.04,
  "s4erp_dw@f16d2p+bench": 18.1,
  "atcnet_s4@noattn+bench": 18.74,
  "seegnet@plif+zs": 18.84,
  "atcnet_s4+bench": 19.62
 },
 "critical_difference": 8.642363345190644,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
