# Spiking round — wbci_shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

arms: role `zs` <- `zs=outputs/experiments/wbci_shu/snn_wbci3c_zs_lr1e4_v1`

> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 (PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending (STATUS §3.3); treat p-values here as approximate.

## Per-entry table (mean over subjects)

| entry                    | model           |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:-------------------------|:----------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| atcnet+zs+adabn          | atcnet          |           11 |        11 |     0.7007 |         0.1343 |     0.6998 | 0.8522 |         0.6656 |     114719 |            169.0000 |     10.0665 |
| eegpt+zs                 | eegpt           |            3 |         3 |     0.6963 |         0.0817 |     0.6963 | 0.8581 |         0.5981 |   25388897 |             23.0000 |      3.8055 |
| eegnex+zs+adabn          | eegnex          |           11 |        11 |     0.6941 |         0.1287 |     0.6927 | 0.8510 |         0.6609 |      59275 |             59.0000 |      2.2725 |
| eegnex+zs                | eegnex          |           11 |        11 |     0.6941 |         0.1322 |     0.6912 | 0.8506 |         0.6609 |      59275 |             59.0000 |      2.2725 |
| eeg_deformer+zs+adabn    | eeg_deformer    |           11 |        11 |     0.6920 |         0.1511 |     0.6904 | 0.8496 |         0.6634 |    1612307 |             24.0000 |      3.0686 |
| tcformer+zs+adabn        | tcformer        |           11 |        11 |     0.6893 |         0.1284 |     0.6883 | 0.8521 |         0.6570 |     134056 |             87.0000 |      7.4593 |
| atcnet+zs                | atcnet          |           11 |        11 |     0.6886 |         0.1324 |     0.6803 | 0.8528 |         0.6656 |     114719 |            169.0000 |     10.0665 |
| tcformer+zs              | tcformer        |           11 |        11 |     0.6810 |         0.1265 |     0.6726 | 0.8495 |         0.6570 |     134056 |             87.0000 |      7.4593 |
| eeg_deformer+zs          | eeg_deformer    |           11 |        11 |     0.6810 |         0.1513 |     0.6766 | 0.8467 |         0.6634 |    1612307 |             24.0000 |      3.0686 |
| cbramod+zs               | cbramod         |           11 |        11 |     0.6808 |         0.1451 |     0.6779 | 0.8458 |         0.6615 |    5023003 |              6.0000 |      1.6960 |
| seegnex@hplif+zs+adabn   | seegnex@hplif   |           11 |        11 |     0.6768 |         0.1262 |     0.6757 | 0.8372 |         0.6439 |      59355 |             43.0000 |      7.7496 |
| seegnex@hplif+zs         | seegnex@hplif   |           11 |        11 |     0.6734 |         0.1284 |     0.6700 | 0.8388 |         0.6439 |      59355 |             43.0000 |      7.7496 |
| seegnet@plif+zs+adabn    | seegnet@plif    |           11 |        11 |     0.6730 |         0.1246 |     0.6712 | 0.8316 |         0.6387 |       3525 |            123.0000 |      1.3571 |
| labram+zs                | labram          |           11 |        11 |     0.6719 |         0.1347 |     0.6692 | 0.8391 |         0.6574 |    5964539 |             13.0000 |      2.7149 |
| seegnet@plif+zs          | seegnet@plif    |           11 |        11 |     0.6662 |         0.1243 |     0.6624 | 0.8322 |         0.6387 |       3525 |            123.0000 |      1.3571 |
| eegnet_official+zs       | eegnet_official |           11 |        11 |     0.6605 |         0.1305 |     0.6585 | 0.8226 |         0.6446 |       3523 |            222.0000 |      1.4236 |
| eegnet_official+zs+adabn | eegnet_official |           11 |        11 |     0.6577 |         0.1298 |     0.6561 | 0.8239 |         0.6446 |       3523 |            222.0000 |      1.4236 |
| seegnex@s_hplif+zs+adabn | seegnex@s_hplif |           11 |        11 |     0.6487 |         0.1191 |     0.6476 | 0.8155 |         0.6062 |      59355 |             36.0000 |     22.2343 |
| seegnex@s_hplif+zs       | seegnex@s_hplif |           11 |        11 |     0.6359 |         0.1207 |     0.6278 | 0.8162 |         0.6062 |      59355 |             36.0000 |     22.2343 |
| fbcnet+zs+adabn          | fbcnet          |           11 |        11 |     0.5396 |         0.0870 |     0.5356 | 0.7295 |         0.5213 |      21864 |             74.0000 |      1.3344 |
| fbcnet+zs                | fbcnet          |           11 |        11 |     0.5291 |         0.0830 |     0.5135 | 0.7275 |         0.5213 |      21864 |             74.0000 |      1.3344 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline           | candidate                | question                                                    |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |     d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:-------------------|:-------------------------|:------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|--------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | eegnex+zs          | seegnex@s_hplif+zs       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment |  11 |          0.6941 |           0.6359 | -0.0582 |         -0.0950 |         -0.0214 | -1.0619 |      2 |      0 |        9 |    0.0128 |   0.1023 | False                 |
|  1 | exploratory   | seegnex@s_hplif+zs | seegnex@s_hplif+zs+adabn | AdaBN alone on the SNN                                      |  11 |          0.6359 |           0.6487 |  0.0127 |          0.0065 |          0.0190 |  1.3682 |      9 |      0 |        2 |    0.0076 | nan      | <NA>                  |
|  2 | exploratory   | seegnex@hplif+zs   | seegnex@s_hplif+zs       | H2 (no EA): temporal-native vs direct (T=4) encoding        |  11 |          0.6734 |           0.6359 | -0.0375 |         -0.0677 |         -0.0073 | -0.8332 |      1 |      1 |        9 |    0.0165 | nan      | <NA>                  |
|  3 | exploratory   | eegnex+zs          | eegnex+zs+adabn          | AdaBN alone on the ANN                                      |  11 |          0.6941 |           0.6941 |  0.0000 |         -0.0069 |          0.0069 |  0.0001 |      6 |      0 |        5 |    0.7897 | nan      | <NA>                  |
|  4 | exploratory   | eegnex+zs          | seegnex@hplif+zs         | direct (T=4) S-EEGNeX vs EEGNeX, no alignment               |  11 |          0.6941 |           0.6734 | -0.0207 |         -0.0476 |          0.0062 | -0.5179 |      4 |      0 |        7 |    0.1307 | nan      | <NA>                  |
|  5 | exploratory   | eegnet_official+zs | seegnet@plif+zs          | S-EEGNet vs EEGNet, no alignment                            |  11 |          0.6605 |           0.6662 |  0.0058 |         -0.0178 |          0.0293 |  0.1642 |      5 |      1 |        5 |    0.9594 | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

_(no data yet)_

## Energy / firing rates (median over subjects)

| entry                    |   rate_lif1 |      snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   rate_lif3 |   n_subject_rows |
|:-------------------------|------------:|--------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|------------:|-----------------:|
| seegnex@s_hplif+zs       |      0.1299 | 34433512.0000 | 125011004.5833 |        270.9041 | 992897512.0000 |       4567.3286 |                     16.8596 |      0.1801 |          11.0000 |
| seegnex@s_hplif+zs+adabn |      0.1317 | 34433512.0000 | 126564042.3611 |        272.3018 | 992897512.0000 |       4567.3286 |                     16.7730 |      0.1783 |          11.0000 |
| seegnex@hplif+zs         |      0.1295 | 48646048.0000 | 498696451.1111 |        672.5986 | 992897512.0000 |       4567.3286 |                      6.7906 |      0.2000 |          11.0000 |
| seegnex@hplif+zs+adabn   |      0.1341 | 48646048.0000 | 516463040.5556 |        688.5886 | 992897512.0000 |       4567.3286 |                      6.6329 |      0.2047 |          11.0000 |
| seegnet@plif+zs          |      0.2283 | 30885952.0000 |     58452.9427 |        142.1280 |  30753488.0000 |        141.4660 |                      0.9953 |    nan      |          11.0000 |
| seegnet@plif+zs+adabn    |      0.2394 | 30885952.0000 |     61295.8030 |        142.1305 |  30753488.0000 |        141.4660 |                      0.9953 |    nan      |          11.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 65.90258939580761,
 "friedman_p": 4.3525685147363563e-07,
 "k": 20,
 "N": 11,
 "avg_ranks": {
  "atcnet+zs+adabn": 5.545454545454546,
  "eegnex+zs+adabn": 6.863636363636363,
  "eegnex+zs": 7.454545454545454,
  "atcnet+zs": 7.7727272727272725,
  "tcformer+zs+adabn": 8.0,
  "eeg_deformer+zs+adabn": 8.636363636363637,
  "tcformer+zs": 8.818181818181818,
  "cbramod+zs": 9.409090909090908,
  "seegnet@plif+zs+adabn": 9.454545454545455,
  "seegnex@hplif+zs+adabn": 9.636363636363637,
  "seegnex@hplif+zs": 9.681818181818182,
  "eeg_deformer+zs": 10.181818181818182,
  "labram+zs": 10.636363636363637,
  "eegnet_official+zs": 11.318181818181818,
  "seegnet@plif+zs": 11.363636363636363,
  "eegnet_official+zs+adabn": 11.5,
  "seegnex@s_hplif+zs+adabn": 12.772727272727273,
  "seegnex@s_hplif+zs": 15.227272727272727,
  "fbcnet+zs+adabn": 17.818181818181817,
  "fbcnet+zs": 17.90909090909091
 },
 "critical_difference": 8.939675913986594,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
