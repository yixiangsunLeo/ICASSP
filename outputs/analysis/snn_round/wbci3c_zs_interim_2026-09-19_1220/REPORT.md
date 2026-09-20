# Spiking round — wbci_shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

arms: role `zs` <- `zs=outputs/experiments/wbci_shu/snn_wbci3c_zs_lr1e4_v1`

> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 (PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending (STATUS §3.3); treat p-values here as approximate.

## Per-entry table (mean over subjects)

| entry                    | model           |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:-------------------------|:----------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| seegnex@s_hplif+zs+adabn | seegnex@s_hplif |            3 |         3 |     0.7130 |         0.0821 |     0.7133 | 0.8691 |         0.5888 |      59355 |             38.0000 |      5.3354 |
| seegnex@s_hplif+zs       | seegnex@s_hplif |            3 |         3 |     0.6985 |         0.0808 |     0.6958 | 0.8664 |         0.5888 |      59355 |             38.0000 |      5.3354 |
| eegnex+zs+adabn          | eegnex          |           11 |        11 |     0.6941 |         0.1287 |     0.6927 | 0.8510 |         0.6609 |      59275 |             59.0000 |      2.2725 |
| eegnex+zs                | eegnex          |           11 |        11 |     0.6941 |         0.1322 |     0.6912 | 0.8506 |         0.6609 |      59275 |             59.0000 |      2.2725 |
| seegnex@hplif+zs+adabn   | seegnex@hplif   |           11 |        11 |     0.6768 |         0.1262 |     0.6757 | 0.8372 |         0.6439 |      59355 |             43.0000 |      7.7496 |
| seegnex@hplif+zs         | seegnex@hplif   |           11 |        11 |     0.6734 |         0.1284 |     0.6700 | 0.8388 |         0.6439 |      59355 |             43.0000 |      7.7496 |
| eegnet_official+zs       | eegnet_official |           11 |        11 |     0.6605 |         0.1305 |     0.6585 | 0.8226 |         0.6446 |       3523 |            222.0000 |      1.4236 |
| eegnet_official+zs+adabn | eegnet_official |           11 |        11 |     0.6577 |         0.1298 |     0.6561 | 0.8239 |         0.6446 |       3523 |            222.0000 |      1.4236 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline           | candidate                | question                                                    |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |     d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:-------------------|:-------------------------|:------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|--------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | eegnex+zs          | seegnex@s_hplif+zs       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment |   3 |          0.7693 |           0.6985 | -0.0707 |         -0.2920 |          0.1505 | -0.7942 |      1 |      0 |        2 |    0.2850 |   1.0000 | False                 |
|  1 | exploratory   | eegnex+zs          | eegnex+zs+adabn          | AdaBN alone on the ANN                                      |  11 |          0.6941 |           0.6941 |  0.0000 |         -0.0069 |          0.0069 |  0.0001 |      6 |      0 |        5 |    0.7897 | nan      | <NA>                  |
|  2 | exploratory   | seegnex@s_hplif+zs | seegnex@s_hplif+zs+adabn | AdaBN alone on the SNN                                      |   3 |          0.6985 |           0.7130 |  0.0144 |         -0.0109 |          0.0397 |  1.4184 |      3 |      0 |        0 |    0.1088 | nan      | <NA>                  |
|  3 | exploratory   | eegnex+zs          | seegnex@hplif+zs         | direct (T=4) S-EEGNeX vs EEGNeX, no alignment               |  11 |          0.6941 |           0.6734 | -0.0207 |         -0.0476 |          0.0062 | -0.5179 |      4 |      0 |        7 |    0.1307 | nan      | <NA>                  |
|  4 | exploratory   | seegnex@hplif+zs   | seegnex@s_hplif+zs       | H2 (no EA): temporal-native vs direct (T=4) encoding        |   3 |          0.7381 |           0.6985 | -0.0396 |         -0.2699 |          0.1906 | -0.4275 |      1 |      1 |        1 |    0.6547 | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

_(no data yet)_

## Energy / firing rates (median over subjects)

| entry                    |   rate_lif1 |   rate_lif3 |      snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   n_subject_rows |
|:-------------------------|------------:|------------:|--------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|-----------------:|
| seegnex@s_hplif+zs+adabn |      0.1248 |      0.1708 | 34433512.0000 | 120020532.6389 |        266.4126 | 992897512.0000 |       4567.3286 |                     17.1438 |           3.0000 |
| seegnex@s_hplif+zs       |      0.1309 |      0.1743 | 34433512.0000 | 125834388.6111 |        271.6451 | 992897512.0000 |       4567.3286 |                     16.8136 |           3.0000 |
| seegnex@hplif+zs         |      0.1295 |      0.2000 | 48646048.0000 | 498696451.1111 |        672.5986 | 992897512.0000 |       4567.3286 |                      6.7906 |          11.0000 |
| seegnex@hplif+zs+adabn   |      0.1341 |      0.2047 | 48646048.0000 | 516463040.5556 |        688.5886 | 992897512.0000 |       4567.3286 |                      6.6329 |          11.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 9.620418848167555,
 "friedman_p": 0.08673330012329525,
 "k": 6,
 "N": 11,
 "avg_ranks": {
  "eegnex+zs+adabn": 2.3181818181818183,
  "eegnex+zs": 2.727272727272727,
  "seegnex@hplif+zs": 3.772727272727273,
  "seegnex@hplif+zs+adabn": 3.9545454545454546,
  "eegnet_official+zs": 4.045454545454546,
  "eegnet_official+zs+adabn": 4.181818181818182
 },
 "critical_difference": 2.2732785065123715,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
