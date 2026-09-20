# Spiking round — wbci_shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

arms: role `zs` <- `zs=outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1`; bench <- `results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1`

> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 (PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending (STATUS §3.3); treat p-values here as approximate.

## Per-entry table (mean over subjects)

| entry                     | model               |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:--------------------------|:--------------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| atcnet@noattn+bench       | atcnet@noattn       |           51 |        10 |     0.8356 |         0.1179 |     0.8354 | 0.9014 |         0.8536 |     103594 |            118.0000 |     18.5434 |
| eegnex+zs+adabn           | eegnex              |           51 |        10 |     0.8353 |         0.1144 |     0.8351 | 0.9012 |         0.8462 |      58770 |             25.0000 |      3.5474 |
| eegnex+zs                 | eegnex              |           51 |        10 |     0.8335 |         0.1154 |     0.8333 | 0.9008 |         0.8462 |      58770 |             25.0000 |      3.5474 |
| eegnex+bench              | eegnex              |           51 |        10 |     0.8335 |         0.1154 |     0.8333 | 0.9008 |         0.8462 |      58770 |             25.0000 |      4.0450 |
| atcnet+bench              | atcnet              |           51 |        10 |     0.8328 |         0.1200 |     0.8323 | 0.8993 |         0.8497 |     114554 |             72.0000 |     17.1494 |
| atcnet@nwin1+bench        | atcnet@nwin1        |           51 |        10 |     0.8316 |         0.1182 |     0.8313 | 0.8989 |         0.8462 |      38450 |             87.0000 |      5.6338 |
| atcnet@tcn8+bench         | atcnet@tcn8         |           51 |        10 |     0.8312 |         0.1186 |     0.8308 | 0.8976 |         0.8523 |      41234 |             83.0000 |      8.2539 |
| deepconvnet+bench         | deepconvnet         |           51 |        10 |     0.8276 |         0.1207 |     0.8269 | 0.8946 |         0.8444 |     302552 |             86.0000 |      4.4315 |
| seegnex@hplif+zs          | seegnex@hplif       |           51 |        10 |     0.8271 |         0.1186 |     0.8268 | 0.8943 |         0.8404 |      58850 |             22.0000 |     12.3170 |
| eeg_deformer+bench        | eeg_deformer        |           51 |        10 |     0.8271 |         0.1194 |     0.8264 | 0.8951 |         0.8462 |    1610066 |             18.0000 |      3.1385 |
| seegnex@hplif+zs+adabn    | seegnex@hplif       |           51 |        10 |     0.8270 |         0.1198 |     0.8268 | 0.8937 |         0.8404 |      58850 |             22.0000 |     12.3170 |
| eegnet_official+zs+adabn  | eegnet_official     |           51 |        10 |     0.8262 |         0.1204 |     0.8259 | 0.8920 |         0.8399 |       3026 |            153.0000 |      3.4206 |
| eegnet_official+bench     | eegnet_official     |           51 |        10 |     0.8253 |         0.1204 |     0.8250 | 0.8918 |         0.8399 |       3026 |            153.0000 |      5.0750 |
| eegnet_official+zs        | eegnet_official     |           51 |        10 |     0.8253 |         0.1204 |     0.8250 | 0.8918 |         0.8399 |       3026 |            153.0000 |      3.4206 |
| cbramod+zs                | cbramod             |           51 |        10 |     0.8251 |         0.1158 |     0.8245 | 0.8943 |         0.8363 |    4976602 |              4.0000 |      1.5215 |
| seegnet@plif+zs+adabn     | seegnet@plif        |           51 |        10 |     0.8207 |         0.1201 |     0.8205 | 0.8881 |         0.8347 |       3028 |             80.0000 |      2.1096 |
| tcformer+zs+adabn         | tcformer            |           31 |         3 |     0.8205 |         0.1153 |     0.8203 | 0.8925 |         0.8395 |     133988 |             79.0000 |      3.4086 |
| seegnet@plif+zs           | seegnet@plif        |           51 |        10 |     0.8203 |         0.1207 |     0.8201 | 0.8879 |         0.8347 |       3028 |             80.0000 |      2.1096 |
| labram+zs                 | labram              |           51 |        10 |     0.8191 |         0.1178 |     0.8185 | 0.8877 |         0.8331 |    5917938 |              8.0000 |      2.6588 |
| tcformer+zs               | tcformer            |           31 |         3 |     0.8163 |         0.1153 |     0.8155 | 0.8915 |         0.8395 |     133988 |             79.0000 |      3.4086 |
| seegnex@s_hplif+zs+adabn  | seegnex@s_hplif     |           51 |        10 |     0.8159 |         0.1199 |     0.8157 | 0.8831 |         0.8297 |      58850 |             43.5000 |     52.3939 |
| seegnex@s_hplif+zs        | seegnex@s_hplif     |           51 |        10 |     0.8142 |         0.1204 |     0.8136 | 0.8832 |         0.8297 |      58850 |             43.5000 |     52.3939 |
| s4erp+bench               | s4erp               |           51 |        10 |     0.8133 |         0.1202 |     0.8130 | 0.8824 |         0.8278 |    1370882 |              8.5000 |      7.4350 |
| s4erp_dw@f64d2+bench      | s4erp_dw@f64d2      |           51 |        10 |     0.8068 |         0.1206 |     0.8063 | 0.8771 |         0.8199 |     426370 |              4.5000 |      2.0912 |
| s4erp_dw@f16d8+bench      | s4erp_dw@f16d8      |           51 |        10 |     0.8061 |         0.1184 |     0.8056 | 0.8755 |         0.8201 |     425122 |              4.5000 |      1.7613 |
| s4erp_dw@f16d2p+bench     | s4erp_dw@f16d2p     |           51 |        10 |     0.8027 |         0.1204 |     0.8023 | 0.8734 |         0.8182 |     406978 |              4.5000 |      1.2223 |
| dualcd_transformer+bench  | dualcd_transformer  |           51 |        10 |     0.7917 |         0.1213 |     0.7912 | 0.8626 |         0.8081 |   67900034 |              7.5000 |     51.6345 |
| dualcd_s4_flatten+bench   | dualcd_s4_flatten   |           51 |        10 |     0.7901 |         0.1154 |     0.7891 | 0.8619 |         0.8056 |   66861186 |              8.0000 |     44.9684 |
| eegpt+zs                  | eegpt               |           51 |        10 |     0.7899 |         0.1124 |     0.7893 | 0.8624 |         0.8045 |   25356128 |             10.0000 |     13.2269 |
| dualcd_s4_timepatch+bench | dualcd_s4_timepatch |           51 |        10 |     0.7630 |         0.1248 |     0.7621 | 0.8333 |         0.7856 |    4480386 |             66.0000 |    106.3685 |
| atcnet_s4+bench           | atcnet_s4           |           51 |        10 |     0.7499 |         0.1072 |     0.7484 | 0.8270 |         0.7772 |      27834 |            146.5000 |      5.5987 |
| fbcnet+bench              | fbcnet              |           51 |        10 |     0.6996 |         0.1079 |     0.6950 | 0.7670 |         0.7255 |      20711 |             87.0000 |      3.0069 |
| atcnet_s4@noattn+bench    | atcnet_s4@noattn    |           51 |        10 |     0.6328 |         0.0792 |     0.6281 | 0.6864 |         0.6586 |      25642 |            113.0000 |      3.9856 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline           | candidate                | question                                                       |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |      d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:-------------------|:-------------------------|:---------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|---------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | eegnex+zs          | seegnex@s_hplif+zs       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment    |  51 |          0.8335 |           0.8142 | -0.0193 |         -0.0269 |         -0.0118 |  -0.7193 |      9 |      3 |       39 |    0.0000 |   0.0000 | True                  |
|  1 | exploratory   | seegnex@hplif+zs   | seegnex@s_hplif+zs       | H2 (no EA): temporal-native vs direct (T=4) encoding           |  51 |          0.8271 |           0.8142 | -0.0129 |         -0.0183 |         -0.0076 |  -0.6841 |     12 |      0 |       39 |    0.0000 | nan      | <NA>                  |
|  2 | exploratory   | eegnet_official+zs | seegnet@plif+zs          | S-EEGNet vs EEGNet, no alignment                               |  51 |          0.8253 |           0.8203 | -0.0049 |         -0.0079 |         -0.0019 |  -0.4640 |     16 |      3 |       32 |    0.0009 | nan      | <NA>                  |
|  3 | exploratory   | eegnex+zs          | seegnex@hplif+zs         | direct (T=4) S-EEGNeX vs EEGNeX, no alignment                  |  51 |          0.8335 |           0.8271 | -0.0064 |         -0.0105 |         -0.0022 |  -0.4334 |     13 |      3 |       35 |    0.0019 | nan      | <NA>                  |
|  4 | exploratory   | eegnex+zs          | eegnex+zs+adabn          | AdaBN alone on the ANN                                         |  51 |          0.8335 |           0.8353 |  0.0018 |          0.0003 |          0.0032 |   0.3361 |     32 |      6 |       13 |    0.0213 | nan      | <NA>                  |
|  5 | exploratory   | seegnex@s_hplif+zs | seegnex@s_hplif+zs+adabn | AdaBN alone on the SNN                                         |  51 |          0.8142 |           0.8159 |  0.0017 |         -0.0012 |          0.0047 |   0.1649 |     25 |      4 |       22 |    0.4241 | nan      | <NA>                  |
|  6 | exploratory   | eegnex+bench       | eegnex+zs                | sanity: re-run EEGNeX vs benchmark EEGNeX (same recipe/splits) |  51 |          0.8335 |           0.8335 |  0.0000 |          0.0000 |          0.0000 | nan      |      0 |     51 |        0 |  nan      | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

_(no data yet)_

## Energy / firing rates (median over subjects)

| entry                    |   rate_lif1 |      snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   rate_lif3 |   n_subject_rows |
|:-------------------------|------------:|--------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|------------:|-----------------:|
| seegnex@s_hplif+zs+adabn |      0.1430 | 34433008.0000 | 137274695.6771 |        281.9391 | 992897008.0000 |       4567.3262 |                     16.1997 |      0.1771 |         102.0000 |
| seegnex@s_hplif+zs       |      0.1434 | 34433008.0000 | 137784982.6562 |        282.3983 | 992897008.0000 |       4567.3262 |                     16.1733 |      0.1762 |         102.0000 |
| seegnex@hplif+zs         |      0.1514 | 48644032.0000 | 581512781.4583 |        747.1241 | 992897008.0000 |       4567.3262 |                      6.1132 |      0.1974 |         102.0000 |
| seegnex@hplif+zs+adabn   |      0.1513 | 48644032.0000 | 581568561.6667 |        747.1743 | 992897008.0000 |       4567.3262 |                      6.1128 |      0.1998 |         102.0000 |
| seegnet@plif+zs          |      0.2494 | 30883968.0000 |     63854.0560 |        142.1237 |  30752992.0000 |        141.4638 |                      0.9954 |    nan      |         102.0000 |
| seegnet@plif+zs+adabn    |      0.2500 | 30883968.0000 |     63999.0039 |        142.1239 |  30752992.0000 |        141.4638 |                      0.9954 |    nan      |         102.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 905.9210604505963,
 "friedman_p": 3.46562349484259e-171,
 "k": 31,
 "N": 51,
 "avg_ranks": {
  "atcnet@noattn+bench": 6.735294117647059,
  "eegnex+zs+adabn": 7.117647058823529,
  "atcnet+bench": 8.323529411764707,
  "eegnex+bench": 8.362745098039216,
  "eegnex+zs": 8.362745098039216,
  "atcnet@nwin1+bench": 8.656862745098039,
  "atcnet@tcn8+bench": 9.196078431372548,
  "deepconvnet+bench": 10.715686274509803,
  "seegnex@hplif+zs+adabn": 11.156862745098039,
  "eeg_deformer+bench": 11.254901960784315,
  "seegnex@hplif+zs": 11.303921568627452,
  "eegnet_official+zs+adabn": 11.686274509803921,
  "eegnet_official+bench": 12.107843137254902,
  "eegnet_official+zs": 12.107843137254902,
  "cbramod+zs": 13.558823529411764,
  "seegnet@plif+zs+adabn": 14.656862745098039,
  "seegnet@plif+zs": 14.803921568627452,
  "labram+zs": 15.823529411764707,
  "seegnex@s_hplif+zs+adabn": 16.529411764705884,
  "seegnex@s_hplif+zs": 17.15686274509804,
  "s4erp+bench": 18.264705882352942,
  "s4erp_dw@f64d2+bench": 20.480392156862745,
  "s4erp_dw@f16d8+bench": 20.823529411764707,
  "s4erp_dw@f16d2p+bench": 22.49019607843137,
  "eegpt+zs": 23.50980392156863,
  "dualcd_transformer+bench": 24.558823529411764,
  "dualcd_s4_flatten+bench": 24.980392156862745,
  "dualcd_s4_timepatch+bench": 26.225490196078432,
  "atcnet_s4+bench": 26.65686274509804,
  "fbcnet+bench": 28.254901960784313,
  "atcnet_s4@noattn+bench": 30.137254901960784
 },
 "critical_difference": 6.778322709231871,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
