# Spiking round — shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

arms: role `ea` <- `eaz=outputs/experiments/shu/snn_shu_eaz_lr1e3_v1`; role `zs` <- `zs=outputs/experiments/shu/snn_shu_zs_lr1e3_v1`; bench <- `results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1`

> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 (PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending (STATUS §3.3); treat p-values here as approximate.

## Per-entry table (mean over subjects)

| entry                           | model                  |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:--------------------------------|:-----------------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| eegnex+ea+adabn                 | eegnex                 |           25 |         5 |     0.6560 |         0.0836 |     0.6555 | 0.7018 |         0.6693 |      57106 |             27.0000 |      0.3841 |
| eegnex+bench                    | eegnex                 |           25 |        10 |     0.6549 |         0.0995 |     0.6478 | 0.7053 |         0.6614 |      57106 |             73.0000 |      1.6142 |
| eegnex+ea                       | eegnex                 |           25 |         5 |     0.6497 |         0.0911 |     0.6449 | 0.7037 |         0.6693 |      57106 |             27.0000 |      0.3841 |
| eegnex+zs+adabn                 | eegnex                 |           25 |         5 |     0.6409 |         0.0879 |     0.6400 | 0.6900 |         0.6750 |      57106 |             25.0000 |      0.3130 |
| seegnex@s_hplif+ea+adabn        | seegnex@s_hplif        |           25 |         5 |     0.6373 |         0.0848 |     0.6366 | 0.6805 |         0.6462 |      57186 |             11.0000 |      3.4624 |
| eegnex+zs                       | eegnex                 |           25 |         5 |     0.6358 |         0.0932 |     0.6308 | 0.6904 |         0.6750 |      57106 |             25.0000 |      0.3130 |
| seegnex@s_hplif+ea              | seegnex@s_hplif        |           25 |         5 |     0.6341 |         0.0911 |     0.6254 | 0.6839 |         0.6462 |      57186 |             11.0000 |      3.4624 |
| seegnex@s_hplif_spk+zs+adabn    | seegnex@s_hplif_spk    |           25 |         5 |     0.6332 |         0.0997 |     0.6325 | 0.6788 |         0.6630 |      57186 |             16.0000 |      3.4567 |
| seegnex@s_hplif_spk+zs          | seegnex@s_hplif_spk    |           25 |         5 |     0.6303 |         0.1040 |     0.6245 | 0.6770 |         0.6630 |      57186 |             16.0000 |      3.4567 |
| seegnex@hplif+zs+adabn          | seegnex@hplif          |           25 |         5 |     0.6295 |         0.1038 |     0.6291 | 0.6720 |         0.6681 |      57186 |             14.0000 |      1.2538 |
| seegnex@hplif+zs                | seegnex@hplif          |           25 |         5 |     0.6286 |         0.1044 |     0.6218 | 0.6730 |         0.6681 |      57186 |             14.0000 |      1.2538 |
| seegnex@s_hplif+zs+adabn        | seegnex@s_hplif        |           25 |         5 |     0.6279 |         0.1035 |     0.6272 | 0.6767 |         0.6578 |      57186 |              6.0000 |      3.8647 |
| seegnex@s_hplif_noinit+zs+adabn | seegnex@s_hplif_noinit |           25 |         5 |     0.6256 |         0.1039 |     0.6251 | 0.6686 |         0.6554 |      57186 |             16.0000 |      3.7685 |
| seegnex@s_hplif+zs              | seegnex@s_hplif        |           25 |         5 |     0.6253 |         0.1053 |     0.6206 | 0.6768 |         0.6578 |      57186 |              6.0000 |      3.8647 |
| seegnex@s_hplif_noinit+zs       | seegnex@s_hplif_noinit |           25 |         5 |     0.6223 |         0.0995 |     0.6137 | 0.6711 |         0.6554 |      57186 |             16.0000 |      3.7685 |
| atcnet+bench                    | atcnet                 |           25 |        10 |     0.6196 |         0.1123 |     0.6167 | 0.6628 |         0.6195 |     113722 |             55.5000 |      5.1860 |
| s4erp_dw@f16d8+bench            | s4erp_dw@f16d8         |           25 |        10 |     0.6168 |         0.0917 |     0.6142 | 0.6599 |         0.6190 |     421794 |             10.5000 |      0.5732 |
| atcnet@tcn8+bench               | atcnet@tcn8            |           25 |        10 |     0.6157 |         0.1103 |     0.6127 | 0.6560 |         0.6190 |      40402 |             54.0000 |      2.2831 |
| eegnet_official+zs              | eegnet_official        |           25 |         5 |     0.6157 |         0.1066 |     0.6131 | 0.6615 |         0.6365 |       2610 |              7.0000 |      0.0968 |
| eegnet_official+bench           | eegnet_official        |           25 |        10 |     0.6152 |         0.1072 |     0.6126 | 0.6557 |         0.6160 |       2610 |             41.5000 |      0.4409 |
| eegnet_official+zs+adabn        | eegnet_official        |           25 |         5 |     0.6147 |         0.1060 |     0.6133 | 0.6604 |         0.6365 |       2610 |              7.0000 |      0.0968 |
| atcnet@noattn+bench             | atcnet@noattn          |           25 |        10 |     0.6143 |         0.1094 |     0.6111 | 0.6569 |         0.6228 |     102762 |             34.0000 |      3.9707 |
| eeg_deformer+bench              | eeg_deformer           |           25 |        10 |     0.6143 |         0.0954 |     0.6081 | 0.6605 |         0.6292 |    1503570 |              7.5000 |      0.8386 |
| atcnet@nwin1+bench              | atcnet@nwin1           |           25 |        10 |     0.6135 |         0.1119 |     0.6103 | 0.6559 |         0.6140 |      37618 |             60.0000 |      1.2841 |
| s4erp+bench                     | s4erp                  |           25 |        10 |     0.6127 |         0.0952 |     0.6085 | 0.6589 |         0.6222 |     944898 |              7.0000 |      1.3191 |
| seegnet@plif+zs                 | seegnet@plif           |           25 |         5 |     0.6123 |         0.1016 |     0.6091 | 0.6531 |         0.6307 |       2612 |              3.0000 |      0.0821 |
| seegnet@plif+zs+adabn           | seegnet@plif           |           25 |         5 |     0.6120 |         0.1014 |     0.6109 | 0.6528 |         0.6307 |       2612 |              3.0000 |      0.0821 |
| deepconvnet+bench               | deepconvnet            |           25 |        10 |     0.6096 |         0.1031 |     0.5977 | 0.6518 |         0.6070 |     286302 |              6.5000 |      0.3474 |
| seegnex@s_hplif_afe+zs+adabn    | seegnex@s_hplif_afe    |           25 |         5 |     0.6089 |         0.0921 |     0.6069 | 0.6472 |         0.6273 |      57178 |              3.0000 |      0.3034 |
| dualcd_s4_timepatch+bench       | dualcd_s4_timepatch    |           25 |        10 |     0.6083 |         0.1064 |     0.6038 | 0.6483 |         0.6162 |    3628418 |              7.0000 |      7.7540 |
| seegnex@s_hplif_afe+zs          | seegnex@s_hplif_afe    |           25 |         5 |     0.6071 |         0.0927 |     0.6021 | 0.6484 |         0.6273 |      57178 |              3.0000 |      0.3034 |
| dualcd_transformer+bench        | dualcd_transformer     |           25 |        10 |     0.6052 |         0.0877 |     0.6001 | 0.6464 |         0.6126 |   67048066 |              8.5000 |     23.1936 |
| s4erp_dw@f64d2+bench            | s4erp_dw@f64d2         |           25 |        10 |     0.6052 |         0.0913 |     0.6022 | 0.6476 |         0.6142 |     423042 |              2.5000 |      0.5443 |
| dualcd_s4_flatten+bench         | dualcd_s4_flatten      |           25 |        10 |     0.6039 |         0.0827 |     0.6023 | 0.6424 |         0.6162 |   66009218 |              5.5000 |     18.3543 |
| seegnet@plif+ea+adabn           | seegnet@plif           |           25 |         5 |     0.6021 |         0.1027 |     0.6011 | 0.6385 |         0.6164 |       2612 |              3.0000 |      0.0782 |
| atcnet_s4@noattn+bench          | atcnet_s4@noattn       |           25 |        10 |     0.6014 |         0.0983 |     0.6001 | 0.6347 |         0.6049 |      24810 |             40.0000 |      0.5238 |
| atcnet_s4+bench                 | atcnet_s4              |           25 |        10 |     0.5985 |         0.1006 |     0.5953 | 0.6349 |         0.6052 |      27002 |             50.5000 |      0.6103 |
| s4erp_dw@f16d2p+bench           | s4erp_dw@f16d2p        |           25 |        10 |     0.5974 |         0.0787 |     0.5920 | 0.6402 |         0.6158 |     406146 |              2.0000 |      0.4256 |
| seegnet@plif+ea                 | seegnet@plif           |           25 |         5 |     0.5961 |         0.1010 |     0.5912 | 0.6389 |         0.6164 |       2612 |              3.0000 |      0.0782 |
| fbcnet+bench                    | fbcnet                 |           25 |        10 |     0.5924 |         0.0667 |     0.5794 | 0.6333 |         0.5928 |      13223 |             87.0000 |      0.8103 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline            | candidate                 | question                                                       |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |     d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:--------------------|:--------------------------|:---------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|--------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | eegnex+ea           | seegnex@s_hplif+ea        | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, with EA         |  25 |          0.6497 |           0.6341 | -0.0156 |         -0.0280 |         -0.0033 | -0.5218 |      6 |      2 |       17 |    0.0150 |   0.1197 | False                 |
|  1 | preregistered | eegnex+ea           | eegnex+ea+adabn           | AdaBN effect on the ANN (after EA)                             |  25 |          0.6497 |           0.6560 |  0.0063 |          0.0009 |          0.0118 |  0.4780 |     15 |      2 |        8 |    0.0386 |   0.2703 | False                 |
|  2 | preregistered | eegnex+zs           | eegnex+ea                 | EA effect on the ANN                                           |  25 |          0.6358 |           0.6497 |  0.0139 |         -0.0051 |          0.0328 |  0.3021 |     16 |      0 |        9 |    0.1094 |   0.6563 | False                 |
|  3 | preregistered | eegnex+zs           | seegnex@s_hplif+zs        | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment    |  25 |          0.6358 |           0.6253 | -0.0105 |         -0.0331 |          0.0121 | -0.1915 |     10 |      1 |       14 |    0.4405 |   1.0000 | False                 |
|  4 | preregistered | seegnex@s_hplif+zs  | seegnex@s_hplif+ea        | EA effect on the SNN                                           |  25 |          0.6253 |           0.6341 |  0.0087 |         -0.0164 |          0.0339 |  0.1436 |     14 |      0 |       11 |    0.5272 |   1.0000 | False                 |
|  5 | preregistered | seegnex@s_hplif+ea  | seegnex@s_hplif+ea+adabn  | AdaBN effect on the SNN (after EA)                             |  25 |          0.6341 |           0.6373 |  0.0032 |         -0.0050 |          0.0114 |  0.1601 |     11 |      4 |       10 |    0.6389 |   1.0000 | False                 |
|  6 | exploratory   | eegnex+bench        | eegnex+zs                 | sanity: re-run EEGNeX vs benchmark EEGNeX (same recipe/splits) |  25 |          0.6549 |           0.6358 | -0.0191 |         -0.0340 |         -0.0041 | -0.5264 |      8 |      0 |       17 |    0.0230 | nan      | <NA>                  |
|  7 | exploratory   | seegnex@s_hplif+zs  | seegnex@s_hplif_afe+zs    | ablation: analog front-end (spikes after spatial filter)       |  25 |          0.6253 |           0.6071 | -0.0182 |         -0.0355 |         -0.0009 | -0.4355 |      8 |      0 |       17 |    0.0283 | nan      | <NA>                  |
|  8 | exploratory   | eegnex+zs           | seegnex@hplif+zs          | direct (T=4) S-EEGNeX vs EEGNeX, no alignment                  |  25 |          0.6358 |           0.6286 | -0.0072 |         -0.0316 |          0.0172 | -0.1213 |     11 |      0 |       14 |    0.6766 | nan      | <NA>                  |
|  9 | exploratory   | eegnex+zs           | eegnex+zs+adabn           | AdaBN alone on the ANN                                         |  25 |          0.6358 |           0.6409 |  0.0051 |         -0.0018 |          0.0121 |  0.3035 |     12 |      3 |       10 |    0.2113 | nan      | <NA>                  |
| 10 | exploratory   | seegnex@hplif+zs    | seegnex@s_hplif+zs        | H2 (no EA): temporal-native vs direct (T=4) encoding           |  25 |          0.6286 |           0.6253 | -0.0033 |         -0.0168 |          0.0102 | -0.1012 |     12 |      0 |       13 |    0.7366 | nan      | <NA>                  |
| 11 | exploratory   | eegnet_official+zs  | seegnet@plif+zs           | S-EEGNet vs EEGNet, no alignment                               |  25 |          0.6157 |           0.6123 | -0.0033 |         -0.0164 |          0.0097 | -0.1060 |     11 |      2 |       12 |    0.5633 | nan      | <NA>                  |
| 12 | exploratory   | seegnex@s_hplif+zs  | seegnex@s_hplif_spk+zs    | ablation: binary spike readout instead of LI                   |  25 |          0.6253 |           0.6303 |  0.0050 |         -0.0141 |          0.0242 |  0.1082 |     11 |      0 |       14 |    0.7570 | nan      | <NA>                  |
| 13 | exploratory   | seegnex@s_hplif+zs  | seegnex@s_hplif+zs+adabn  | AdaBN alone on the SNN                                         |  25 |          0.6253 |           0.6279 |  0.0026 |         -0.0018 |          0.0071 |  0.2423 |     12 |      6 |        7 |    0.2122 | nan      | <NA>                  |
| 14 | exploratory   | seegnex@s_hplif+zs  | seegnex@s_hplif_noinit+zs | ablation: default BN init before neurons                       |  25 |          0.6253 |           0.6223 | -0.0030 |         -0.0153 |          0.0092 | -0.1022 |      9 |      1 |       15 |    0.5111 | nan      | <NA>                  |
| 15 | exploratory   | atcnet+bench        | seegnex@s_hplif+ea+adabn  | vs strongest benchmark ANN (ATCNet)                            |  25 |          0.6196 |           0.6373 |  0.0176 |         -0.0137 |          0.0489 |  0.2320 |     15 |      0 |       10 |    0.2209 | nan      | <NA>                  |
| 16 | exploratory   | eegnex+bench        | seegnex@s_hplif+ea+adabn  | vs benchmark EEGNeX                                            |  25 |          0.6549 |           0.6373 | -0.0176 |         -0.0364 |          0.0012 | -0.3868 |     12 |      0 |       13 |    0.1094 | nan      | <NA>                  |
| 17 | exploratory   | atcnet@noattn+bench | seegnex@s_hplif+ea+adabn  | vs benchmark #1 on 2C (ATCNet w/o attention)                   |  25 |          0.6143 |           0.6373 |  0.0230 |         -0.0076 |          0.0535 |  0.3102 |     16 |      2 |        7 |    0.1361 | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

|    | question                                                    |   n |   gain_baseline |   gain_candidate |   interaction |   interaction_median |   ci95_lo |   ci95_hi |     d_z |   wins |   losses |   p_value |   p_holm |
|---:|:------------------------------------------------------------|----:|----------------:|-----------------:|--------------:|---------------------:|----------:|----------:|--------:|-------:|---------:|----------:|---------:|
|  0 | H1: EA gain SNN(stream) minus EA gain ANN                   |  25 |          0.0139 |           0.0087 |       -0.0051 |              -0.0189 |   -0.0295 |    0.0193 | -0.0867 |     10 |       15 |    0.3395 |   0.6790 |
|  1 | H1b: AdaBN gain SNN(stream) minus AdaBN gain ANN (after EA) |  25 |          0.0063 |           0.0032 |       -0.0032 |              -0.0081 |   -0.0125 |    0.0062 | -0.1388 |     10 |       14 |    0.1300 |   0.3899 |

## Energy / firing rates (median over subjects)

| entry                           |   rate_lif1 |       snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   rate_lif3 |   rate_lif5 |   n_subject_rows |
|:--------------------------------|------------:|---------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|------------:|------------:|-----------------:|
| seegnex@s_hplif_noinit+zs       |      0.0694 |  19457008.0000 |  36715665.8403 |        122.5463 | 551937008.0000 |       2538.9102 |                     20.7180 |      0.0487 |    nan      |          25.0000 |
| seegnex@s_hplif_noinit+zs+adabn |      0.0726 |  19457008.0000 |  38491351.3478 |        124.1445 | 551937008.0000 |       2538.9102 |                     20.4513 |      0.0455 |    nan      |          25.0000 |
| seegnex@s_hplif+ea              |      0.1114 |  19457008.0000 |  59916391.7181 |        143.4270 | 551937008.0000 |       2538.9102 |                     17.7018 |      0.1958 |    nan      |          25.0000 |
| seegnex@s_hplif+ea+adabn        |      0.1127 |  19457008.0000 |  60630280.3385 |        144.0695 | 551937008.0000 |       2538.9102 |                     17.6228 |      0.1981 |    nan      |          25.0000 |
| seegnex@s_hplif_spk+zs+adabn    |      0.1200 |  19456000.0000 |  64396734.5719 |        147.4547 | 551937008.0000 |       2538.9102 |                     17.2182 |      0.2036 |      0.3104 |          25.0000 |
| seegnex@s_hplif+zs+adabn        |      0.1202 |  19457008.0000 |  64545489.2241 |        147.5932 | 551937008.0000 |       2538.9102 |                     17.2021 |      0.1958 |    nan      |          25.0000 |
| seegnex@s_hplif_spk+zs          |      0.1215 |  19456000.0000 |  65284608.4523 |        148.2537 | 551937008.0000 |       2538.9102 |                     17.1254 |      0.2004 |      0.3250 |          25.0000 |
| seegnex@s_hplif+zs              |      0.1270 |  19457008.0000 |  68210902.1053 |        150.8920 | 551937008.0000 |       2538.9102 |                     16.8260 |      0.2045 |    nan      |          25.0000 |
| seegnex@hplif+zs+adabn          |      0.1139 |  28676032.0000 | 245473509.5949 |        352.8359 | 551937008.0000 |       2538.9102 |                      7.1957 |      0.2047 |    nan      |          25.0000 |
| seegnex@hplif+zs                |      0.1143 |  28676032.0000 | 246303096.3489 |        353.5825 | 551937008.0000 |       2538.9102 |                      7.1805 |      0.2058 |    nan      |          25.0000 |
| seegnex@s_hplif_afe+zs          |    nan      | 543745008.0000 |   1856391.3158 |       2502.8978 | 551937008.0000 |       2538.9102 |                      1.0144 |      0.2266 |    nan      |          25.0000 |
| seegnex@s_hplif_afe+zs+adabn    |    nan      | 543745008.0000 |   1869922.5460 |       2502.9100 | 551937008.0000 |       2538.9102 |                      1.0144 |      0.2283 |    nan      |          25.0000 |
| seegnet@plif+zs                 |      0.2521 |  17155968.0000 |     64537.5956 |         78.9755 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+ea                 |      0.2527 |  17155968.0000 |     64700.5986 |         78.9757 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+zs+adabn           |      0.2585 |  17155968.0000 |     66175.1629 |         78.9770 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+ea+adabn           |      0.2598 |  17155968.0000 |     66519.9044 |         78.9773 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 148.65638749967104,
 "friedman_p": 1.0351392904015493e-14,
 "k": 40,
 "N": 25,
 "avg_ranks": {
  "eegnex+bench": 9.72,
  "eegnex+ea+adabn": 9.8,
  "eegnex+ea": 12.74,
  "eegnex+zs+adabn": 13.16,
  "seegnex@s_hplif+ea+adabn": 14.74,
  "eegnex+zs": 15.06,
  "seegnex@s_hplif+ea": 15.74,
  "seegnex@s_hplif+zs+adabn": 16.36,
  "seegnex@s_hplif_spk+zs+adabn": 16.94,
  "seegnex@s_hplif+zs": 17.78,
  "seegnex@s_hplif_spk+zs": 17.8,
  "seegnex@hplif+zs": 18.7,
  "seegnex@hplif+zs+adabn": 18.8,
  "seegnex@s_hplif_noinit+zs+adabn": 19.44,
  "atcnet+bench": 20.06,
  "seegnex@s_hplif_noinit+zs": 20.24,
  "eeg_deformer+bench": 20.42,
  "eegnet_official+zs": 20.78,
  "atcnet@noattn+bench": 20.94,
  "s4erp_dw@f16d8+bench": 21.0,
  "s4erp+bench": 21.06,
  "atcnet@tcn8+bench": 21.44,
  "eegnet_official+bench": 21.68,
  "eegnet_official+zs+adabn": 21.9,
  "atcnet@nwin1+bench": 22.28,
  "seegnet@plif+zs": 22.4,
  "seegnet@plif+zs+adabn": 22.74,
  "dualcd_s4_flatten+bench": 23.22,
  "seegnex@s_hplif_afe+zs": 23.4,
  "seegnex@s_hplif_afe+zs+adabn": 23.44,
  "s4erp_dw@f64d2+bench": 23.56,
  "dualcd_transformer+bench": 23.64,
  "deepconvnet+bench": 23.96,
  "dualcd_s4_timepatch+bench": 24.16,
  "seegnet@plif+ea+adabn": 25.64,
  "s4erp_dw@f16d2p+bench": 26.38,
  "fbcnet+bench": 26.46,
  "atcnet_s4@noattn+bench": 26.72,
  "atcnet_s4+bench": 27.58,
  "seegnet@plif+ea": 28.12
 },
 "critical_difference": 12.85466959660867,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
