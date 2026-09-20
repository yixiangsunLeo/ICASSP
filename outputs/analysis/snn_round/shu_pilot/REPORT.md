# Spiking round — shu

metric for tests: **accuracy**; unit = subject (mean over split seeds first)

## Per-entry table (mean over subjects)

| entry                           | model                  |   n_subjects |   n_cells |   accuracy |   accuracy_std |   macro_f1 |    auc |   val_macro_f1 |   n_params |   best_epoch_median |   gpu_hours |
|:--------------------------------|:-----------------------|-------------:|----------:|-----------:|---------------:|-----------:|-------:|---------------:|-----------:|--------------------:|------------:|
| eegnex+eaz+adabn                | eegnex                 |           25 |         5 |     0.6560 |         0.0836 |     0.6555 | 0.7018 |         0.6693 |      57106 |             27.0000 |      0.3841 |
| eegnex+bench                    | eegnex                 |           25 |        10 |     0.6549 |         0.0995 |     0.6478 | 0.7053 |         0.6614 |      57106 |             73.0000 |      1.6142 |
| eegnex+eaz                      | eegnex                 |           25 |         5 |     0.6497 |         0.0911 |     0.6449 | 0.7037 |         0.6693 |      57106 |             27.0000 |      0.3841 |
| eegnex+zs4                      | eegnex                 |           25 |         5 |     0.6465 |         0.1046 |     0.6382 | 0.6956 |         0.6808 |      57106 |             72.0000 |      0.5947 |
| eegnex+zs4+adabn                | eegnex                 |           25 |         5 |     0.6458 |         0.1029 |     0.6450 | 0.6936 |         0.6808 |      57106 |             72.0000 |      0.5947 |
| seegnex@hplif+zs4+adabn         | seegnex@hplif          |           25 |         5 |     0.6431 |         0.1020 |     0.6423 | 0.6884 |         0.6666 |      57186 |             20.0000 |      1.4573 |
| seegnex@hplif+zs4               | seegnex@hplif          |           25 |         5 |     0.6417 |         0.1012 |     0.6297 | 0.6898 |         0.6666 |      57186 |             20.0000 |      1.4573 |
| eegnex+zs+adabn                 | eegnex                 |           25 |         5 |     0.6409 |         0.0879 |     0.6400 | 0.6900 |         0.6750 |      57106 |             25.0000 |      0.3130 |
| eegnex+ea+adabn                 | eegnex                 |           25 |         5 |     0.6390 |         0.0916 |     0.6383 | 0.6838 |         0.6551 |      57106 |             17.0000 |      0.3496 |
| seegnex@s_hplif+eaz+adabn       | seegnex@s_hplif        |           25 |         5 |     0.6373 |         0.0848 |     0.6366 | 0.6805 |         0.6462 |      57186 |             11.0000 |      3.4624 |
| eegnex+ea                       | eegnex                 |           25 |         5 |     0.6358 |         0.0905 |     0.6315 | 0.6864 |         0.6551 |      57106 |             17.0000 |      0.3496 |
| eegnex+zs                       | eegnex                 |           25 |         5 |     0.6358 |         0.0932 |     0.6308 | 0.6904 |         0.6750 |      57106 |             25.0000 |      0.3130 |
| seegnex@s_hplif+eaz             | seegnex@s_hplif        |           25 |         5 |     0.6341 |         0.0911 |     0.6254 | 0.6839 |         0.6462 |      57186 |             11.0000 |      3.4624 |
| seegnex@s_hplif+zs4+adabn       | seegnex@s_hplif        |           25 |         5 |     0.6338 |         0.1064 |     0.6332 | 0.6789 |         0.6607 |      57186 |             13.0000 |      3.4437 |
| seegnex@s_hplif_spk+zs+adabn    | seegnex@s_hplif_spk    |           25 |         5 |     0.6332 |         0.0997 |     0.6325 | 0.6788 |         0.6630 |      57186 |             16.0000 |      3.4567 |
| seegnex@s_hplif+zs4             | seegnex@s_hplif        |           25 |         5 |     0.6327 |         0.1101 |     0.6233 | 0.6803 |         0.6607 |      57186 |             13.0000 |      3.4437 |
| seegnex@s_hplif_spk+zs          | seegnex@s_hplif_spk    |           25 |         5 |     0.6303 |         0.1040 |     0.6245 | 0.6770 |         0.6630 |      57186 |             16.0000 |      3.4567 |
| seegnex@hplif+zs+adabn          | seegnex@hplif          |           25 |         5 |     0.6295 |         0.1038 |     0.6291 | 0.6720 |         0.6681 |      57186 |             14.0000 |      1.2538 |
| seegnex@hplif+zs                | seegnex@hplif          |           25 |         5 |     0.6286 |         0.1044 |     0.6218 | 0.6730 |         0.6681 |      57186 |             14.0000 |      1.2538 |
| seegnex@s_hplif+zs+adabn        | seegnex@s_hplif        |           25 |         5 |     0.6279 |         0.1035 |     0.6272 | 0.6767 |         0.6578 |      57186 |              6.0000 |      3.8647 |
| seegnex@s_hplif_noinit+zs+adabn | seegnex@s_hplif_noinit |           25 |         5 |     0.6256 |         0.1039 |     0.6251 | 0.6686 |         0.6554 |      57186 |             16.0000 |      3.7685 |
| seegnex@s_hplif+zs              | seegnex@s_hplif        |           25 |         5 |     0.6253 |         0.1053 |     0.6206 | 0.6768 |         0.6578 |      57186 |              6.0000 |      3.8647 |
| seegnex@s_hplif+eamu            | seegnex@s_hplif        |           25 |         5 |     0.6238 |         0.0934 |     0.6188 | 0.6649 |         0.6333 |      57186 |             13.0000 |      3.2859 |
| eegnex+eamu+adabn               | eegnex                 |           25 |         5 |     0.6236 |         0.0932 |     0.6226 | 0.6681 |         0.6500 |      57106 |             17.0000 |      0.4269 |
| eegnex+eamu                     | eegnex                 |           25 |         5 |     0.6232 |         0.0907 |     0.6188 | 0.6733 |         0.6500 |      57106 |             17.0000 |      0.4269 |
| seegnex@s_hplif_noinit+zs       | seegnex@s_hplif_noinit |           25 |         5 |     0.6223 |         0.0995 |     0.6137 | 0.6711 |         0.6554 |      57186 |             16.0000 |      3.7685 |
| atcnet+bench                    | atcnet                 |           25 |        10 |     0.6196 |         0.1123 |     0.6167 | 0.6628 |         0.6195 |     113722 |             55.5000 |      5.1860 |
| seegnex@s_hplif+eamu+adabn      | seegnex@s_hplif        |           25 |         5 |     0.6191 |         0.0965 |     0.6184 | 0.6623 |         0.6333 |      57186 |             13.0000 |      3.2859 |
| seegnex@hplif+ea                | seegnex@hplif          |           25 |         5 |     0.6172 |         0.0814 |     0.6114 | 0.6558 |         0.6415 |      57186 |              9.0000 |      1.1230 |
| s4erp_dw@f16d8+bench            | s4erp_dw@f16d8         |           25 |        10 |     0.6168 |         0.0917 |     0.6142 | 0.6599 |         0.6190 |     421794 |             10.5000 |      0.5732 |
| eegnet_official+zs4+adabn       | eegnet_official        |           25 |         5 |     0.6166 |         0.1051 |     0.6149 | 0.6571 |         0.6327 |       2610 |             45.0000 |      0.1631 |
| atcnet@tcn8+bench               | atcnet@tcn8            |           25 |        10 |     0.6157 |         0.1103 |     0.6127 | 0.6560 |         0.6190 |      40402 |             54.0000 |      2.2831 |
| eegnet_official+zs              | eegnet_official        |           25 |         5 |     0.6157 |         0.1066 |     0.6131 | 0.6615 |         0.6365 |       2610 |              7.0000 |      0.0968 |
| seegnex@s_hplif+ea              | seegnex@s_hplif        |           25 |         5 |     0.6155 |         0.0933 |     0.6100 | 0.6520 |         0.6298 |      57186 |              5.0000 |      4.7827 |
| eegnet_official+bench           | eegnet_official        |           25 |        10 |     0.6152 |         0.1072 |     0.6126 | 0.6557 |         0.6160 |       2610 |             41.5000 |      0.4409 |
| eegnet_official+ea              | eegnet_official        |           25 |         5 |     0.6151 |         0.0965 |     0.6129 | 0.6562 |         0.6273 |       2610 |             19.0000 |      0.0971 |
| eegnet_official+zs4             | eegnet_official        |           25 |         5 |     0.6150 |         0.1049 |     0.6117 | 0.6570 |         0.6327 |       2610 |             45.0000 |      0.1631 |
| eegnet_official+zs+adabn        | eegnet_official        |           25 |         5 |     0.6147 |         0.1060 |     0.6133 | 0.6604 |         0.6365 |       2610 |              7.0000 |      0.0968 |
| atcnet@noattn+bench             | atcnet@noattn          |           25 |        10 |     0.6143 |         0.1094 |     0.6111 | 0.6569 |         0.6228 |     102762 |             34.0000 |      3.9707 |
| eeg_deformer+bench              | eeg_deformer           |           25 |        10 |     0.6143 |         0.0954 |     0.6081 | 0.6605 |         0.6292 |    1503570 |              7.5000 |      0.8386 |
| atcnet@nwin1+bench              | atcnet@nwin1           |           25 |        10 |     0.6135 |         0.1119 |     0.6103 | 0.6559 |         0.6140 |      37618 |             60.0000 |      1.2841 |
| seegnex@hplif+ea+adabn          | seegnex@hplif          |           25 |         5 |     0.6135 |         0.0824 |     0.6123 | 0.6542 |         0.6415 |      57186 |              9.0000 |      1.1230 |
| seegnex@s_hplif+ea+adabn        | seegnex@s_hplif        |           25 |         5 |     0.6128 |         0.0939 |     0.6121 | 0.6501 |         0.6298 |      57186 |              5.0000 |      4.7827 |
| s4erp+bench                     | s4erp                  |           25 |        10 |     0.6127 |         0.0952 |     0.6085 | 0.6589 |         0.6222 |     944898 |              7.0000 |      1.3191 |
| seegnet@plif+zs                 | seegnet@plif           |           25 |         5 |     0.6123 |         0.1016 |     0.6091 | 0.6531 |         0.6307 |       2612 |              3.0000 |      0.0821 |
| seegnet@plif+zs+adabn           | seegnet@plif           |           25 |         5 |     0.6120 |         0.1014 |     0.6109 | 0.6528 |         0.6307 |       2612 |              3.0000 |      0.0821 |
| eegnet_official+ea+adabn        | eegnet_official        |           25 |         5 |     0.6111 |         0.0988 |     0.6102 | 0.6557 |         0.6273 |       2610 |             19.0000 |      0.0971 |
| deepconvnet+bench               | deepconvnet            |           25 |        10 |     0.6096 |         0.1031 |     0.5977 | 0.6518 |         0.6070 |     286302 |              6.5000 |      0.3474 |
| seegnex@s_hplif_afe+zs+adabn    | seegnex@s_hplif_afe    |           25 |         5 |     0.6089 |         0.0921 |     0.6069 | 0.6472 |         0.6273 |      57178 |              3.0000 |      0.3034 |
| dualcd_s4_timepatch+bench       | dualcd_s4_timepatch    |           25 |        10 |     0.6083 |         0.1064 |     0.6038 | 0.6483 |         0.6162 |    3628418 |              7.0000 |      7.7540 |
| seegnet@plif+zs4+adabn          | seegnet@plif           |           25 |         5 |     0.6071 |         0.1130 |     0.6058 | 0.6503 |         0.6228 |       2612 |             20.0000 |      0.1102 |
| seegnex@s_hplif_afe+zs          | seegnex@s_hplif_afe    |           25 |         5 |     0.6071 |         0.0927 |     0.6021 | 0.6484 |         0.6273 |      57178 |              3.0000 |      0.3034 |
| seegnet@plif+zs4                | seegnet@plif           |           25 |         5 |     0.6056 |         0.1109 |     0.5985 | 0.6495 |         0.6228 |       2612 |             20.0000 |      0.1102 |
| dualcd_transformer+bench        | dualcd_transformer     |           25 |        10 |     0.6052 |         0.0877 |     0.6001 | 0.6464 |         0.6126 |   67048066 |              8.5000 |     23.1936 |
| s4erp_dw@f64d2+bench            | s4erp_dw@f64d2         |           25 |        10 |     0.6052 |         0.0913 |     0.6022 | 0.6476 |         0.6142 |     423042 |              2.5000 |      0.5443 |
| dualcd_s4_flatten+bench         | dualcd_s4_flatten      |           25 |        10 |     0.6039 |         0.0827 |     0.6023 | 0.6424 |         0.6162 |   66009218 |              5.5000 |     18.3543 |
| seegnet@plif+eaz+adabn          | seegnet@plif           |           25 |         5 |     0.6021 |         0.1027 |     0.6011 | 0.6385 |         0.6164 |       2612 |              3.0000 |      0.0782 |
| atcnet_s4@noattn+bench          | atcnet_s4@noattn       |           25 |        10 |     0.6014 |         0.0983 |     0.6001 | 0.6347 |         0.6049 |      24810 |             40.0000 |      0.5238 |
| atcnet_s4+bench                 | atcnet_s4              |           25 |        10 |     0.5985 |         0.1006 |     0.5953 | 0.6349 |         0.6052 |      27002 |             50.5000 |      0.6103 |
| s4erp_dw@f16d2p+bench           | s4erp_dw@f16d2p        |           25 |        10 |     0.5974 |         0.0787 |     0.5920 | 0.6402 |         0.6158 |     406146 |              2.0000 |      0.4256 |
| seegnet@plif+ea+adabn           | seegnet@plif           |           25 |         5 |     0.5964 |         0.0969 |     0.5957 | 0.6328 |         0.6207 |       2612 |              3.0000 |      0.0773 |
| seegnet@plif+eaz                | seegnet@plif           |           25 |         5 |     0.5961 |         0.1010 |     0.5912 | 0.6389 |         0.6164 |       2612 |              3.0000 |      0.0782 |
| fbcnet+bench                    | fbcnet                 |           25 |        10 |     0.5924 |         0.0667 |     0.5794 | 0.6333 |         0.5928 |      13223 |             87.0000 |      0.8103 |
| seegnet@plif+ea                 | seegnet@plif           |           25 |         5 |     0.5912 |         0.0952 |     0.5869 | 0.6338 |         0.6207 |       2612 |              3.0000 |      0.0773 |
| seegnet@plif+eamu               | seegnet@plif           |           25 |         5 |     0.5908 |         0.0939 |     0.5881 | 0.6312 |         0.6189 |       2612 |              3.0000 |      0.0789 |
| seegnet@plif+eamu+adabn         | seegnet@plif           |           25 |         5 |     0.5882 |         0.0950 |     0.5865 | 0.6317 |         0.6189 |       2612 |              3.0000 |      0.0789 |

## Pre-registered paired tests (Holm within family) + exploratory

|    | family        | baseline            | candidate                | question                                                       |   n |   mean_baseline |   mean_candidate |   delta |   delta_ci95_lo |   delta_ci95_hi |     d_z |   wins |   ties |   losses |   p_value |   p_holm | significant_holm_05   |
|---:|:--------------|:--------------------|:-------------------------|:---------------------------------------------------------------|----:|----------------:|-----------------:|--------:|----------------:|----------------:|--------:|-------:|-------:|---------:|----------:|---------:|:----------------------|
|  0 | preregistered | seegnet@plif+ea     | seegnex@hplif+ea         | S-EEGNet vs S-EEGNeX (same direct encoding)                    |  25 |          0.5912 |           0.6172 |  0.0259 |          0.0016 |          0.0503 |  0.4395 |     20 |      0 |        5 |    0.0149 |   0.1191 | False                 |
|  1 | preregistered | eegnex+ea           | seegnex@s_hplif+ea       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, with EA         |  25 |          0.6358 |           0.6155 | -0.0203 |         -0.0374 |         -0.0032 | -0.4892 |      9 |      1 |       15 |    0.0345 |   0.2414 | False                 |
|  2 | preregistered | eegnex+zs           | eegnex+ea                | EA effect on the ANN                                           |  25 |          0.6358 |           0.6358 |  0.0000 |         -0.0169 |          0.0169 |  0.0002 |     13 |      0 |       12 |    0.8612 |   1.0000 | False                 |
|  3 | preregistered | eegnex+zs           | seegnex@s_hplif+zs       | H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment    |  25 |          0.6358 |           0.6253 | -0.0105 |         -0.0331 |          0.0121 | -0.1915 |     10 |      1 |       14 |    0.4405 |   1.0000 | False                 |
|  4 | preregistered | seegnex@s_hplif+zs  | seegnex@s_hplif+ea       | EA effect on the SNN                                           |  25 |          0.6253 |           0.6155 | -0.0098 |         -0.0319 |          0.0123 | -0.1829 |      9 |      0 |       16 |    0.4593 |   1.0000 | False                 |
|  5 | preregistered | eegnex+ea           | eegnex+ea+adabn          | AdaBN effect on the ANN (after EA)                             |  25 |          0.6358 |           0.6390 |  0.0032 |         -0.0031 |          0.0094 |  0.2097 |     13 |      2 |       10 |    0.3011 |   1.0000 | False                 |
|  6 | preregistered | seegnex@s_hplif+ea  | seegnex@s_hplif+ea+adabn | AdaBN effect on the SNN (after EA)                             |  25 |          0.6155 |           0.6128 | -0.0027 |         -0.0078 |          0.0023 | -0.2231 |      9 |      2 |       14 |    0.2416 |   1.0000 | False                 |
|  7 | preregistered | seegnex@hplif+ea    | seegnex@s_hplif+ea       | H2: temporal-native vs direct (T=4) encoding                   |  25 |          0.6172 |           0.6155 | -0.0017 |         -0.0182 |          0.0149 | -0.0414 |     11 |      0 |       14 |    0.5098 |   1.0000 | False                 |
|  8 | exploratory   | eegnex+bench        | seegnex@s_hplif+ea+adabn | vs benchmark EEGNeX                                            |  25 |          0.6549 |           0.6128 | -0.0421 |         -0.0627 |         -0.0214 | -0.8402 |      4 |      0 |       21 |    0.0010 | nan      | <NA>                  |
|  9 | exploratory   | eegnex+bench        | eegnex+zs                | sanity: re-run EEGNeX vs benchmark EEGNeX (same recipe/splits) |  25 |          0.6549 |           0.6358 | -0.0191 |         -0.0340 |         -0.0041 | -0.5264 |      8 |      0 |       17 |    0.0230 | nan      | <NA>                  |
| 10 | exploratory   | seegnex@s_hplif+zs  | seegnex@s_hplif+zs+adabn | AdaBN alone on the SNN                                         |  25 |          0.6253 |           0.6279 |  0.0026 |         -0.0018 |          0.0071 |  0.2423 |     12 |      6 |        7 |    0.2122 | nan      | <NA>                  |
| 11 | exploratory   | eegnex+zs           | eegnex+zs+adabn          | AdaBN alone on the ANN                                         |  25 |          0.6358 |           0.6409 |  0.0051 |         -0.0018 |          0.0121 |  0.3035 |     12 |      3 |       10 |    0.2113 | nan      | <NA>                  |
| 12 | exploratory   | atcnet+bench        | seegnex@s_hplif+ea+adabn | vs strongest benchmark ANN (ATCNet)                            |  25 |          0.6196 |           0.6128 | -0.0068 |         -0.0292 |          0.0155 | -0.1267 |     10 |      0 |       15 |    0.4118 | nan      | <NA>                  |
| 13 | exploratory   | atcnet@noattn+bench | seegnex@s_hplif+ea+adabn | vs benchmark #1 on 2C (ATCNet w/o attention)                   |  25 |          0.6143 |           0.6128 | -0.0015 |         -0.0223 |          0.0193 | -0.0298 |     10 |      0 |       15 |    0.5272 | nan      | <NA>                  |

## Interaction (H1): does alignment help the SNN more than the ANN?

|    | question                                                    |   n |   gain_baseline |   gain_candidate |   interaction |   interaction_median |   ci95_lo |   ci95_hi |     d_z |   wins |   losses |   p_value |   p_holm |
|---:|:------------------------------------------------------------|----:|----------------:|-----------------:|--------------:|---------------------:|----------:|----------:|--------:|-------:|---------:|----------:|---------:|
|  0 | H1: EA gain SNN(stream) minus EA gain ANN                   |  25 |          0.0000 |          -0.0098 |       -0.0098 |              -0.0124 |   -0.0267 |    0.0071 | -0.2390 |      9 |       16 |    0.3002 |   0.4829 |
|  1 | H1b: AdaBN gain SNN(stream) minus AdaBN gain ANN (after EA) |  25 |          0.0032 |          -0.0027 |       -0.0059 |              -0.0085 |   -0.0142 |    0.0025 | -0.2902 |      7 |       18 |    0.0980 |   0.2939 |
|  2 | H1 (direct SNN): EA gain SNN(direct) minus EA gain ANN      |  25 |          0.0000 |          -0.0114 |       -0.0115 |              -0.0101 |   -0.0311 |    0.0082 | -0.2406 |      8 |       16 |    0.2414 |   0.4829 |

## Energy / firing rates (median over subjects)

| entry                           |   rate_lif1 |       snn_macs |        snn_acs |   snn_energy_uJ |       ann_macs |   ann_energy_uJ |   energy_ratio_ann_over_snn |   rate_lif3 |   rate_lif5 |   n_subject_rows |
|:--------------------------------|------------:|---------------:|---------------:|----------------:|---------------:|----------------:|----------------------------:|------------:|------------:|-----------------:|
| seegnex@s_hplif_noinit+zs       |      0.0694 |  19457008.0000 |  36715665.8403 |        122.5463 | 551937008.0000 |       2538.9102 |                     20.7180 |      0.0487 |    nan      |          25.0000 |
| seegnex@s_hplif_noinit+zs+adabn |      0.0726 |  19457008.0000 |  38491351.3478 |        124.1445 | 551937008.0000 |       2538.9102 |                     20.4513 |      0.0455 |    nan      |          25.0000 |
| seegnex@s_hplif+eamu+adabn      |      0.1033 |  19457008.0000 |  55633219.7873 |        139.5721 | 551937008.0000 |       2538.9102 |                     18.1907 |      0.2017 |    nan      |          25.0000 |
| seegnex@s_hplif+eamu            |      0.1062 |  19457008.0000 |  57289047.9772 |        141.0624 | 551937008.0000 |       2538.9102 |                     17.9985 |      0.2046 |    nan      |          25.0000 |
| seegnex@s_hplif+ea+adabn        |      0.1096 |  19457008.0000 |  59031888.1173 |        142.6309 | 551937008.0000 |       2538.9102 |                     17.8006 |      0.2026 |    nan      |          25.0000 |
| seegnex@s_hplif+eaz             |      0.1114 |  19457008.0000 |  59916391.7181 |        143.4270 | 551937008.0000 |       2538.9102 |                     17.7018 |      0.1958 |    nan      |          25.0000 |
| seegnex@s_hplif+eaz+adabn       |      0.1127 |  19457008.0000 |  60630280.3385 |        144.0695 | 551937008.0000 |       2538.9102 |                     17.6228 |      0.1981 |    nan      |          25.0000 |
| seegnex@s_hplif+ea              |      0.1137 |  19457008.0000 |  61243527.2634 |        144.6214 | 551937008.0000 |       2538.9102 |                     17.5556 |      0.2088 |    nan      |          25.0000 |
| seegnex@s_hplif_spk+zs+adabn    |      0.1200 |  19456000.0000 |  64396734.5719 |        147.4547 | 551937008.0000 |       2538.9102 |                     17.2182 |      0.2036 |      0.3104 |          25.0000 |
| seegnex@s_hplif+zs+adabn        |      0.1202 |  19457008.0000 |  64545489.2241 |        147.5932 | 551937008.0000 |       2538.9102 |                     17.2021 |      0.1958 |    nan      |          25.0000 |
| seegnex@s_hplif+zs4+adabn       |      0.1208 |  19457008.0000 |  64555473.4887 |        147.6022 | 551937008.0000 |       2538.9102 |                     17.2010 |      0.1514 |    nan      |          25.0000 |
| seegnex@s_hplif+zs4             |      0.1207 |  19457008.0000 |  64585362.7277 |        147.6291 | 551937008.0000 |       2538.9102 |                     17.1979 |      0.1530 |    nan      |          25.0000 |
| seegnex@s_hplif_spk+zs          |      0.1215 |  19456000.0000 |  65284608.4523 |        148.2537 | 551937008.0000 |       2538.9102 |                     17.1254 |      0.2004 |      0.3250 |          25.0000 |
| seegnex@s_hplif+zs              |      0.1270 |  19457008.0000 |  68210902.1053 |        150.8920 | 551937008.0000 |       2538.9102 |                     16.8260 |      0.2045 |    nan      |          25.0000 |
| seegnex@hplif+ea                |      0.1059 |  28676032.0000 | 229319038.9016 |        338.2969 | 551937008.0000 |       2538.9102 |                      7.5050 |      0.2153 |    nan      |          25.0000 |
| seegnex@hplif+ea+adabn          |      0.1071 |  28676032.0000 | 231672537.7574 |        340.4150 | 551937008.0000 |       2538.9102 |                      7.4583 |      0.2130 |    nan      |          25.0000 |
| seegnex@hplif+zs+adabn          |      0.1139 |  28676032.0000 | 245473509.5949 |        352.8359 | 551937008.0000 |       2538.9102 |                      7.1957 |      0.2047 |    nan      |          25.0000 |
| seegnex@hplif+zs                |      0.1143 |  28676032.0000 | 246303096.3489 |        353.5825 | 551937008.0000 |       2538.9102 |                      7.1805 |      0.2058 |    nan      |          25.0000 |
| seegnex@hplif+zs4               |      0.1171 |  28676032.0000 | 251528039.1121 |        358.2850 | 551937008.0000 |       2538.9102 |                      7.0863 |      0.1870 |    nan      |          25.0000 |
| seegnex@hplif+zs4+adabn         |      0.1193 |  28676032.0000 | 256526966.7031 |        362.7840 | 551937008.0000 |       2538.9102 |                      6.9984 |      0.1891 |    nan      |          25.0000 |
| seegnex@s_hplif_afe+zs          |    nan      | 543745008.0000 |   1856391.3158 |       2502.8978 | 551937008.0000 |       2538.9102 |                      1.0144 |      0.2266 |    nan      |          25.0000 |
| seegnex@s_hplif_afe+zs+adabn    |    nan      | 543745008.0000 |   1869922.5460 |       2502.9100 | 551937008.0000 |       2538.9102 |                      1.0144 |      0.2283 |    nan      |          25.0000 |
| seegnet@plif+eamu+adabn         |      0.2510 |  17155968.0000 |     64253.1554 |         78.9753 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+eamu               |      0.2517 |  17155968.0000 |     64435.4906 |         78.9754 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+zs                 |      0.2521 |  17155968.0000 |     64537.5956 |         78.9755 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+eaz                |      0.2527 |  17155968.0000 |     64700.5986 |         78.9757 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+zs4                |      0.2528 |  17155968.0000 |     64721.6270 |         78.9757 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+zs+adabn           |      0.2585 |  17155968.0000 |     66175.1629 |         78.9770 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+ea                 |      0.2598 |  17155968.0000 |     66510.9366 |         78.9773 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+eaz+adabn          |      0.2598 |  17155968.0000 |     66519.9044 |         78.9773 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+zs4+adabn          |      0.2608 |  17155968.0000 |     66762.3414 |         78.9775 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |
| seegnet@plif+ea+adabn           |      0.2670 |  17155968.0000 |     68351.2569 |         78.9790 |  17024992.0000 |         78.3150 |                      0.9916 |    nan      |    nan      |          25.0000 |

## Overfitting at best epoch (train / val / test)

_(no data yet)_

## Friedman + Nemenyi (complete cases)

```
{
 "friedman_stat": 287.3470104157791,
 "friedman_p": 1.0029497374009947e-29,
 "k": 66,
 "N": 25,
 "avg_ranks": {
  "eegnex+eaz+adabn": 15.78,
  "eegnex+bench": 16.14,
  "seegnex@hplif+zs4+adabn": 19.28,
  "eegnex+zs4": 19.72,
  "seegnex@hplif+zs4": 19.72,
  "eegnex+eaz": 19.8,
  "eegnex+zs4+adabn": 21.4,
  "eegnex+zs+adabn": 22.28,
  "seegnex@s_hplif+eaz+adabn": 23.46,
  "seegnex@s_hplif+zs4+adabn": 24.66,
  "seegnex@s_hplif+eaz": 24.94,
  "eegnex+ea+adabn": 25.34,
  "eegnex+zs": 25.4,
  "seegnex@s_hplif+zs4": 26.18,
  "eegnex+ea": 26.26,
  "seegnex@s_hplif_spk+zs+adabn": 27.54,
  "seegnex@s_hplif+zs+adabn": 27.6,
  "seegnex@s_hplif_spk+zs": 28.56,
  "seegnex@s_hplif+zs": 28.64,
  "seegnex@s_hplif+eamu": 30.1,
  "eegnex+eamu+adabn": 30.82,
  "seegnex@hplif+zs+adabn": 30.9,
  "seegnex@hplif+zs": 31.22,
  "eegnex+eamu": 31.46,
  "seegnex@s_hplif_noinit+zs+adabn": 32.48,
  "eegnet_official+zs4+adabn": 32.84,
  "eegnet_official+ea": 32.92,
  "atcnet+bench": 33.0,
  "seegnex@hplif+ea": 33.58,
  "seegnex@s_hplif_noinit+zs": 33.7,
  "eegnet_official+zs4": 33.86,
  "eeg_deformer+bench": 33.98,
  "eegnet_official+zs": 34.02,
  "seegnex@s_hplif+eamu+adabn": 34.4,
  "atcnet@noattn+bench": 34.6,
  "s4erp_dw@f16d8+bench": 34.64,
  "eegnet_official+bench": 34.74,
  "atcnet@tcn8+bench": 34.96,
  "s4erp+bench": 34.98,
  "eegnet_official+zs+adabn": 35.72,
  "seegnet@plif+zs": 36.3,
  "seegnex@hplif+ea+adabn": 36.4,
  "seegnex@s_hplif+ea": 36.54,
  "atcnet@nwin1+bench": 36.76,
  "seegnet@plif+zs+adabn": 37.14,
  "eegnet_official+ea+adabn": 37.44,
  "dualcd_s4_flatten+bench": 38.24,
  "seegnex@s_hplif_afe+zs+adabn": 38.84,
  "s4erp_dw@f64d2+bench": 38.86,
  "seegnex@s_hplif_afe+zs": 38.96,
  "dualcd_transformer+bench": 39.16,
  "dualcd_s4_timepatch+bench": 39.4,
  "deepconvnet+bench": 40.02,
  "seegnex@s_hplif+ea+adabn": 40.02,
  "seegnet@plif+zs4+adabn": 40.18,
  "seegnet@plif+zs4": 41.48,
  "atcnet_s4@noattn+bench": 42.36,
  "seegnet@plif+eaz+adabn": 42.44,
  "s4erp_dw@f16d2p+bench": 42.9,
  "fbcnet+bench": 43.14,
  "atcnet_s4+bench": 44.46,
  "seegnet@plif+ea+adabn": 45.08,
  "seegnet@plif+eaz": 46.42,
  "seegnet@plif+eamu": 47.88,
  "seegnet@plif+ea": 48.94,
  "seegnet@plif+eamu+adabn": 50.02
 },
 "critical_difference": 22.365186053125207,
 "alpha": 0.05,
 "higher_is_better": true,
 "dropped_models": []
}
```
