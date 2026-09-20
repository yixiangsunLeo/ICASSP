---
title: "三队列对照表（2026-09-20 13:10，未完成格子为 —）"
tags:
  - "#modality/eeg"
  - "#method/snn"
created: "2026-09-20"
updated: "2026-09-20 13:10"
status: "active"
---

# 三队列对照表（zs 臂；未完成用 —）

数字全部来自脚本，没有手算。未齐的模型不填部分被试的均值。

- 分析：`outputs/analysis/snn_round/{wbci2c,shu,wbci3c}_zs_2026-09-20_1300/per_model.csv`
- 组表：`outputs/analysis/dataset_tables_2026-09-20/`（`build_dataset_tables.py --include eegnex --ours seegnex@s_hplif`；覆盖不齐的行已从 `--snn-*` 里剔除）
- 2C / SHU 的 EEGNet / FBCNet / ATCNet / EEG-Deformer / EEGNeX：bench 快照（与本轮 zs 重跑逐被试一致）
- 3C：本轮 LOSO、patience 25（配方对齐），**不是**旧 patience-100 那一版
- 主指标：2C / 3C = Acc ± SD（被试间）；SHU = BAcc ± SD。单位 %；AUC 在 [0,1]
- **Ours 预注册主模型仍是 S-EEGNeX-S**（`seegnex@s_hplif`）。D / S-EEGNet 只并列，不按测试集改主模型

未填：2C TCFormer 3/10；3C EEGPT 3/11；SHU `seegnex@s_plif` 3/10；MDGEEG / EEGMoE 本轮没排。

## WBCIC-SHU 2C（51 人，chance 50%）

| 模型 | Acc ± SD | Macro-F1 | AUC | cell |
|:---|---:|---:|---:|:---|
| EEGNet | 82.53 ± 12.04 | 82.50 | 0.892 | 10/10 |
| FBCNet | 69.96 ± 10.79 | 69.50 | 0.767 | 10/10 |
| ATCNet | 83.28 ± 12.00 | 83.23 | 0.899 | 10/10 |
| EEGNeX | 83.35 ± 11.54 | 83.33 | 0.901 | 10/10 |
| LaBraM | 81.91 ± 11.78 | 81.85 | 0.888 | 10/10 |
| EEGPT | 78.99 ± 11.24 | 78.93 | 0.862 | 10/10 |
| EEG-Deformer | 82.71 ± 11.94 | 82.64 | 0.895 | 10/10 |
| TCFormer | — | — | — | 3/10 |
| CBraMod | 82.51 ± 11.58 | 82.45 | 0.894 | 10/10 |
| MDGEEG | — | — | — | 未排 |
| EEGMoE | — | — | — | 未排 |
| S-EEGNeX-S（预注册 Ours） | 81.42 ± 12.04 | 81.36 | 0.883 | 10/10 |
| S-EEGNeX-D | 82.71 ± 11.86 | 82.68 | 0.894 | 10/10 |
| S-EEGNet | 82.03 ± 12.07 | 82.01 | 0.888 | 10/10 |

## SHU 2022（25 人，主指标 BAcc）

| 模型 | BAcc ± SD | Macro-F1 | AUC | cell |
|:---|---:|---:|---:|:---|
| EEGNet | 61.53 ± 10.70 | 61.26 | 0.656 | 10/10 |
| FBCNet | 59.27 ± 6.67 | 57.94 | 0.633 | 10/10 |
| ATCNet | 61.95 ± 11.25 | 61.67 | 0.663 | 10/10 |
| EEGNeX | 65.47 ± 9.99 | 64.78 | 0.705 | 10/10 |
| LaBraM | 64.22 ± 10.15 | 63.85 | 0.686 | 10/10 |
| EEGPT | 56.72 ± 5.74 | 56.55 | 0.597 | 10/10 |
| EEG-Deformer | 61.43 ± 9.54 | 60.81 | 0.661 | 10/10 |
| TCFormer | 61.72 ± 9.82 | 61.15 | 0.667 | 10/10 |
| CBraMod | 64.59 ± 9.98 | 64.20 | 0.695 | 10/10 |
| MDGEEG | — | — | — | 未排 |
| EEGMoE | — | — | — | 未排 |
| S-EEGNeX-S（预注册 Ours） | 63.59 ± 10.64 | 62.72 | 0.686 | 10/10 |
| S-EEGNeX-D | 64.55 ± 9.90 | 63.49 | 0.695 | 10/10 |
| S-EEGNet | 60.46 ± 10.96 | 59.88 | 0.646 | 10/10 |
| S-EEGNeX-S@LIF（H3） | 64.25 ± 9.99 | 63.52 | 0.691 | 10/10 |
| S-EEGNeX-S@PLIF（H3） | — | — | — | 3/10 |

SHU H3 的 LIF 行 Acc = 64.23 ± 9.98（`seegnex@s_lif+zs`）；上表 BAcc 64.25 来自同一行的 `balanced_accuracy`。

## WBCIC-SHU 3C（11 人 LOSO，chance 33.3%）

| 模型 | Acc ± SD | Macro-F1 | AUC | cell |
|:---|---:|---:|---:|:---|
| EEGNet | 66.05 ± 13.05 | 65.85 | 0.823 | 11/11 |
| FBCNet | 52.91 ± 8.30 | 51.35 | 0.727 | 11/11 |
| ATCNet | 68.86 ± 13.24 | 68.03 | 0.853 | 11/11 |
| EEGNeX | 69.41 ± 13.22 | 69.12 | 0.851 | 11/11 |
| LaBraM | 67.19 ± 13.47 | 66.92 | 0.839 | 11/11 |
| EEGPT | — | — | — | 3/11 |
| EEG-Deformer | 68.10 ± 15.13 | 67.66 | 0.847 | 11/11 |
| TCFormer | 68.10 ± 12.65 | 67.26 | 0.849 | 11/11 |
| CBraMod | 68.08 ± 14.51 | 67.79 | 0.846 | 11/11 |
| MDGEEG | — | — | — | 未排 |
| EEGMoE | — | — | — | 未排 |
| S-EEGNeX-S（预注册 Ours） | 63.59 ± 12.07 | 62.78 | 0.816 | 11/11 |
| S-EEGNeX-D | 67.34 ± 12.84 | 67.00 | 0.839 | 11/11 |
| S-EEGNet | 66.62 ± 12.43 | 66.24 | 0.832 | 11/11 |

## 还在跑

71034 SHU `seegnex@s_plif`；71213 2C TCFormer；71222 3C EEGPT。齐了再跑同一条 `analyze` + `build_dataset_tables` 即可把 `—` 换成数字。

## 文件

- `outputs/analysis/snn_round/wbci2c_zs_2026-09-20_1300/{per_model.csv,REPORT.md}`
- `outputs/analysis/snn_round/shu_zs_2026-09-20_1300/{per_model.csv,REPORT.md}`
- `outputs/analysis/snn_round/wbci3c_zs_2026-09-20_1300/{per_model.csv,REPORT.md}`
- `outputs/analysis/dataset_tables_2026-09-20/{dataset_tables.md,dataset_tables_long.csv,tab_2c.tex,tab_shu.tex,tab_3c.tex}`
