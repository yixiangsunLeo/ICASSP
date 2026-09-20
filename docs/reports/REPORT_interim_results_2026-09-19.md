---
title: "REPORT — 主实验中期结果（zs 臂，09-19 12:20 快照）"
tags:
  - "#modality/eeg"
  - "#method/snn"
  - "#pipeline/5_dl_model"
created: "2026-09-19"
updated: "2026-09-19 12:30"
status: "interim"
---

# 主实验中期结果（zs 臂，2026-09-19 12:20 落盘快照）

## 核心结论

- **协议可比性已被数据证实**：同折重跑的 EEGNeX / EEGNet 在 2C 上与 bench 快照**逐被试完全相同**（51 人 51 平，Δ = 0.0000）。新条目与 18 个 ANN 的配对比较没有任何口径差。
- **已完整落盘、可进论文的行**：2C 的 EEGNeX、EEGNet、S-EEGNet（各 10/10 cell）；3C 的 EEGNeX、EEGNet、S-EEGNeX-D（各 11/11）。
- **趋势（未完整，不是结论）**：三个队列上 SNN 都略低于 EEGNeX——S-EEGNet 比 EEGNet −0.5 pp（2C，完整）；S-EEGNeX-D −0.8 pp（2C，6/10）、−2.1 pp（3C，完整，n=11 不显著）；S-EEGNeX-S −1.8 pp（2C，4/10，41 人）。能耗口径不变：D 版 ≈ 1/7、S 版 ≈ 1/17 于 EEGNeX。
- 无失败 cell；eaz 臂已按 D-035 取消，本报告只有 zs。

## 目标

回答「现在出了哪些结果」，并把已完整的行与仍在跑的行分开，避免半截测试集被当成最终数字。

## 方法 / 协议

`scripts/analyze_snn_round.py --dataset wbci_shu --run zs=<run> [--bench <2C bench>]`，输出在
`outputs/analysis/snn_round/{wbci2c,wbci3c}_zs_interim_2026-09-19_1220/`。协议 = bench 配方（Adam 1e-4 / 128 / patience 25 / val macro-F1 / seed 0；2C split 41+42 五折、3C LOSO 11 折）；zs = 逐 trial 逐通道 z-score，无被试对齐。逐被试配对 Wilcoxon；Acc = 被试均值 ± SD（ddof=1）。

## 结果

### 2C（51 人）

| 条目 | cell | Acc（%） | 相对参照（逐被试配对） |
|:---|:---|:---|:---|
| EEGNeX（重跑） | **10/10** | 83.35 ± 11.54 | = bench（51 平） |
| EEGNet（重跑） | **10/10** | 82.53 ± 12.04 | = bench |
| **S-EEGNet**（`seegnet@plif`） | **10/10** | 82.03 ± 12.07 | vs EEGNet **−0.49 pp**，p = 0.0009，胜/平/负 16/3/32 |
| S-EEGNeX-D（`seegnex@hplif`） | 6/10 ⏳ | 82.53 ± 11.95 | vs EEGNeX −0.82 pp，p = 0.0006，14/2/35（split 41 完整 + split 42 fold0） |
| S-EEGNeX-S（`seegnex@s_hplif`） | 4/10 ⏳ | 81.61 ± 12.16（41 人） | vs EEGNeX（同 41 人 83.45）**−1.84 pp**，CI95 [−2.83, −0.84]，p = 0.0002，13/0/28 |

参照：bench 前几名 ATCNet@noattn 83.56、EEGNeX 83.35、ATCNet 83.28、ATCNet@nwin1 83.16、DeepConvNet 82.76、EEG-Deformer 82.71、EEGNet 82.53。AdaBN 评估列（探索性）：EEGNeX +0.18 pp（p = 0.02），S 版 +0.09 pp（n.s.）。

### 3C（11 人，LOSO）

| 条目 | cell | Acc（%） | 相对参照 |
|:---|:---|:---|:---|
| EEGNeX | **11/11** | 69.41 ± 13.22 | 旧 patience-100 配方 68.78 ± 13.72（`RESULTS_TRUTH`）→ 配方改动对 EEGNeX 无影响 |
| EEGNet | **11/11** | 66.05 ± 13.05 | — |
| **S-EEGNeX-D** | **11/11** | 67.34 ± 12.84 | vs EEGNeX **−2.07 pp**，p = 0.13，4/0/7（n = 11，功效低） |
| S-EEGNeX-S | 3/11 ⏳ | — | 太少，不报 |
| S-EEGNet | 0/11 | — | 排在 S 版之后 |

能耗（3C，中位被试）：D 版 673 µJ vs EEGNeX 4567 µJ（**6.8×**，n = 11）；S 版 272 µJ（**16.8×**，n = 3）。

### SHU（25 人）

仍只有先导 split 41 的 25 cell（`shu_pilot_lr1e4`）；双划分作业 71034 排在 2C-D 作业之后，预计今天 17:30 上卡、09-20 中午出齐。

## 分析

1. 2C 上 EEGNeX / EEGNet 重跑与 bench 逐被试全等，说明数据、划分、配方、随机种子链路完全确定；论文里「同协议」不是口头说法。
2. 三个队列的方向一致：S-EEGNet 与 EEGNet 差 0.5 pp（显著但小）；D 版与 EEGNeX 差 0.8–2 pp；S 版差得更多。2C 的 S 版目前点估计 −1.84、CI 上界 −0.84，**按预注册 δ = 1 pp 的非劣性目前不成立**——但只有 4/10 cell、41/51 人，等 10/10 再下结论（AGENTS §11：2C 出齐前不写「非劣」结论）。
3. 主模型仍按预注册（S 版主、D 版兜底），**不按这些测试数字换主模型**；论文如何选「Ours」行由学长定（D-033 的 `--ours` 开关）。
4. AdaBN 单独几乎不动，配合 D-035 砍掉 EA 臂，论文不写对齐。

## 与前序关系

SHU 先导（D −0.48 n.s.、S −1.38 p = 0.04、S-EEGNet −0.94）与本快照同向。3C 的 EEGNeX 新旧配方一致，说明 D-034 补跑 ATCNet / Deformer / FBCNet 主要是为口径统一，预计数字变化不大。

## 下一步

- 2C S 版 4 个 cell（71230 / 71232，每个 4.5–7.7 h）→ 09-20 清晨；D 版 3 个 → 今天 17:30；3C S 版 8 个 + S-EEGNet 11 个 → 09-20 03:00；SHU zs → 09-20 中午；外部基线 → 09-21 中午。
- 出齐后用同一命令重跑分析（去掉 `_interim`），再 `build_dataset_tables.py --splice`。

## 文件清单

- `outputs/analysis/snn_round/wbci2c_zs_interim_2026-09-19_1220/{REPORT.md,per_model.csv,paired_tests.csv,energy.csv,subject_matrix.csv}`
- `outputs/analysis/snn_round/wbci3c_zs_interim_2026-09-19_1220/`（同上）
- 输入：`outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1/cells/*/result.json`（40）、`snn_wbci3c_zs_lr1e4_v1`（36）、bench 快照 `results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1`
