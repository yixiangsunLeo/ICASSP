---
title: "跨被试运动想象实验 · 梳理简报（给学长开会用，2026-09-18）"
tags:
  - "#modality/eeg"
created: "2026-09-18"
updated: "2026-09-18"
status: "archived"
---

# 跨被试运动想象实验 · 梳理简报（给学长开会用）

> **路径说明（2026-09-18 17:30 迁移后）**：本文写于迁移前。文中 `SYX/eeg-mi-online/` 现为 `SYX/ICASSP/`（本仓库根），
> `1_session_drift/ 2_baseline/ 4_experiments/` 等跨 session 结果区已移到 `SYX/_archive/eeg-mi-online_legacy_2026-09-18/`，
> `SYX/eeg-fm-bench/` 现为 `SYX/Benchmark/`。活文档版见 `docs/PRIOR_RESULTS.md`。
>
> 日期 2026-09-18。回答学长的三个问题：① 代码还在不在 ② 跑了几个数据集、每个数据集对应跑了几个模型 ③ 结果是什么。
> 所有数字都从备份里的原始结果表（`results_snapshot_2026-09-13/experiments/*/tables/*.csv`、`cells/**/result.json`）读出，未做任何改动。

---

## 0. 一页结论

| 问题 | 答案 |
|:---|:---|
| 代码还在吗 | **在，已恢复。** 2026-09-13 打包备份 `/share/home/yuan/SYX_backup_2026-09-13.tar.gz`（125 MB，sha256 校验通过），含全部代码、git 历史、结果表；GitHub 镜像 `stevencummings341-prog/EEG`（main）。2026-09-18 已解压回 `/share/home/yuan/SYX/eeg-mi-online/`：15,588 个条目与包内清单一致，`git rev-parse HEAD` = `1e183c9`（与 LEGACY 记录一致），工作区仅 `.gitignore` 一处 09-13 清理时留下的未提交改动。 |
| 跑了几个数据集 | **3 个独立被试队列**（全部运动想象、全部跨被试）：WBCIC-SHU 2C（51 人）、SHU 2022（25 人）、WBCIC-SHU 3C（11 人）。另有 2C/SHU 的跨 session 阶段（Phase 0–2c）。 |
| 每个数据集跑了几个模型 | 2C：**18 个条目**；SHU 2022：**18 个条目**（同一名单）+ 旧协议 9 个；3C：**7 个模型 × 2 套训练配方** + 对抗训练/λ 扫描/被试内天花板。 |
| 哪个模型最强 | **EEGNeX**：SHU 第 1（65.5%，领先 3.5pp）、2C 并列第 1（83.4%，与第 1 差 0.2pp 不显著）、3C 论文配方第 1（69.9%）。参数 58 k，GPU 时间约为 ATCNet 的 1/4。 |
| 已经证伪的方向 | ATCNet×S4ERP 两个融合方向（显著更差）、ATCNet 三个消融（差异不显著）、域对抗训练（三条负证据）。 |
| 数据还在吗 | 2C 与 SHU 的 processed npz 在 `/share/workspace2/moto_imagination/{WBCIC_SHU,SHU}/processed/`，可直接用；3C processed 数据已删，脚本 7 分钟可重生成。**checkpoint 与逐 trial 预测已全部删除**（41 G + 11 G），要用需重训。 |
| ICASSP | ICASSP 2027 截稿**已延期**到 2026-09-23 23:59 AoE = **北京时间 9 月 24 日 20:00**（官网明示不再延期）。4 页正文 + 1 页参考文献，每位作者需 ORCID。 |

---

## 1. 数据集（3 个跨被试队列）

| 队列 | 被试 | 通道 × 采样点 | 类别 | 总 trial | 被试划分协议 | 出处 |
|:---|---:|:---|:---|---:|:---|:---|
| **WBCIC-SHU 2C** | 51（sub-001..051） | 58 × 1000 @ 250 Hz | 2（左/右手抓握） | 29,600（148 个 ok session） | 5 折被试 k-fold，6:2:2，划分 seed 41/42，每个被试恰好被测 1 次/seed | Yang et al., *Sci Data* 12:488 (2025) |
| **SHU 2022** | 25 | 32 × 1000 @ 250 Hz | 2（左/右手） | 11,988（125 session） | 同上（15:5:5） | Ma et al. 2022 |
| **WBCIC-SHU 3C** | 11（原编号 52–62） | 58 × 1000 @ 250 Hz | 3（左/右手/双脚） | 9,899（33 session） | LOSO 11 折，训练 ses-01+02 / 验证 ses-03 或验证=被试 | 同 2C 论文；2C 与 3C 是**互斥**队列 |

预处理（全部相同口径）：去 EOG/ECG → Pz 重参考并移除 → 0.5–40 Hz → 50 Hz 陷波 → 0–4 s → 250 Hz；训练时逐 trial z-score（fit-free，不跨划分）。
数据事实：2C 类别完美平衡（14800:14800）；SHU 67/125 个 session 不平衡（需并报 balanced accuracy）；SHU 作者自己的跨 session 准确率 53.7%，接近 chance，属地板效应。

---

## 2. 模型清单（每个数据集跑了什么）

### 2.1 2C 与 SHU 2022：同一张 18 条目名单（run `wbci_2c_cross_subject_bench_v1` / `shu_cross_subject_bench_v1`，各 180 cells，全部完成）

| 组 | 条目 | 来源 | 参数 @58ch |
|:---|:---|:---|---:|
| 已发表（官方代码） | `eegnet_official`、`eegnex`、`eeg_deformer`、`atcnet` | Lawhern 2018 / Chen 2024 / Ding 2024 / Altaheri 2023 | 3.0 k / 58.8 k / 1.61 M / 114.6 k |
| 经典 | `deepconvnet`、`fbcnet` | Schirrmeister 2017 / Mane 2021 | 302.6 k / 20.7 k |
| 学长的 S4/DINO 包 | `s4erp`、`dualcd_s4_flatten`、`dualcd_transformer`、`dualcd_s4_timepatch` | ERP-Benchmark 源头 | 1.37 M / 66.9 M / 67.9 M / 4.48 M |
| 融合 A（学长指定：空间滤波换 ATCNet 的 depthwise） | `s4erp_dw@f64d2`、`@f16d8`、`@f16d2p` | 自研 | 426 k / 425 k / 407 k |
| 融合 B（共享因果 SSM 替 5 个 TCN） | `atcnet_s4`、`atcnet_s4@noattn` | 自研 | 27.8 k / 25.6 k |
| ATCNet 消融 | `atcnet@tcn8`、`@nwin1`、`@noattn` | 配置变体 | 41.2 k / 38.5 k / 103.6 k |

训练配方（18 条目统一）：Adam 1e-4，batch 128，无 scheduler，patience 25（监控 macro-F1），最多 500 epoch，单训练 seed，划分 seed 41/42。

### 2.2 WBCIC-SHU 3C（LOSO 11 折）

| run | 配方 | 模型 | cells |
|:---|:---|:---|---:|
| `paper_baseline_3c_821_v1` | 论文配方（Adam 1e-4 / batch 128 / patience 100） | EEGNeX、EEG-Deformer、ATCNet、S4ERP、EEGNet、DualCD-Transformer、DualCD-S4-Flatten | 77（7×11，完整） |
| `foundation_cross_subject_3class_v1` | 我们的配方（AdamW 1e-3 / batch 64 / cosine / patience 25） | ATCNet、EEGNeX、EEG-Deformer、S4ERP、S4ERP-Adv、EEGNet、DualCD-S4-Flatten（仅 7 折） | 73 |
| `adv_821recipe_3c_v1` / `atcnet_adv_3class_v1` / `foundation_cross_subject_adv_l{01,05,10,20}_w5` | 域对抗（S4ERP-Adv、ATCNet-Adv、λ 扫描） | — | 11 / 22 / 4×11 |
| `wbci_3c_within_ceiling_v1` | 被试内（每 session 内 5 折） | EEGNet、ATCNet | 165×2 |

### 2.3 SHU 2022 旧协议（`shu_cross_subject_5fold_v1`，225 cells，保留作历史对照，不与新表混读）
9 模型：eegnex、s4erp、eeg_deformer、atcnet、eegnet_official、dualcd_transformer、dualcd_s4_flatten、deepconvnet、fbcnet（16:4:5 划分、监控 accuracy）。

### 2.4 跨 session 阶段（Phase 0–2c，2C 与 SHU 各一套，2026-06~07 完成）
漂移诊断、within/cross-session 基线（EEGNet / DeepConvNet / FBCNet，5 seeds）、多源训练、无学习对齐（session z-score / EA / 滤波器组等）、原型漂移分析。结论摘要见 `results.md`；与本次跨被试主线无代码依赖。

---

## 3. 核心结果

### 3.1 WBCIC-SHU 2C（51 被试，mean ± std over subjects，best-val checkpoint）

| 名次 | 模型 | Acc | Macro-F1 | AUROC | 参数 | GPU-h（10 cells） |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | atcnet@noattn | 83.56 ± 11.79 | 83.54 | 0.901 | 103.6 k | 18.5 |
| 2 | **eegnex** | **83.35 ± 11.54** | 83.33 | 0.901 | **58.8 k** | **4.0** |
| 3 | atcnet | 83.28 ± 12.00 | 83.23 | 0.899 | 114.6 k | 17.1 |
| 4 | atcnet@nwin1 | 83.16 | 83.13 | 0.899 | 38.5 k | 5.6 |
| 5 | atcnet@tcn8 | 83.12 | 83.08 | 0.898 | 41.2 k | 8.3 |
| 6 | deepconvnet | 82.76 | 82.69 | 0.895 | 302.6 k | 4.4 |
| 7 | eeg_deformer | 82.71 | 82.64 | 0.895 | 1.61 M | 3.1 |
| 8 | eegnet_official | 82.53 | 82.50 | 0.892 | 3.0 k | 5.1 |
| 9 | s4erp | 81.33 | 81.30 | 0.882 | 1.37 M | 7.4 |
| 10–12 | s4erp_dw@f64d2 / @f16d8 / @f16d2p | 80.68 / 80.61 / 80.27 | | | ~0.42 M | 1.2–2.1 |
| 13 | dualcd_transformer | 79.17 | | | 67.9 M | 51.6 |
| 14 | dualcd_s4_flatten | 79.01 | | | 66.9 M | 45.0 |
| 15 | dualcd_s4_timepatch | 76.30 | | | 4.48 M | 106.4 |
| 16 | atcnet_s4 | 74.99 | | | 27.8 k | 5.6 |
| 17 | fbcnet | 69.96 | | | 20.7 k | 3.0 |
| 18 | atcnet_s4@noattn | 63.28 | | | 25.6 k | 4.0 |

Friedman：χ² = 612，p ≈ 10⁻¹¹⁹，k = 18，N = 51，Nemenyi CD = 3.69 个平均名次。前 8 名（83.6 → 82.5）之间差 1.0pp，配对检验 atcnet vs atcnet@noattn 差 0.29pp（p = 0.09，Holm 后不显著）。

### 3.2 SHU 2022（25 被试）

| 名次 | 模型 | Acc | Macro-F1 | AUROC | 参数 | GPU-h |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | **eegnex** | **65.49 ± 9.95** | 64.78 | 0.705 | 57.1 k | 1.6 |
| 2 | atcnet | 61.96 ± 11.23 | 61.67 | 0.663 | 113.7 k | 5.2 |
| 3 | s4erp_dw@f16d8 | 61.68 | 61.42 | 0.660 | 421.8 k | 0.6 |
| 4 | atcnet@tcn8 | 61.57 | | | 40.4 k | 2.3 |
| 5 | eegnet_official | 61.52 | 61.26 | 0.656 | 2.6 k | 0.4 |
| 6–9 | atcnet@noattn / eeg_deformer / atcnet@nwin1 / s4erp | 61.43 / 61.43 / 61.35 / 61.27 | | | | |
| 10–14 | deepconvnet / dualcd_s4_timepatch / dualcd_transformer / s4erp_dw@f64d2 / dualcd_s4_flatten | 60.96 / 60.83 / 60.52 / 60.52 / 60.39 | | | | |
| 15–18 | atcnet_s4@noattn / atcnet_s4 / s4erp_dw@f16d2p / fbcnet | 60.14 / 59.85 / 59.74 / 59.24 | | | | |

Friedman p = 8×10⁻⁵，CD = 5.27。EEGNeX 领先第 2 名 3.5pp，是三张表里唯一"一家独大"的格子。旧协议 run 上 EEGNeX 同样第 1（64.5%），结论稳定。chance band 51.4–53.7%，整体地板效应。

### 3.3 WBCIC-SHU 3C（11 被试 LOSO，两套配方，排名会随配方变化）

| 模型 | 论文配方 Acc | 我们的配方 Acc | DSGNet 论文报的该模型数字 |
|:---|---:|---:|---:|
| **EEGNeX** | **69.91 ± 13.56** | 68.78 ± 13.72 | 64.88 |
| EEG-Deformer | 68.49 ± 14.83 | 67.81 ± 13.97 | 65.29 |
| ATCNet | 68.39 ± 13.26 | **70.85 ± 15.20** | 68.34 |
| S4ERP | 68.26 ± 14.37 | 64.68 ± 13.56 | — |
| EEGNet | 66.36 ± 13.05 | 67.22 ± 12.79 | 64.92 |
| DualCD-Transformer | 65.13 ± 13.12 | — | — |
| DualCD-S4-Flatten | 64.54 ± 13.56 | 69.87（仅 7 折，不可比） | — |
| S4ERP-Adv | 68.54（另一 run） | 67.60 | — |
| DSGNet（论文自报） | — | — | 68.56 |

我们复现的 EEGNeX / EEG-Deformer 分别比 DSGNet 论文表里的数字高 5.0 / 3.2pp（ATCNet 一致），即 **DSGNet 论文的 baseline 被低估**。
所有模型在 3C 上都严重过拟合（train 80–98% vs test 65–70%，gap 15–33pp），EEGNet 最小、S4ERP 最大。

### 3.4 已经结案的方向（有配对检验，不必再讨论）

| 方向 | 证据 | 结论 |
|:---|:---|:---|
| 融合 A：S4ERP 换 ATCNet depthwise 空间前端（学长 08-26 指定） | 2C 上三个变体 vs s4erp：−0.64 / −0.72 / −1.05pp，Holm 后**全部显著更差**；SHU 上不显著 | 参数减 70% 但精度略降，**未能超过基座** |
| 融合 B：共享因果 SSM 替 5 个 TCN | 2C：atcnet_s4 −8.3pp（p ≈ 10⁻⁹）；@noattn −20.3pp | **显著更差**，方向失败 |
| ATCNet 消融 | @noattn +0.29、@tcn8 −0.16、@nwin1 −0.12pp，全部不显著 | 注意力、TCN 宽度、5 窗集成都**不是** ATCNet 有效的原因；三分之二的参数可砍 |
| 域对抗（S4ERP-Adv / ATCNet-Adv / λ 扫描） | 论文配方下 +0.28pp；ATCNet 上 −4.22pp；λ 0.1→2.0 只差 0.56pp | **关闭** |
| 难被试 | 3C sub-011 被试内也只有 34.7%（chance 33.3%） | 是 BCI illiteracy，不是迁移失败 |

---

## 4. 代码与数据现状

| 项 | 位置 | 状态 |
|:---|:---|:---|
| 备份包 | `/share/home/yuan/SYX_backup_2026-09-13.tar.gz` + `.sha256` | 校验通过（OK） |
| 恢复目录 | `/share/home/yuan/SYX/eeg-mi-online/` | 已恢复并核对（15,588 条目，HEAD `1e183c9`） |
| GitHub | `stevencummings341-prog/EEG` main = `1e183c9` | 与备份一致 |
| 模型代码 | `code/models/{paper_baselines,atcnet,eeg_foundation,...}` | EEGNeX 为官方 Keras 逐层移植的 PyTorch 版（`paper_baselines/eegnex_official_torch.py`，58,770 参数） |
| 训练框架 | `code/run.py` + `code/experiments/cross_subject_protocols.py` + `code/training/e2e_trainer.py` | 被试级折由 split_seed 确定性生成 → **新模型可与已有 18 个模型逐被试配对比较** |
| 汇总/统计 | `scripts/summarize_cross_subject.py`、`code/summaries/cross_subject_bench.py` | 配对 Wilcoxon + Holm、Friedman + Nemenyi CD 图 |
| Slurm | `scripts/slurm/{bench_gpu.sbatch,submit_bench.sh}` | 4 卡分组 + 48 h 自动接力 |
| processed 数据 | `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean`（2C）、`.../SHU/processed/npz_clean` | 在，未动 |
| 3C processed | 已删 | `python scripts/preprocess_wbci_3c.py` 约 7 分钟重生成 |
| checkpoints / predictions.npz | 已删（41 G / 11 G） | 要做 trial 级分析或复用权重需重训 |
| 环境 | conda `mi_torch_cu118`（torch 2.7.1+cu118） | 可用 |
| 集群 | QOS 上限 4 GPU / 48 h / 作业；今日 5 个 GPU 节点 40 卡中仅约 4 张空闲 | 排队要留余量 |

---

## 5. 开会需要学长拍板的问题（我不替学长决定）

1. **EEGNeX 怎么改、加什么。** 现有事实：EEGNeX 在 3 个队列都是最强或并列最强、最便宜（58 k 参数，2C 上 10 cells 4 GPU-h）；瓶颈是跨被试过拟合（train–test gap 21pp），不是容量。
2. **SNN 套在哪一层、用什么口径比。** 旧 SNN 文献多只在 BCI IV 2a（9 人）上做；我们的差异化是 87 人 / 3 队列 / 与 18 个 ANN 配对。要不要报能耗（SOPs）。
3. **进论文的数据集与 baseline 范围。** 3 个队列都用还是只用 2C+SHU；ANN baseline 表直接复用现有 18 条目（同折同划分，可配对）。
4. **要不要补 BCI IV 2a**（评审默认期待）。EEG-FM-Bench 项目已把 BNCI2014001 等 5 套 BNCI 数据下载并转成 pkl，但接入本框架需要写 loader。
5. **单训练 seed 的问题**：现有表是 1 个训练 seed × 2 个划分 seed；新实验是否加 seed。
6. **时间**：截稿 9/24 20:00（北京）。以 EEGNeX 的成本估，一个 SNN 变体在三个队列上约 25–35 GPU-h（T=4）；4 卡并行下每天能跑 3–4 个变体，写作至少要留 2 天。

---

## 6. 文件索引（恢复完成后均在 `SYX/eeg-mi-online/`）

- 权威结果：`docs/RESULTS_TRUTH.md`（3C）、`results_snapshot_2026-09-13/experiments/{wbci_shu,shu}/*/tables/`（2C / SHU）
- 计划与决定：`docs/PLAN_cross_subject_benchmark_and_fusion.md`（27 条锁定决定）、`docs/ANALYSIS_atcnet_and_adversarial.md`
- 流水：`progress.md`（最新 2026-08-27）、`experiment_log.md`
- 复现：`REPRODUCE.md`、`AGENTS.md`
- 前身/后续项目：`SYX/eeg-fm-bench/`（EEG 基础模型 benchmark，2026-09-14 交接文档 `SYX/HANDOFF_2026-09-14.md`）
