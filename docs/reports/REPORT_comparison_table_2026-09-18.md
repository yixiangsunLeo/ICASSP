---
title: "给学长：对比模型表（已发表基线 + 年份 + 目前跑完的 SNN 结果）— 2026-09-18 晚"
tags:
  - "#modality/eeg"
  - "#method/snn"
  - "#pipeline/5_dl_model"
created: "2026-09-18"
updated: "2026-09-18"
status: "active"
---

# 对比模型表：现在跑完了什么、表里放什么、还缺什么（2026-09-18 19:40）

> 回应学长 09-18 微信：「效果跟 eegnex 差不多就直接写论文 / 到时候不对比这个 eegnex / 再加几个 25、26 年的模型 /
> 先把要对比的模型的表格拿出来 / 之前那个 eegnex 的结果是哪几个数据集 / 列一个表格，直接丢 overleaf，表格格式不要改」。
> 表已放进 `paper/main.tex`（Table 1，`tab:main`），**表格格式（`tabular{lccccc}` + booktabs + 同一组列）没动，只换了行**；
> 每个数字都由 `scripts/build_comparison_table.py` 从结果 CSV 生成，没有手填。

## 0. 核心结论

1. **EEGNeX 之前的结果在 3 个跨被试队列上**：WBCIC-SHU 2C（51 人，58 ch，左/右手）、SHU 2022（25 人，32 ch）、WBCIC-SHU 3C（11 人，58 ch，LOSO）。EEGNeX 三个队列都是第一或并列第一：**83.35 / 65.49 / 69.91**（详见 §4.1）。
2. **这两天跑的实验里，已经完成的只有 SHU 先导（120 cell，09-18 13:10）**。主实验 20 个作业 17:00 才提交，19:30 时 4 组在跑、其余排队，2C 一个 cell 都还没落盘（§4.4）。所以**现在能进表的 SNN 数字只有 SHU（split 41）一列**，2C / 3C 两列是空的。
3. **先导上「效果跟 EEGNeX 差不多」成立于 D 版，S 版差 1.4pp**（SHU，25 人，同折同配方，未对齐）：
   S-EEGNeX-D 64.17 vs EEGNeX 64.65（−0.48pp，p=0.16，n.s.）；S-EEGNeX-S 63.27（−1.38pp，p=0.040）；S-EEGNet 60.56 vs EEGNet 61.50（−0.94pp，p=0.078）。
   能耗：S 版 148 µJ/trial、D 版 358 µJ vs EEGNeX 2539 µJ（低 17× / 7×），S-EEGNet 79 ≈ EEGNet 78（EEGNet 结构不适合脉冲化）。2C 是确认队列，结论以 2C 为准（≥ 09-21 晚出齐）。
4. **overleaf 里的表**：6 个已发表 ANN 基线（按年份 2017 → 2025：DeepConvNet、EEGNet、FBCNet、ATCNet、EEGNeX、EEG-Deformer）+ 3 个 SNN 行；18 个条目里的 12 个自研 / 消融条目**不进论文表**（只在本报告 §4.3）。年份写在模型名后括号里，列结构不变。
5. **「再加几个 25、26 年的模型」**：现有 18 条目里最新的是 EEG-Deformer（IEEE JBHI 2025）。可加的候选 4 个（§7.1：Zhang et al. 2025 的 SNN、CBraMod ICLR 2025、MIRepNet 2025、DSGNet JBHI 2026），**都要写接入代码并占 GPU**；4 张卡到 ~09-21 都被主实验占满，截稿 09-24 20:00，**现实上只够加 1–2 个**。需要学长选。
6. **EEGNeX 那一行我暂时保留**（理由 §5.4）；学长若坚持不比，删一行即可（tex 里有注释标记）。

## 1. 目标

把「我们要对比的模型」按学长的口径列成一张表：只放能进论文的对比对象，每个模型标出版年份，填上目前**已经落盘**的结果，放进 overleaf 且不改表格格式；同时列出主实验进度与还缺的格子。

## 2. 方法：表怎么来的

- 新脚本 `scripts/build_comparison_table.py`（CPU，登录节点可跑；不在作业 import 闭包内）读 4 份脚本生成的 CSV，输出到 `outputs/analysis/comparison_table_2026-09-18/`：
  `comparison_long.csv`（每个格子一行，带来源文件与行键）、`comparison_table.md`（本报告 §4.1/4.3）、`pilot_paired.md`（§4.2）、`tab_main_rows.tex`（贴进 `paper/main.tex`）。
- 数据来源：

| 列 | 来源 | 协议 |
|:---|:---|:---|
| 2C | `results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1/tables/per_model.csv` | 5 折被试 k-fold，6:2:2，split 41+42（每被试测 2 次取均值），bench 配方 |
| SHU | `results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1/tables/per_model.csv` | 同上 |
| 3C | `results_snapshot_2026-09-13/experiments/wbci_shu/paper_baseline_3c_821_v1/tables/per_cell.csv`（11 折重算 mean / SD，ddof=1） | LOSO 11 折，**论文配方（patience 100）**，其余同 bench |
| SNN | `outputs/analysis/snn_round/shu_pilot_lr1e4/{per_model,energy}.csv`（`analyze_snn_round.py` 输出） | SHU 先导：lr 1e-4，`zs` 臂，**只有 split 41**（每被试测 1 次） |

- 年份口径 = 正式出版年份（卷期年）。EEG-Deformer 线上 2024-11、卷期 IEEE JBHI 29(3):1909–1918 **2025** 年 3 月（dblp / Europe PMC 已核），故写 2025；FBCNet 只有 arXiv（2021）。`paper/refs.bib` 已相应修正（`ding2025deformer`、新增 `mane2021fbcnet`，Zhang 2025 的题目作者已核实补全）。
- 参数量统一按 58 ch / 2 类（2C 口径）：ANN 取 bench 表 `n_params`；SNN 在 CPU 上用 `build_model` 实例化计数（S-EEGNeX 58,850；S-EEGNet 3,028），与 32 ch 先导 `result.json` 里的 57,186 / 2,612 一致。
- 能耗口径：45 nm，MAC 4.6 pJ / AC 0.9 pJ，逐层 dense MACs × 实测发放率；SHU 32 ch、4 s trial；ANN 两行 = 同形状 dense MACs × 4.6 pJ。ATCNet / Deformer / DeepConvNet / FBCNet 没做 MAC 核算 → `--`。

## 3. 协议：三处不能混比的地方（表里都标了）

| 标记 | 问题 | 处理 |
|:---|:---|:---|
| † | SNN 先导只有 split 41（5 cell），bench 行是 split 41+42（10 cell）；同一被试的测试折不同 | 表里只作为"目前的数"，**逐被试配对只用先导自己重跑的 EEGNeX / EEGNet**（§4.2）；主实验补齐 split 42 后重跑脚本即替换 |
| ‡ | 3C 的 ANN 数字来自论文配方（patience 100），主实验 3C 用 bench 配方（patience 25） | 主实验 3C 会重跑 EEGNeX / EEGNet；ATCNet / Deformer 若要同配方需再补 ~30 GPU-h（§7.3） |
| SD | 3C 那个 run 自带的 `per_model.csv` 用总体 SD（÷n），bench 与 SNN 表用样本 SD（÷n−1） | 脚本从 `per_cell.csv` 按 ddof=1 重算，与 `docs/RESULTS_TRUTH.md` 一致（EEGNeX 69.91 ± 13.56） |

其余全部相同：预处理、0–4 s / 250 Hz / 1000 点、逐 trial z-score、Adam 1e-4 / batch 128 / patience 25 / monitor macro-F1 / 单训练 seed、best-val checkpoint、按被试划分、测试被试不参与选模。

## 4. 结果

### 4.1 论文表（已放进 `paper/main.tex` Table 1；acc %，mean ± SD over subjects）

| Model | Year | 2C (51) | SHU (25) | 3C (11) | Params @58ch | µJ/trial (SHU, 32ch) |
|:---|:---:|---:|---:|---:|---:|---:|
| DeepConvNet | 2017 | 82.76 ± 12.07 | 60.96 ± 10.31 | — | 302.6k | — |
| EEGNet | 2018 | 82.53 ± 12.04 | 61.52 ± 10.72 | 66.36 ± 13.05 ‡ | 3.0k | 78 |
| FBCNet | 2021 | 69.96 ± 10.79 | 59.24 ± 6.67 | — | 20.7k | — |
| ATCNet | 2023 | 83.28 ± 12.00 | 61.96 ± 11.23 | 68.39 ± 13.26 ‡ | 114.6k | — |
| **EEGNeX** | 2024 | **83.35 ± 11.54** | **65.49 ± 9.95** | **69.91 ± 13.56** ‡ | 58.8k | 2539 |
| EEG-Deformer | 2025 | 82.71 ± 11.94 | 61.43 ± 9.54 | 68.49 ± 14.83 ‡ | 1.61M | — |
| S-EEGNet (PLIF) | ours | 在跑 | 60.56 ± 11.09 † | 在跑 | 3.0k | 79 |
| S-EEGNeX-D (HetPLIF, T=4) | ours | 在跑 | 64.17 ± 10.12 † | 在跑 | 58.9k | 358 |
| **S-EEGNeX-S (HetPLIF, temporal-native)** | ours | 在跑 | 63.27 ± 11.01 † | 在跑 | 58.9k | 148 |

（overleaf 里保留 1 位小数；空格子是 `\todo{}`，主实验出齐后由脚本重生成。）

### 4.2 先导内部配对（唯一 apples-to-apples 的 SNN vs ANN；SHU，25 人，split 41，lr 1e-4，未对齐）

| 条目 | acc ± SD | balanced acc | macro-F1 | 参数 | GPU-h（5 cell） |
|:---|---:|---:|---:|---:|---:|
| EEGNeX（同折重跑） | 64.65 ± 10.46 | 64.65 | 63.82 | 57,106 | 0.59 |
| S-EEGNeX-D | 64.17 ± 10.12 | 64.24 | 62.97 | 57,186 | 1.46 |
| S-EEGNeX-S | 63.27 ± 11.01 | 63.29 | 62.33 | 57,186 | 3.44 |
| EEGNet（同折重跑） | 61.50 ± 10.49 | 61.52 | 61.17 | 2,610 | 0.16 |
| S-EEGNet | 60.56 ± 11.09 | 60.59 | 59.85 | 2,612 | 0.11 |

| 比较（逐被试 Wilcoxon，n=25） | Δacc (pp) | 95% CI | 胜/平/负 | p |
|:---|---:|:---|:---|---:|
| S-EEGNeX-S vs EEGNeX（预注册 H0-NI） | −1.38 | [−2.96, +0.19] | 7/0/18 | 0.040 |
| S-EEGNeX-D vs EEGNeX | −0.48 | [−1.84, +0.88] | 10/1/14 | 0.162 |
| S-EEGNeX-S vs S-EEGNeX-D（H2） | −0.90 | [−1.93, +0.13] | 8/1/16 | 0.070 |
| S-EEGNet vs EEGNet（B1） | −0.94 | [−1.95, +0.07] | 10/0/15 | 0.078 |
| EEGNeX 重跑 vs bench EEGNeX（sanity） | −0.83 | [−1.52, −0.15] | 7/2/16 | 0.021 |

最后一行说明：同一 EEGNeX 在 split 41 上比 41+42 的均值低 0.8pp——**划分本身就带 ~1pp 的差**，这就是为什么 † 行不能直接和 bench 行相减。
对齐臂（EA + 逐通道 z-score、AdaBN）的先导结论在 `docs/PLAN_spiking_eegnex_icassp2027.md` §0.5 与 `outputs/analysis/snn_round/shu_pilot_lr1e3/`，这里不重复。

### 4.3 完整 18 条目（2C / SHU；含自研与消融，只进报告）

| Model | 2C (51) | SHU (25) | 3C (11) | Params @58ch |
|:---|---:|---:|---:|---:|
| S4ERP（学长 S4 包） | 81.33 ± 12.02 | 61.27 ± 9.52 | 68.26 ± 14.37 ‡ | 1.37M |
| S4ERP-DW (f64d2) | 80.68 ± 12.06 | 60.52 ± 9.13 | — | 426.4k |
| S4ERP-DW (f16d8) | 80.61 ± 11.84 | 61.68 ± 9.17 | — | 425.1k |
| S4ERP-DW (f16d2p) | 80.27 ± 12.04 | 59.74 ± 7.87 | — | 407.0k |
| DualCD-Transformer | 79.17 ± 12.13 | 60.52 ± 8.77 | 65.13 ± 13.12 ‡ | 67.90M |
| DualCD-S4-Flatten | 79.01 ± 11.54 | 60.39 ± 8.27 | 64.54 ± 13.56 ‡ | 66.86M |
| DualCD-S4-TimePatch | 76.30 ± 12.48 | 60.83 ± 10.64 | — | 4.48M |
| ATCNet-S4 | 74.99 ± 10.72 | 59.85 ± 10.06 | — | 27.8k |
| ATCNet-S4 (no attn) | 63.28 ± 7.92 | 60.14 ± 9.83 | — | 25.6k |
| ATCNet (tcn8) | 83.12 ± 11.86 | 61.57 ± 11.03 | — | 41.2k |
| ATCNet (nwin1) | 83.16 ± 11.82 | 61.35 ± 11.19 | — | 38.5k |
| ATCNet (no attn) | 83.56 ± 11.79 | 61.43 ± 10.94 | — | 103.6k |

（6 个已发表条目见 §4.1；18 = 6 + 12。2C Friedman χ²=612、Nemenyi CD=3.69；前 8 名相差 1.0pp。）

### 4.4 主实验进度（19:30；作业 71018–71037，`logs/slurm/snn_*_job_ids.txt`）

| 组 | 数据集 · 臂 | 作业 | 状态 | 完成 cell / 预期 |
|:---|:---|:---|:---|---:|
| c2z_g1 | 2C · zs | 71018 → 71019 → 71020 | **RUNNING**（gpu01，17:00 起） | 0 / 20 |
| c2z_g2 | 2C · zs | 71021 → 71022 → 71023 | **RUNNING**（18:47 起） | 0 / 30 |
| c2e_g1 | 2C · eaz | 71024 → 71025 → 71026 | **RUNNING**（18:47 起） | 0 / 20 |
| c2e_g2 | 2C · eaz | 71027 → 71028 → 71029 | PENDING（QOS 4 卡上限） | 0 / 30 |
| c3z_g1 | 3C · zs | 71030 → 71031 | **RUNNING**（gpu03，18:28 起） | 4 / 55 |
| c3e_g1 | 3C · eaz | 71032 → 71033 | PENDING | 0 / 55 |
| shz_g1 | SHU · zs | 71034 → 71035 | PENDING | 25 / 70（先导的 25 个） |
| she_g1 | SHU · eaz | 71036 → 71037 | PENDING | 0 / 70 |

失败 0（所有 `failures__*.json` 的 `n_cells_failed = 0`）。2C 单组约 30 GPU-h → **2C 两臂现实上 ≥ 09-21 晚出齐**。

## 5. 分析（从第一性原理）

### 5.1 为什么表里只放已发表模型

对比表要回答的问题是「我们的模型 vs 领域里**已知**的最强方法」。读者只能校准有出处、有官方代码的模型；自研的 S4ERP / DualCD / 融合 / 消融条目没有外部锚点，放进去只会稀释表格、招来「这些是什么」的问题。它们的正确位置是 CD 图（k=19 的排名统计）和消融段落。这也是学长「改成我们自己要对比的模型」的直接含义。

### 5.2 为什么按年份排、年份怎么取

学长要年份，本质上是回答审稿人的「基线够不够新」。按 2017 → 2025 排，读者一眼看到覆盖了经典 CNN（DeepConvNet / EEGNet）、滤波器组（FBCNet）、注意力 + TCN（ATCNet）、当前最强小网络（EEGNeX）、卷积 Transformer（EEG-Deformer）。年份一律取卷期出版年，不取 arXiv 年，避免被挑「引用不规范」。

### 5.3 现在这张表能支持什么、不能支持什么

- **能说**：在 SHU 上（25 人、同折同配方），S-EEGNeX-D 与 EEGNeX 差 0.5pp 且不显著，能耗低 7×；S 版能耗低 17× 但差 1.4pp（p=0.04）；两个 SNN 都高于除 EEGNeX 外的全部 17 个 ANN 条目的 **bench 均值**（但那是不同划分，只能作参考，不能做检验）。
- **不能说**：非劣性（要 2C 的单侧 bootstrap 下界，n=51）、H1 交互（要 2C 两臂）、3C 任何 SNN 结论。**摘要措辞要等 2C。**
- SHU 是地板效应队列（chance band 51.4–53.7%），1pp 的差在 2C（被试间配对 SD 1.4–2.5pp）才有分辨力。

### 5.4 EEGNeX 那一行去不去（学长：「到时候不对比这个 eegnex」）

我建议**保留**，三条理由：
1. 模型名字就叫 S-EEGNeX，权重形状与 EEGNeX 逐层相同，审稿人第一个问题一定是「比 EEGNeX 差多少」；表里没有这一行反而显得回避。
2. 方案 v2 的立论就是**非劣性**（S-EEGNeX 在 EEGNeX 的 1pp 内 + 能耗低一个数量级 + 对齐交互），非劣性只能对着 EEGNeX 说。
3. 先导数据本身不难看：D 版 −0.48pp n.s.；如果 2C 上 S 版也进 1pp，这一行是加分项。
如果学长的意思是「最终主模型改成学长自己改版的 EEGNeX、不再叫 S-EEGNeX」，那 EEGNeX 行的去留另议——tex 里该行有注释标记，删一行即可。

### 5.5 「加 25、26 年的模型」的第一性原理

加基线的目的是堵住「基线过时」这一条审稿意见，而不是堆数量。要有效，新基线必须满足三条：(a) 2025–2026 正式发表、有官方代码；(b) 能在**同一协议**（原始 EEG 输入、跨被试、同折同配方、无测试被试标签）下跑；(c) 与论文主张相关。按 (c)，对一篇 SNN 论文最相关的 2025 基线是**另一个 2025 的 SNN**（Zhang et al., *Neural Networks* 2025，Wu 组，PLIF 直接跑采样点 + EA——正是评审说我们不能 claim 首创的那篇）；其次是一个 2025 的强 ANN / 基础模型，用来展示「跨被试的瓶颈是被试偏移不是容量」（与 67M 参数 DualCD 垫底一致）。

## 6. 与前序工作的关系

- 18 条目的数字、协议、27 条锁定决定：`docs/PRIOR_RESULTS.md`、`docs/PLAN_cross_subject_benchmark_and_fusion.md`；3C 勘误：`docs/RESULTS_TRUTH.md`。本报告没有改任何数字，只是换了呈现方式（已发表 / 自研分组、加年份、统一 SD 口径）。
- 先导结论与主实验决定：`docs/PLAN_spiking_eegnex_icassp2027.md` §0.5、`docs/DECISIONS.md` D-020~D-026；上一份给学长的报告：`docs/reports/REPORT_for_senior_2026-09-18.md`（其 §6 的 7 个待拍板项仍然有效）。
- 论文规则（每个数字指回 CSV、两块主表不混比、禁用措辞）：`AGENTS.md` §7。

## 7. 下一步 / 需要学长拍板

### 7.1 候选 2025 / 2026 基线（未跑；选定后我写 adapter + 注册成 model entry，进同一张表逐被试配对）

| # | 模型 | 年份 / 出处 | 类型 | 代码 / 权重 | 接入成本 | GPU（2C+SHU 未对齐臂，估） | 与协议的兼容性 | 我的优先级 |
|:--|:---|:---|:---|:---|:---|---:|:---|:---:|
| 1 | **Zhang et al. 2025** "A lightweight SNN for EEG-based MI classification"（Zhang, Wang, An, Zheng, **Wu Dongrui**） | *Neural Networks* 191:107741, 2025 | **SNN**（PLIF，SpikingJelly，轻量） | 公开：github.com/Zhr1110/SnnForMI | ~1 天（去掉 / 安装 spikingjelly 依赖，包成 `{logits, features, confidence}`） | ~10–15 GPU-h（轻量卷积 SNN） | 原始 EEG 输入、跨被试，完全兼容；还能同时跑我们的 EA 臂 | **1**：SNN 论文必须和 2025 的 SNN 比；也是评审点名的那篇 |
| 2 | **CBraMod** | ICLR 2025 | EEG 基础模型（criss-cross Transformer，4.0M，TUEG 预训练） | 公开；**权重已在 `../Benchmark/weights/CBraMod.pth`** | ~1 天（200 Hz 重采样、1 s patch、通道位置编码；只读 Benchmark 的第三方代码，不混文件） | ~20–40 GPU-h（全量微调 4M 参数 × 10 cell × 2 数据集） | 需重采样到 200 Hz，输入口径与其他行不同（要在脚注写明）；EEG-FM-Compass 里 12 个 FM 中排名第 1 | 2：展示「大预训练模型跨被试也不占优」 |
| 3 | **MIRepNet** | arXiv 2507.20254（2025-07），HUST Wu 组 | MI 专用基础模型（通道模板 + 掩码重建 + 监督预训练） | 公开：github.com/staraink/MIRepNet + HF 权重 | ~1 天 | ~15–30 GPU-h | 250 Hz、8–32 Hz、EA、通道模板插值；**预训练语料是 7 个 MOABB 数据集，不含 SHU / WBCIC-SHU → 无泄漏**；尚未正式发表 | 3 |
| 4 | **DSGNet** | IEEE JBHI 2026（DOI 10.1109/JBHI.2026.3689121） | 多域对齐 ANN（用了 WBCIC-SHU 3C 的那篇） | 只有架构预览（`../_archive/.../code/models/dsgnet/_model_raw.py`），**训练代码未放**，class-alignment loss 与按域分批要自己重建 | ≥ 2 天，风险最高 | ~10 GPU-h（3C）| 只有它自报的 3C 数字（68.56%，低于我们复现的 EEGNeX 69.91） | 4：除非学长想在 3C 上直接对标 |
| — | LUNA / BrainOmni / TFM / EEGMamba（2025 FM，权重都在 Benchmark） | 2025 | FM | 有 | ~1 天/个 | 20–40 GPU-h/个 | EEG-FM-Compass Table 2 上 MI 全部弱于 EEGNet（如 4 类 BNCI2014001：LUNA 28.9%、TFM 32.0% vs EEGNet 45.0%） | 不建议 |

**时间预算**：4 卡到 ~09-21 被主实验占满；新基线最早 09-21 晚开跑；每个新模型 2C+SHU 未对齐臂 ≈ 1 卡 1 天；截稿 09-24 20:00 → **最多 1–2 个**。若学长选 2 个，建议 #1 + #2，并在 09-19 就写好 adapter、用 CPU 微型 run 验过，等卡一空就投。

### 7.2 其他待拍板

1. **EEGNeX 行去留**（§5.4）。
2. **年份放法**：现在在模型名后括号里（列结构不变）；要单独一列也行（一行改 `tabular` 列定义）。
3. **3C 列**：(a) 只保留 EEGNeX / EEGNet 的 3C（主实验会用 bench 配方重跑），ATCNet / Deformer 3C 写 `--`；或 (b) 用论文配方的数字并脚注‡（现状）；或 (c) 主实验后补 ATCNet / Deformer / DeepConvNet / FBCNet 的 3C bench 配方（≈ 30 GPU-h）。
4. FBCNet（70.0% 2C）要不要留：它是 2021 的已发表模型且同协议，留着能说明「固定滤波器组 + 方差层在跨被试上明显落后」；嫌表长可删。
5. 上一份报告 §6 的 7 项（框架、transductive 写法、S/D 谁当主模型、seed、BCI IV 2a、能耗表述、学长 EEGNeX 改版是否注册进表）仍待定。

### 7.3 我这边接下来做的（不需要拍板）

- 盯主实验（每天早 / 晚），2C 出齐后跑 `analyze_snn_round.py --aligned-arm eaz`，再跑 `build_comparison_table.py --snn-2c/--snn-shu/--snn-3c` 重生成 Table 1 的行（†‡ 自动消失）。
- BCI IV 2a loader（截止 09-20 早）；分析脚本补非劣性 bootstrap / 精确 Wilcoxon / HL+BCa / Bonferroni–Dunn / Page's L。
- 学长选定 2025/26 模型后立即写 adapter（09-19），CPU 微型 run 验过后等卡。

## 8. 文件清单

| 文件 | 内容 |
|:---|:---|
| `paper/main.tex`（Table 1，`tab:main`） | 已发表基线（带年份）+ SNN 行；格式不变；注释里有重生成命令、候选 2025/26 模型、EEGNeX 行标记 |
| `paper/refs.bib` | `ding2025deformer`（卷期已核）、新增 `mane2021fbcnet`、`zhang2025snn_mi_ea`（题目作者已核）、候选 `wang2025cbramod` / `liu2025mirepnet`（VERIFY 作者表） |
| `scripts/build_comparison_table.py` | 表生成脚本（CSV → md / tex / long csv） |
| `outputs/analysis/comparison_table_2026-09-18/{comparison_long.csv, comparison_table.md, pilot_paired.md, tab_main_rows.tex}` | 生成的表与每格来源 |
| `outputs/analysis/snn_round/shu_pilot_lr1e4/{per_model,paired_tests,energy}.csv` | 先导数字出处 |
| `results_snapshot_2026-09-13/experiments/{wbci_shu/wbci_2c_cross_subject_bench_v1, shu/shu_cross_subject_bench_v1}/tables/per_model.csv`；`wbci_shu/paper_baseline_3c_821_v1/tables/per_cell.csv` | ANN 数字出处 |
| `docs/DECISIONS.md` D-032 | 论文表名单 / 年份 / SD / 3C 口径的决定 |
