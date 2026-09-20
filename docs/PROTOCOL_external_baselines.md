---
title: "外部新基线（TCFormer / LaBraM / EEGPT / CBraMod）+ 3C 补齐：评估协议（已锁定）与执行方案"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/benchmark"
created: "2026-09-18"
updated: "2026-09-19 00:05"
status: "locked (2026-09-19 00:00, D-034) — 改动只能通过 docs/DECISIONS.md 新条目"
---

# 外部新基线：评估协议（已锁定）+ 执行方案

> 用户 2026-09-18 23:12 指令：三个数据集上补跑 MDGEEG / EEGMoE / EEGPT / LaBraM / TCFormer 作对照；多子智能体并行、有检验；
> 先确认协议再执行；代码后续公开。23:50 收窄：**"论文只需要一个实验：一套协议 × 三个数据集 × 这些模型做对比，不加别的"**，
> 并要求确认已跑完的 ANN 与 SNN 三行协议一致。

## 0. 核心结论

1. **一个实验 = zs 臂（逐 trial 逐通道 z-score，不做被试对齐）× {2C, SHU, 3C} × bench 配方**。所有进论文 Table 1 的行必须来自这一口径。不跑 eaz 臂（若 09-21 后有空卡再议，换配置重投即可）。
2. **协议一致性核对（2026-09-18 23:55，盘上文件）**：bench（DeepConvNet / EEGNet / FBCNet / ATCNet / EEGNeX / EEG-Deformer）与 SNN 主实验在 2C、SHU 上——`splits/*.json` 逐字节相同（2C seed 41+42，SHU seed 41；42 由同一确定性函数生成）；每 cell `spec` 相同（Adam / 1e-4 / batch 128 / wd 0 / 无调度 / max 500 / patience 25 / macro-F1 / per_sample_zscore / seed 0）。配置级仅 3 处差异且不影响训练与选模：`curves`（画曲线）、`micro_batch_per_model`（SNN-D 梯度累积，有效 batch 128）、`eval.adabn`（多几列评估）。**结论：一致。**
3. **3C 列不一致，必须补**：表中 ATCNet / EEG-Deformer / EEGNet / EEGNeX 的 3C 数字来自 `paper_baseline_3c_821_v1`，**patience 100**（DSGNet 论文配方），其余全同；主实验 3C 用 patience 25，已在重跑 EEGNeX / EEGNet。→ 补跑 **ATCNet、EEG-Deformer、DeepConvNet、FBCNet 的 3C（zs、bench 配方、LOSO 11 折）**，约 15 GPU-h；补完 ‡ 与 "—" 消失。
4. 新基线名单（模型审计见 §2）：**TCFormer、LaBraM-base、EEGPT、CBraMod**。**MDGEEG 删除**（官方仓库只有结构 `model.py`，训练算法未公开、论文闭源）；**EEGMoE 换 CBraMod**（EEGMoE 无代码无权重、不在 EEG-FM-Benchmark 里）。

## 1. 协议（已锁定）

| # | 项 | 内容 |
|:--|:---|:---|
| P1 | 队列与划分 | 2C：51 人 `kfold_subject` 5 折、val 10 被试、`split_seeds [41, 42]` → 每模型 10 cell；SHU：25 人同上 → 10 cell；3C：11 人 LOSO、val 2 被试 → 11 cell。`cross_subject` 块与 bench / 主实验逐字相同 |
| P2 | 臂 | **只跑 zs**（`normalization: per_sample_zscore`）。`eval.adabn: true` 照主实验开着（只多几列，不影响任何数字的主口径） |
| P3 | 配方 | Adam 1e-4 / batch 128 / wd 0 / 无 scheduler / max 500 / patience 25 / monitor val macro-F1 / 训练 seed 0 / best-val checkpoint（同时记 last）。**所有模型同一配方**，不用各自论文的 lr / 调度 / 增强 |
| P4 | 输入格式 | 所有模型吃同一批 z-score 后的 trial `[C, 1000] @250 Hz, 0–4 s`。模型侧适配器只做格式转换（决定 8"只为兼容不为调优"）：LaBraM / CBraMod 250→200 Hz（torchaudio sinc 重采样；数据已 0.5–40 Hz 带限）→ 4×200 点片段（LaBraM 另查 `standard_1020` 通道 id；CBraMod 通道无关）；EEGPT 250→256 Hz（1024 点 = 16×64 patch）+ 词表通道 id（SHU：T3/T4/T5/T6→T7/T8/P7/P8，A1/A2 不在词表 → 丢弃，30 通道）；TCFormer 原样。**不做**各 FM 自带的幅值缩放（LaBraM 0.1 mV）、额外滤波、EEGPT 默认 EA |
| P5 | 预训练与微调 | LaBraM / EEGPT / CBraMod 加载官方权重（strict），**全参数微调** + 新线性头（EEG-FM-Compass 口径）。权重不进 git：`weights/` 为 gitignore 目录（本机是指向 `../Benchmark/weights/` 的软链接），公开代码附 `weights/README.md` + sha256。EEGPT 权重为 HF 镜像 `eeg-telecom-paris/eegpt-large-official` 转换件（结构与官方 large4E 一致、strict 加载通过、未与 figshare 原件逐张量核对）→ **论文脚注**。预训练语料：LaBraM（TUEG + 15 个公开集）、EEGPT（PhysioMI/HGD/TSU/SEED/M3CV）、CBraMod（TUEG）均不含 SHU 2022 / WBCIC-SHU → 无泄漏 |
| P6 | 结构超参 | 取论文/官方配置：TCFormer = 官方 `configs/tcformer.yaml`（F1 32、核 [20,32,64]、d_group 16、D 2、depth 5、q_heads 4 / kv_heads 2、dropout 0.4/0.3；注意类默认值不同）；LaBraM = `labram_base_patch200_200`；EEGPT = large（embed 512×4, depth 8, patch 64）；CBraMod = 官方默认（4.0 M）。写进 `code/configs/models/<name>.yaml`，`per_dataset` 只放通道集合 |
| P7 | 指标 | accuracy（2C/3C 主）、balanced accuracy（SHU 主）、macro-F1、AUC、kappa、NLL/Brier/ECE；逐被试 mean ± SD（ddof=1）；参数量按 58 ch；GPU-h |
| P8 | 统计 | 新条目进入未对齐块：逐被试配对 Wilcoxon vs EEGNeX 与 vs S-EEGNeX-S（HL + BCa CI + 胜/平/负）；Friedman + Nemenyi CD；Bonferroni–Dunn 以 S-EEGNeX 为对照（k 由 19 增为 23）。全部标探索性，不改预注册 5 项家族。测试被试不参与任何选择 |
| P9 | 产物位置 | 新 cell 写进主实验同一 run 目录 `outputs/experiments/{wbci_shu,shu}/snn_<ds>_zs_lr1e4_v1/cells/`，`analyze_snn_round.py` 自动纳入。用 `code/configs/experiments/bl_<ds>_zs_lr1e4.yaml`（由 `scripts/make_baseline_configs.py` 从 `snn_<ds>_zs_lr1e4.yaml` 生成，除 `models` / `micro_batch_per_model` / `source_config` 外逐字相同，`output_dir` 相同）与 tag `bl*`。不改任何 `snn_*.yaml` |
| P10 | 代码公开 | 官方代码原样 vendoring 到 `code/models/external/<name>/_official/`（LICENSE + `PROVENANCE.md`：URL、commit、日期、sha256、逐条偏差）；`adapter.py` 只包契约 `{logits, features, confidence}` + `describe()`；`code/configs/models/<name>.yaml`；`tests/models/test_<name>.py` |

## 2. 模型审计（2026-09-18 23:15–23:30）

| 模型 | 出处 | 官方代码 | 官方权重 | 处理 |
|:---|:---|:---|:---|:---|
| TCFormer | Altaheri, Karray, Karimi, *Sci Rep* 15:32959, 2025 | ✅ github.com/altaheri/TCFormer（纯 PyTorch，einops 已装） | 不需要 | 原样 vendoring，剥掉 Lightning 外壳 `ClassificationModule`，直接用 `TCFormerModule` |
| LaBraM-base | Jiang et al., ICLR 2024 | ✅ EEG-FM-Benchmark `models/FM/LaBraM/`（= 官方 `modeling_finetune.py`；timm 1.0.19 已装） | ✅ `labram-base.pth`（5.8 M；sha256 在 `weights/MANIFEST.yaml`） | 我们 58 / 32 个通道名全部在 `standard_1020` 词表 |
| EEGPT | Wang et al., NeurIPS 2024 | ✅ EEG-FM-Benchmark `models/FM/EEGPT/`（纯 torch） | ⚠️ 镜像转换件（见 P5） | 词表 62 通道：WBCIC 58 全在；SHU 改名 + 丢 A1/A2 |
| CBraMod | Wang et al., ICLR 2025 | ✅ EEG-FM-Benchmark `models/FM/CBraMod/`（纯 torch） | ✅ `CBraMod.pth`（4.0 M） | 通道无关（卷积位置编码），`[B, C, 4, 200]` |
| MDGEEG | Song et al., *Neurocomputing* 614:128889, 2025 | ⚠️ 只有 `model.py`，训练算法未公开 | — | **删除**（用户 23:37） |
| EEGMoE | Gao, Wang, Zhao, *TNNLS* 37(8), 2026 | ❌ 无 | ❌ 无 | **换 CBraMod**（用户 23:37） |

## 3. 算力与排产

- QOS `32cpu4gpu64g`：每用户 **4 GPU** 硬上限、48 h 墙钟；全集群 40 张 4090（24 GB）当前全占；我们 4 张在跑主实验，新作业排同一配额，**最早 09-20 夜 ~ 09-21 上卡**。
- 预算（zs，三队列；bench 实测 EEGNeX 2C 4.0 / SHU 1.6 GPU-h，ATCNet 17 / 5.2）：TCFormer ≈ 12 h、LaBraM ≈ 20–25 h、EEGPT ≈ 25–35 h、CBraMod ≈ 20 h、ANN 补 3C ≈ 15 h → **≈ 95–110 GPU-h**；09-21 ~ 09-23 约有 4 卡 × 2.5 天。
- 作业切分（按 (模型, 数据集) 一作业，2C 先投；小模型多进程共卡）：TCFormer 三数据集 3 进程一卡；FM 每 (模型, 数据集) 一卡；ANN 补 3C 4 模型 2 进程一卡。每个作业先 `SMOKE_FIRST`（1 折 2 epoch 写到 `_smoke` 目录）再全量，失败即退出、不烧卡。`N_LINKS=2` 接力续跑。

## 4. 多智能体执行（协议锁定后立即启动）

| 角色 | 只能写这些路径 | 交付 | 检验门 |
|:---|:---|:---|:---|
| 协调者（主 Agent） | `code/models/external/{__init__,_common}.py`、`code/models/registry.py`（**唯一一次**：懒加载分支 + 静态名单）、`weights/` 软链接、`scripts/make_baseline_configs.py` → `code/configs/experiments/bl_*.yaml`、`scripts/slurm/{submit_baselines.sh, baseline_gpu.sbatch}`（新增，不改现有文件）、docs | 配置、提交、监控、分析、报告 | G2 全套 pytest；G3 主实验原命令 CPU 微型 run（闭包未破）；G5 覆盖断言 `n_cells == 期望` |
| Agent-T（TCFormer） | `code/models/external/tcformer/**`、`code/configs/models/tcformer.yaml`、`tests/models/test_tcformer.py` | vendoring + adapter + yaml + 测试 | G1：58ch/2cls、58ch/3cls、32ch/2cls 前向；参数量对论文 77.8 k @22ch/4cls；`describe()`；CPU 微型 run |
| Agent-L（LaBraM） | `code/models/external/labram/**`、`labram.yaml`、`test_labram.py` | 同上 + strict 加载 + 重采样/分段/通道 id 单测 | G1 + 权重 sha256 + 缺失键 = 0 |
| Agent-P（EEGPT） | `code/models/external/eegpt/**`、`eegpt.yaml`、`test_eegpt.py` | 同上 + SHU 通道改名/丢弃单测 | G1 + strict 加载 `target_encoder.*` |
| Agent-C（CBraMod） | `code/models/external/cbramod/**`、`cbramod.yaml`、`test_cbramod.py` | 同 Agent-L | 同 Agent-L |

约束：子 Agent 不碰 `registry.py`、`scripts/slurm/` 现有文件、任何 `snn_*.yaml`、`paths.local.yaml`、别人的目录；不在登录节点跑 GPU；不 commit。

## 5. 风险

1. `registry.py` 在 20 个主实验作业的 import 闭包内：改坏 → 接力作业启动即失败（可重投续跑）。缓解：懒加载 + 静态名单 + compileall + 全套 pytest + 主实验原命令 CPU 微型 run，在 71018 最早可能结束（~09-20 00:00）前完成。
2. FM 显存 / 时长超估：作业内先冒烟；`micro_batch_per_model` 兜底。
3. EEGPT 权重为镜像转换件：脚注。
4. 截稿 09-24 20:00：新作业 09-21 上卡，zs 全套 ≈ 100 GPU-h，4 卡 ≈ 1–1.5 天 → 09-22 出齐可期。
