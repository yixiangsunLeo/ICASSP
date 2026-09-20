---
title: "REPORT — AdaHGNN 导出并按 zs 协议提交三队列（D-037）"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/hypergraph"
created: "2026-09-20"
updated: "2026-09-20 17:40"
status: "superseded"
---

> **已作废（2026-09-20 17:40）**：本报告描述的 `network.py` 运行时改写与作业 71803–71808 已按用户要求撤回（作业全部在 PENDING 状态取消，从未训练）。当前状态与适配方案见 `PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`（D-038 / D-039）。下文仅作历史记录。

# AdaHGNN：最新版本、导出、协议对齐、yuan 号提交

## 核心结论

DeepHypergraph 里最新可导出的网络是 **V4 的 `EEG_AdaMSHyper`（AdaHGNN）**。已按与 TCFormer / LaBraM / EEGPT / CBraMod **同一套 zs 协议**接到本仓库，条目名 `adahgnn`，在 yuan 上提交 2C / SHU / 3C 三队列训练（每队列一卡）。**不跑 DANN**（本项目域对抗已结案；共享配方是 plain CE）。效果好不好要等 31 个 cell 的 `result.json`；出齐前不写结论。

## 目标

学长 09-20 15:23：看 `/share/home/yuan/LY/DeepHypergraph` 里哪版是最新模型，导出进我们的实验，三数据集、协议与这几天的 baseline 对齐，在本号跑，用来对照、看能不能发。

## 方法

### 版本选择

| 候选 | mtime | 是什么 | 决定 |
|:---|:---|:---|:---|
| `Ada-MSHyper-main` | 2025-09-02 | 时序预测论文代码，不是 EEG 分类 | 不用 |
| `project_aved` … `project_V3` | 2025-10 → 2026-01-16 | AVED/DTU/KUL 上的 AdaHGNN + DANN 脚本演进 | 祖先 |
| **`project_V4`** | **2026-01-16 16:15** | 同一网络拆成 `models.py` / `config.py` / `train.py` | **导出源** |
| V3 `AVED_multi/adahgnn_DANN_LOSO_0116_1.py` | 2026-01-16 21:46（树内最新 .py） | 同一 `EEG_AdaMSHyper` 的单文件 LOSO+DANN 实验 | 架构与 V4 同类；V4 才是可 vendoring 的模块 |

原设定：AVED 听注意、32 ch、T=128 @128 Hz、2 类、LOSO+DANN。我们的任务是 MI、32/58 ch、T=1000 @250 Hz。分类头按 2/3 类重建，**不加载 AVED checkpoint**。

### 运行时端口（`dhg` 不在环境里）

`_official/models.py` 原样保存（sha256 在 `PROVENANCE.md`）。训练走 `network.py`：

1. 源码逐通道 `TemporalStem` 循环 → `groups=C` 卷积（每个 feature map 仍有独立权重和 BN；测试里与循环版数值对齐）。
2. `dhg.nn.HGNNPConv` 默认路径 `Linear → mean v2v → ReLU → Dropout(0.5)` → 共享关联矩阵上的 `D_v^{-1} H D_e^{-1} H^T`（与把图复制 B 份再卷积代数相同）。
3. **去掉** GRL / 领域分类器 / NHC 约束损失 / mixup。训练 = 共享 Adam 1e-4 / batch 128 / patience 25 / plain CE。

32 ch 空间金字塔与源码 Config 一致 `[32, 8, 2]`。58 ch 是源码没写的唯一扩展：`[58, 29, 1]`，池化仍是连续下标均分（源码类名是 Anatomical，实现并不是 10-20 解剖分区）。

参数量（测试冻结）：32 ch / 2 类 **1,099,042**；58 ch / 2 类 **1,869,244**；58 ch / 3 类 **1,869,373**。

## 协议

与 D-034 锁定口径相同，只换模型条目：

| 项 | 内容 |
|:---|:---|
| 臂 | 只 zs（`per_sample_zscore`） |
| 配方 | Adam 1e-4 / batch 128 / wd 0 / 无 scheduler / max 500 / patience 25 / val macro-F1 / seed 0 |
| 划分 | 2C、SHU：5 折 × split 41+42；3C：LOSO。`cross_subject` 块从 `snn_*_zs_lr1e4.yaml` 原样拷 |
| 输入 | `[B, C, 1000]` @250 Hz，不做重采样 / 额外滤波 / EA |
| 产物 | 同一 `outputs/experiments/{wbci_shu,shu}/snn_*_zs_lr1e4_v1/cells/` |
| 配置 | **新文件** `bl_*_adahgnn_zs_lr1e4.yaml`，**不改** 正在被 71035 / 71223 读取的 live yaml |
| 本轮不加 | eaz、DANN、seed 43、一卡三进程 |

## 结果

接入与单测已过；全量训练数字 **running / planned**，不是 done。

- `pytest tests/models/test_adahgnn.py`：合同、32/58 ch、2/3 类、官方 sha256、yaml 与 snn 配方逐键相同、grouped stem ≡ 循环 stem、`describe()` JSON 可写。
- 作业：**71803–71808**（清单 `logs/slurm/bl_adahgnn_job_ids.txt`）。`N_LINKS=2`，`SMOKE_FIRST` 先 1 折 2 epoch。31 cell（10+10+11）。NICE=50。2C 首链改到 gpu3node。未动正在跑的 71034 / 71213 / 71222。

## 分析 / 审计要点

- 源码超图嵌入经过不可微的 `adj > 0.5`，CE 对结构嵌入无梯度——源码如此，未改。
- 有 BN，不用 micro-batch（否则 BN 统计量不等于配方 batch 128）。stem 的卷积按 batch 切片、BN 仍看满 batch。
- 不 scancel 71034 / 71213 / 71222；NICE=50，不抢正在收尾的卡。QOS 4 卡中 3 张在跑，第 4 张给本轮第一份作业。

## 与前序关系

D-034 外部四模型 + 3C ANN 补跑仍有效。AdaHGNN 是学长另给的实验室模型，探索性，不进预注册五族。分析时 `analyze_snn_round.py` 扫同一 run 目录即可配对。

## 下一步

1. 盯冒烟：`logs/slurm/bl2c_adahgnn_1-*.err` 空、`_smoke/bl2c_adahgnn` 无失败 cell，再放全量。
2. 若 58 ch × batch 128 OOM：保持配方有效 batch，先降 `conv_chunk_batch`（已是工程切片），不要改 lr。
3. 31 个 `result.json` 齐后跑 `analyze_snn_round.py`，再 `--extra` 进队列表。

## 文件清单

- `code/models/external/adahgnn/`（`_official/`、`network.py`、`adapter.py`、`PROVENANCE.md`）
- `code/configs/models/adahgnn.yaml`
- `code/configs/experiments/bl_{wbci2c,shu,wbci3c}_adahgnn_zs_lr1e4.yaml`
- `scripts/make_adahgnn_configs.py`、`scripts/slurm/baselines_plan_adahgnn.txt`
- `tests/models/test_adahgnn.py`
- `docs/DECISIONS.md` D-037
