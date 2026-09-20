---
title: "REPORT — AdaHGNN-MI：实现、验证、三队列提交（D-040）"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/hypergraph"
created: "2026-09-20"
updated: "2026-09-20 18:30"
status: "active"
---

# AdaHGNN-MI：从方案到提交

## 核心结论

学长的 `EEG_AdaMSHyper` 已按 `PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`（用户 17:38 拍板：接受 T2、原版不跑、mixup 1 臂、DANN 不开、命名 AdaHGNN-MI、F=32）实现为 **AdaHGNN-MI**（≈0.1 M 参数），两个条目 `adahgnn@mi`（主）与 `adahgnn@mi_mix`（同类别 mixup 臂），21 项单测全过，已在 yuan 提交三队列 **62 cell**（作业 71825–71830）。全集群 40/40 卡满载，目前排队。**没有任何测试集数字；出齐前不写结论。**

## 目标

在与 18 ANN + 4 外部基线 + SNN 完全相同的 zs 协议下，让学长模型在三个跨被试 MI 队列上达到该架构能达到的最高准确率，并可逐被试配对比较。

## 方法（实现了什么）

`code/models/external/adahgnn/mi_model.py`，官方文件 `_official/models.py` 原样保留（sha256 与实验室文件一致，注册为 `adahgnn` 但不跑）。

| 层 | 原版（AVED 设定） | AdaHGNN-MI | 审计条目 |
|:--|:---|:---|:--|
| 时间滤波 | 每通道独立 stem，1→256 逐点 + depthwise 63/31 @128 Hz，`d_model`=128 | **跨通道共享** `Conv(1,125)+Conv(1,63)` → BN，F=32 | A1 A4 A5 |
| 时移 | `LearnableTemporalShift(fs=128)` | 去掉 | A4 |
| 超图结构 | 嵌入 N(0,1)、阈值 0/1、`nonzero`——不可微 | 嵌入 N(0, d^-1/2)，softmax(3·relu) top-5 **软权重**，可微；`A = D_v^-1 H D_e^-1 H^T` | A2 |
| 空间滤波 | 无（先功率再混合） | **信号域 Laplacian** \(\tilde S=(I-\gamma A_1)S\)，γ 每电极 sigmoid，初始 0.5 | A6 |
| 功率读出 | GELU → AvgPool(8) → Dropout → 均值 | GELU → AvgPool(16) → Dropout(0.5) → **softmax 时间权重**（初始均匀）→ 每节点仿射 | A4 |
| 层级 | 连续下标均分（C3+C4 一组） | **解剖分半球** 58→13 区→{SM-L, SM-R, Mid, Rest}；32→14→4 | A3 |
| 超图卷积 | `Linear → HGNNPConv(Linear → mean v2v → ReLU → Dropout)` + NHC | 同结构（稠密等价，单测对 `dhg` 1e-5），**无 NHC** | A9 |
| 读出 | 三尺度均值 concat | `[mean(尺度1); flatten(尺度2); flatten(尺度3)]` → 128 → BN → ReLU → Dropout → 类别 | A7 |
| 训练 | AdamW/1e-3, 按测试被试选点, DANN, 跨类别 mixup | 共享 zs 配方；`mi_mix` 臂 = **同类别** Dirichlet mixup（p=0.5、α=2、k=3，混合后重 z-score） | A8 |

其他：`adapter.py` 按 yaml `arch` 分派；`adahgnn.yaml` 加 `per_dataset.channel_set` 与 `variants.{mi, mi_mix}`；`scripts/make_adahgnn_configs.py` roster = 两条目并重生成 `bl_*_adahgnn_zs_lr1e4.yaml`；`baselines_plan_adahgnn.txt` 每队列一卡两进程。BatchNorm 头对单样本 ragged batch 有保护（协议 loader `drop_last=False`）。

## 协议

与 D-034 相同：zs（逐 trial 逐通道 z-score）、Adam 1e-4 / batch 128 / wd 0 / 无 scheduler / max 500 / patience 25 / 监控验证 macro-F1 / seed 0；2C、SHU 5 折 × split 41+42，3C LOSO；cell 写进同一 `snn_*_zs_lr1e4_v1`（目录 `adahgnn-mi__*`、`adahgnn-mi_mix__*`）。**论文行选择规则（预先声明）**：两臂中验证集 macro-F1 均值更高者进对照表，另一臂探索性；不看测试集。

## 验证（提交前）

| 检查 | 结果 |
|:--|:--|
| `tests/models/test_adahgnn.py` | **21 passed**：官方 sha256 = 实验室文件；三队列 forward/backward 形状，**所有参数都有梯度**；结构嵌入有梯度、行和 1、≤k、初始不饱和；稠密聚合 vs `dhg.Hypergraph.v2v(mean)` < 1e-5；解剖映射为划分，C3/C4、FC1/FC2、CP1/CP2、F3/F4、P3/P4 不同区且 C3∈SM_L、C4∈SM_R、Cz∈MID；单样本训练 batch 不崩；mixup 同类别、非自身、重 z-score、p=0 恒等；3 步 Adam 无 NaN；`describe()` JSON；配置协议逐键一致；plan 一致 |
| `test_external_integration`（配置/roster/懒加载子集）、`test_fusion_and_specs` | 9 + 53 passed |
| 数据顺序 | 三队列 npz `channel_names` 与 `configs/datasets/{wbci_shu,shu}.yaml` 顺序**一致**（解剖映射按名字，顺序错会静默错位，故核对） |
| 提交前 roster | `submit_baselines.sh` 6 个 (config, model) 解析通过；`DRY_RUN` 命令核对 |

## 结果

无。作业 **71825–71830** 排队中（NICE=50、SMOKE_FIRST、N_LINKS=2；3C 首链 71829 改 gpu3node）。每队列一卡两进程（激活各 ≈4–5 GB）。估算：上卡后每 cell 15–30 min，每队列 5–10 h。

## 分析 / 风险

- 集群 40 卡全部被占，何时上卡取决于其他用户；我们自己的 71213 / 71222（gpu01）、71034（gpu03）今晚结束后会释放 3 卡。
- 冒烟先跑 1 折 2 epoch；任何 import / 形状 / OOM 问题会在几分钟内让作业退出，不烧全量。修复后重投同一命令即续跑。
- 未做先导阶梯（用户「全都跑」），所以各组件的贡献没有单独证据；两臂的验证集指标可以事后比较（不涉及测试集）。
- 与原版差别实质性：论文中写 "AdaHGNN-MI (adapted from EEG-AdaMSHyper)"，改动清单即上表。

## 与前序关系

D-034 协议、D-037 导出、D-038 撤回改写、D-039 方案；本报告 = D-040。`REPORT_adahgnn_submit_2026-09-20.md` 已作废。

## 下一步

1. 上卡后看冒烟：`logs/slurm/bl2c_adahgnn_1-71825.err` 为空、`_smoke/bl2c_adahgnn_mi/failures__*_smoke.json` `n_cells_failed=0`。
2. 出齐后 `analyze_snn_round.py`（自动纳入两条目）→ 按验证 macro-F1 选臂 → `build_dataset_tables.py` 加行 → 报告。

## 文件清单

- 模型：`code/models/external/adahgnn/{mi_model.py, adapter.py, PROVENANCE.md}`；`code/configs/models/adahgnn.yaml`
- 配置：`code/configs/experiments/bl_{wbci2c,shu,wbci3c}_adahgnn_zs_lr1e4.yaml`（`scripts/make_adahgnn_configs.py` 生成）；`scripts/slurm/baselines_plan_adahgnn.txt`
- 测试：`tests/models/test_adahgnn.py`
- 作业：`logs/slurm/bl_adahgnn_job_ids.txt`（71825–71830）
- 方案：`docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`；决定 `docs/DECISIONS.md` D-040
