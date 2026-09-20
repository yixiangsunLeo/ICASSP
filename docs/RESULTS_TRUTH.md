---
title: "WBCIC-SHU 三分类跨被试 · 权威结果表"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-08-25"
updated: "2026-08-25"
status: "authoritative"
---

# WBCIC-SHU 三分类跨被试 · 权威结果表

> **这份文档是 WBCIC-SHU 三分类跨被试结果的唯一事实来源。**
> 表里每个数字都是 2026-08-25 从 `outputs/experiments/wbci_shu/*/cells/*/result.json`
> 现场重算的，口径与 `scripts/summarize_cross_subject.py` 一致（对各折 `rows[].accuracy`
> 取算术平均）。与本文档冲突的旧文档一律以本文档为准，勘误见 §8。

## 0. 先记住三件事

1. **本仓库不存在任何 2 分类的 WBCIC-SHU 实验。** 所有跑过的 run 都是 `n_classes=3`。
   任何写着"2-class"的表述都是错的（`report_824` §4.1 / §5.2 犯了这个错）。
2. **有两套训练配方，差别很大，而且对不同模型的影响方向相反。** 看任何数字之前先确认它属于哪套。
3. **`val_mode` 的影响只有约 0.5pp，可以忽略。** 曾经被当成主要疑点，已被 2×2 数据否定（§5）。

## 1. 四批实验的身份

四批实验恰好构成 `配方 × val_mode` 的 2×2 设计：

| run_id | 配方 | val_mode | 折数 | 磁盘状态 |
|:---|:---|:---|---:|:---|
| `foundation_cross_subject_3class_v1` | **我们的** | subjects | 73 cells | 在 |
| `foundation_3c_loso_paper_v1` | **我们的** | sessions | 55 cells | **已删**（数字见 §4） |
| `paper_baseline_3c_821_v1` | **论文** | subjects | 77 cells | 在，且有 `tables/` |
| `atcnet_3c_loso_paper_recipe_v1` | **论文** | sessions | 11 cells | **已删**（数字见 §4） |

两套配方的完整差异：

| 项 | 我们的配方 | 论文配方（DSGNet §IV-A） |
|:---|:---|:---|
| optimizer | AdamW | Adam |
| lr | 模型默认（S4 族 1e-3） | 1e-4 |
| batch_size | 64 | 128 |
| max_epochs | 100 | 500 |
| weight_decay | 1e-4 | 0 |
| scheduler | cosine | 无 |
| grad_clip_norm | 4.0 | 0 |
| early_stopping_patience | 25 | 100 |

共同项：LOSO 11 折、`split_seed=0`、`monitor=macro_f1`、`per_sample_zscore`、`seeds=[0]`。

数据：11 被试 × 3 session，58ch × 1000 @250Hz，9899 trials，3 类（左手/右手/双脚），
manifest `outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv`（33 行）。

## 2. 主表 A — 我们的配方 × subjects

`foundation_cross_subject_3class_v1`。标准差为**样本标准差**（÷n−1）。

| 模型 | Accuracy | Macro-F1 | 折数 |
|:---|---:|---:|---:|
| **ATCNet** | **70.85 ± 15.20%** | 70.37% | 11 |
| DualCD-S4-Flatten | 69.87 ± 11.20% | 69.82% | **7** ⚠ |
| EEGNeX | 68.78 ± 13.72% | 68.55% | 11 |
| EEG-Deformer | 67.81 ± 13.97% | 67.18% | 11 |
| S4ERP-Adv | 67.60 ± 14.22% | 67.45% | 11 |
| EEGNet | 67.22 ± 12.79% | 66.99% | 11 |
| S4ERP | 64.68 ± 13.56% | 64.32% | 11 |

⚠ DualCD-S4-Flatten 只有 7 折（fold 0-6），作业 44048 于 08-24 21:16 被取消。
**它那 7 折偏容易，不能和 11 折的数字并排读。** 限定在同样 7 折上的配对比较：

| 模型 | 同 7 折 Accuracy |
|:---|---:|
| ATCNet | **74.02%** |
| EEGNeX | 71.83% |
| S4ERP-Adv | 70.73% |
| EEG-Deformer | 70.56% |
| DualCD-S4-Flatten | 69.87% |
| EEGNet | 69.03% |
| S4ERP | 67.62% |

配对后 DualCD-S4-Flatten 排第 5，落后 ATCNet 4.15pp。

## 3. 主表 B — 论文配方 × subjects

`paper_baseline_3c_821_v1`，7 模型全部 11 折完整。

| 模型 | Accuracy | Macro-F1 |
|:---|---:|---:|
| **EEGNeX** | **69.91 ± 13.56%** | 69.68% |
| EEG-Deformer | 68.49 ± 14.83% | 68.25% |
| ATCNet | 68.39 ± 13.26% | 67.55% |
| S4ERP | 68.26 ± 14.37% | 68.18% |
| EEGNet | 66.36 ± 13.05% | 66.06% |
| DualCD-Transformer | 65.13 ± 13.12% | 65.00% |
| DualCD-S4-Flatten | 64.54 ± 13.56% | 64.36% |

**换配方会改变模型排名。** 在论文配方下最强的是 EEGNeX 而不是 ATCNet，且 ATCNet 与
S4ERP 几乎打平（差 0.13pp）。"ATCNet 明显更强"这个印象只在我们的配方下成立。

## 4. 已删除的两批（sessions 协议）

输出、checkpoint、SLURM 日志三样全删，只剩汇总数字，来源
`docs/DSGNET_SHUv5_3C_ANCHOR.md`（2026-08-10；原在 `4_experiments/wbci_shu/foundation_cross_subject/`，2026-09-18 归档时因论文 3C 队列 / LOSO 协议与"DSGNet baseline 被低估"的论断依赖它而移回 `docs/`）。
**无法做配对统计检验，要用必须重跑。**

| 模型 | Accuracy | 配方 |
|:---|---:|:---|
| ATCNet | 71.29 ± 13.22% | 我们的 |
| ATCNet | 68.91 ± 12.84% | 论文 |
| DualCD-S4-Flatten | 65.99 ± 13.65% | 我们的 |
| S4ERP | 64.77 ± 13.88% | 我们的 |
| DualCD-Transformer | 63.40 ± 13.23% | 我们的 |
| DualCD-S4-Timepatch | 61.07 ± 10.24% | 我们的 |
| **DualCD-S4-Pos** | **45.11 ± 6.66%** | 我们的 |

DualCD-S4-Pos 的 45.11% 接近三分类随机水平（33.3%），标准差异常小（±6.66 vs 其他 ±13）、
F1（41.32%）明显低于 Acc——典型的表征坍塌。**2026-08-25 已排查，结论见 §4.1。**

### 4.1 dualcd_s4_pos 为什么坍塌（已排查，三个假设全部证伪）

三个 S4 变体**只差一个池化层**，其他（骨干、六项损失、多视图）完全相同。准确率随
`feature_dim` 单调下降：

| 变体 | 池化 | feature_dim | 参数量 | Acc |
|:---|:---|---:|---:|---:|
| flatten | 全部摊平，保留 487 个 patch | 62,336 | 66.9M | 65.99% |
| timepatch | 6 个时间箱的 mean+std | 1,536 | 4.48M | 61.07% |
| pos | 注意力加权平均成**一个**向量 | 128 | 3.17M | **45.11%** |

排查过程（全部在 fold 0 上冒烟，20 epoch）：

| 假设 | 做法 | 结果 |
|:---|:---|:---|
| H1：teacher pool 不进 EMA 导致蒸馏目标坏掉 | 修 `_update_teacher_ema`（§9） | **证伪**。val 37-41%，与历史一致 |
| H2：自监督项压制监督项（默认 8.4% vs 84.8%） | `w_dino=w_ibot=0.1`，监督占比升到 35% | **证伪**。val 37-42% |
| H3：同上，改从监督侧加权 | `w_base=5.0`，监督占比升到 48% | **证伪**。val 37-40% |

关键观察：H2/H3 下监督项**确实学得更快**（`base` 11 epoch 内从 1.058 降到 0.873，
默认要 20 epoch 才到 0.946），但 **val 准确率完全不动**。也就是说分类器在拟合训练集，
学到的东西却不泛化——**不是梯度被抢走了，是池化把信息扔了**。

`AttentionPooling` 把 487 个 patch 做 softmax 加权求和，输出**单个** 128 维向量。
运动想象的判别信息是时间模式（ERD/ERS 在试次内随时间演化），一次全局加权平均把它抹平了。
flatten 保留全部时间分辨率、timepatch 保留 6 个时间箱、pos 保留 0 —— 这正是上表的排序。

**结论：`pos` 不是 bug，也不是配方问题，是池化设计不适合 MI。不要再在它上面花时间。**
DINO 这条线只有 `flatten`（65.99%，但 66.9M 参数）和 `timepatch`（61.07%，4.48M）可用。
注意学长嫌 flatten 慢的原因（62336 维投影头 = 66.9M 里的 97%）正是它效果好的原因，
**在这套 DINO 实现里快和好不可兼得**。

## 5. 因子分解：配方 vs val_mode

ATCNet 在四个格子里都有数，可以完整分解：

| ATCNet | val=subjects | val=sessions | **val_mode 效应** |
|:---|---:|---:|---:|
| 我们的配方 | 70.85% | 71.29% | +0.44pp |
| 论文配方 | 68.39% | 68.91% | +0.52pp |
| **配方效应** | **+2.46pp** | **+2.38pp** | |

结论：
- **`val_mode` 的影响约 0.5pp，两套配方下高度一致，可以忽略。**
- **配方的影响约 2.4pp，两个协议下高度一致。**

S4ERP 的方向相反：

| S4ERP | val=subjects | val=sessions |
|:---|---:|---:|
| 我们的配方 | 64.68% | 64.77% |
| 论文配方 | 68.26% | — |

配方效应 **−3.58pp**（论文配方反而更好），与 ATCNet 的 +2.46pp 符号相反。
合理解释见 §5.1：S4ERP 过拟合最严重，论文配方的低学习率（1e-4 vs 1e-3）+ 无 scheduler
起到了正则作用；ATCNet 容量小不怕过拟合，更吃高学习率 + cosine。

**推论：任何跨这两批实验的比较都无效。** 只能在同一 run_id 内部比。

### 5.1 过拟合程度（实测）

`paper_baseline_3c_821_v1` 开了 `curves: true`，逐 epoch 记录 train/val/test。
下表取各折 best epoch 那一行，11 折平均。`train` 是训练集子集在 eval 模式下的准确率。

| 模型 | train | val | test | 过拟合 gap |
|:---|---:|---:|---:|---:|
| S4ERP | 97.7% | 65.0% | 68.3% | **+32.7pp** |
| DualCD-Transformer | 91.5% | 62.0% | 65.1% | +29.4pp |
| EEG-Deformer | 94.2% | 66.5% | 68.5% | +27.7pp |
| DualCD-S4-Flatten | 89.5% | 62.3% | 64.5% | +27.3pp |
| ATCNet | 90.0% | 67.8% | 68.4% | +22.1pp |
| EEGNeX | 87.9% | 66.5% | 69.9% | +21.4pp |
| EEGNet | 80.0% | 65.3% | 66.4% | **+14.7pp** |

`report_824` §3.1 引用的 "S4ERP train-val gap +32.7%" 数值正确，但它把出处写成
"2-class 实验"——实际来自本表，即 3 分类的 `paper_baseline_3c_821_v1`。

**所有模型都严重过拟合**（最好的 EEGNet 也有 +14.7pp）。这是跨被试任务的主要瓶颈，
且与参数量的相关性弱：3.5K 参数的 EEGNet gap 最小，1.4M 的 S4ERP 最大，
但 67M 的 DualCD-Flatten（+27.3pp）反而好于 1.6M 的 EEG-Deformer（+27.7pp）。

## 6. 对抗训练：+2.92% 的确切含义

唯一成立的对抗比较来自 `foundation_cross_subject_3class_v1` 内部（同 run、同折、同配方）：

| 模型 | Accuracy | 差值 |
|:---|---:|---:|
| S4ERP-Adv | 67.60% | **+2.92pp** |
| S4ERP | 64.68% | — |

**这个数字是干净的**，但有两个尚未解决的问题：

1. **配方效应比它还大。** S4ERP 换成论文配方能拿到 68.26%（+3.58pp），比对抗训练的
   +2.92pp 更高。目前没人跑过 S4ERP-Adv 的论文配方版本，所以无法排除
   "对抗只是在补我们配方过早早停的窟窿"这个解释。
   → 正在验证：`adv_821recipe_3c_v1`（SLURM 45793，2026-08-25 提交，约 12 小时）。
2. **单 seed、11 折、折间标准差 13-15%。** 未做配对显著性检验。要下结论至少需要
   同折配对检验 + 3 个 seed。

## 7. λ 扫描（我们的配方 × subjects，11 折）

| λ_adv | run_id | Accuracy |
|---:|:---|---:|
| 0.1 | `foundation_cross_subject_adv_l01_w5` | 67.79% |
| 0.5 | `foundation_cross_subject_adv_l05_w5` | 67.37% |
| 1.0 | `foundation_cross_subject_adv_l10_w5` | 67.60% |
| 2.0 | `foundation_cross_subject_adv_l20_w5` | 67.23% |

全部落在 0.56pp 区间内，远小于折间标准差。**对抗强度不是瓶颈。**

`foundation_cross_subject_v1` 只有 3 个 cell（早期 smoke），不要引用。

## 8. 勘误：`docs/report_824_s4erp_adv_3class.md` 的三处错误（该报告已于 2026-09-18 归档到 `../_archive/eeg-mi-online_legacy_2026-09-18/docs/`）

| # | 位置 | 错误 | 更正 |
|:---|:---|:---|:---|
| 1 | §4.1、§5.2 | 把 `paper_baseline_3c_821_v1` 称为 "2-class baseline"，并据此论证"对抗训练在 3-class 下有效、2-class 下无差异" | 该 run 是 **3 分类**（`data_dims.n_classes=3`）。本仓库没有 2 分类实验。**该论证及其结论无效。** 真实差别是训练配方（§1） |
| 2 | §5.1 | DualCD-S4-Flatten 写 `67.07 ± 8.40%`，脚注却标 7/11 折 | 67.07% 是**前 6 折**的均值。fold 6 以 86.67% 完成后，7 折均值是 **69.87 ± 11.20%** |
| 3 | 全表 | 标准差用总体标准差（÷n） | 本文档统一用样本标准差（÷n−1）。换算：`报告值 = 本文档值 × √((n−1)/n)` |

另有一条过期待办：§6 写"**需要换 val_mode=sessions 复跑确认**"。该实验已于 2026-08-07~10
完成（`foundation_3c_loso_paper_v1`），结论是 val_mode 不是原因（§5）。

其余数字（ATCNet / EEGNeX / EEG-Deformer / S4ERP-Adv / EEGNet / S4ERP 六行）与重算完全一致。

## 9. 已知代码问题

| 状态 | 位置 | 问题 |
|:---|:---|:---|
| **已修 2026-08-25** | `code/models/eeg_foundation/models.py` `_update_teacher_ema` | teacher **pool** 不在 EMA 更新列表里。只影响 `dualcd_s4_pos`（唯一池化带参数的变体，66,497 个参数），其 teacher 注意力池化一直停在随机初始化，违反 DINO 的 teacher=EMA(student) 前提。`flatten`/`timepatch` 池化无参数，**行为不变，已有结果仍可比**。同样的缺陷在学长原始包 `models_eeg_foundation/models.py:95-102` 里也存在 |
| 未修（暂无影响） | `code/models/eeg_foundation/adapter.py:179-181` | `EEGFoundationConfig.adv_warmup_epochs` 声明了但构造模型时没传下去，配置里写的值被静默忽略。当前所有配置都是 5，与模型默认值相同，故无行为差异 |
| 未修（未来陷阱） | `code/models/eeg_foundation/adapter.py:246-251` | `training_step` 在 `is_dualcd` 为真时提前 return。**将来若注册 `dualcd_*_adv` 变体，对抗分支会被静默跳过**——和已修的 `c0cc95e` 是同一类问题 |

已确认修复过的历史 bug（勿重复排查）：`ae95cd1` manifest、`c0cc95e` uses_custom_loss、
`1e0eb9c` n_classes=3、`bf49e24` lambda warmup，以及 SLURM conda 环境和 CUDA device-side assert。

## 10. 文件夹现状

被文档引用但**不存在**的路径（outputs 与 checkpoints 均已删）：
`foundation_3c_loso_paper_v1`、`atcnet_3c_loso_paper_recipe_v1`、`foundation_3c_loso_v1`。

7 个 wbci_shu run 里只有 `paper_baseline_3c_821_v1` 生成过 `tables/`。

磁盘：`checkpoints/` 70G（`shu/cross_subject_5fold_v1` 单个 39G），`outputs/` 14G。

## 11. 进行中的作业（2026-08-25 提交）

| JobID | 名称 | 内容 | 预计 |
|---:|:---|:---|:---|
| 45792 | `pos_smoke` | 修完 EMA bug 后 `dualcd_s4_pos` 跑 fold 0 / 20 epoch，验证是否脱离坍塌 | ~30 分钟 |
| 45793 | `adv821` | `s4erp_adv` 论文配方 11 折，验证 +2.92% 是否被训练预算吃掉 | ~12 小时 |

`pos_smoke` 的判读基准（`foundation_cross_subject_3class_v1` 的 fold 0）：
ATCNet 66.00 / S4ERP-Adv 65.56 / EEGNet 63.11 / EEGNeX 61.00 / DualCD-Flatten 60.00 /
EEG-Deformer 59.33 / S4ERP 57.44。落进这个区间 = bug 是坍塌根因；仍在 45% 附近 = 另有原因。
