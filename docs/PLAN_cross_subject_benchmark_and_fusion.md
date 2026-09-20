---
title: "跨被试 benchmark 与 ATCNet×S4ERP 融合 · 执行计划"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-08-27"
updated: "2026-08-27"
status: "p0_done_awaiting_p1"
---

# 跨被试 benchmark 与 ATCNet×S4ERP 融合 · 执行计划

> **2026-09-18 阅读说明**：本计划的 18 条目实验已于 2026-08~09 全部跑完（结果见 `docs/PRIOR_RESULTS.md`），
> 这里保留它是因为 **§0 的 27 条锁定决定（B-1..27）与 §4 协议就是 ICASSP SNN 轮的配对基线口径**。
> 文中要求更新 `0_docs/FILE_CATALOG.md / operation_log.md / 0_docs/STATUS.md` 的指令已被 `docs/DECISIONS.md` D-029 取代
> （四份活文档：`AGENTS.md`、`docs/STATUS.md`、`progress.md`、`docs/DECISIONS.md`）；`4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md`
> 现为 `docs/CROSS_SUBJECT_PROTOCOL_MEMO.md`；其余 `0_docs/ 4_experiments/` 引用指向 `../_archive/eeg-mi-online_legacy_2026-09-18/`。
>
> **状态（2026-08-27 原文）：P0 已完成（代码就绪，134 个测试全过），等待 P1 smoke 的开工指令。**
> **未跑任何 GPU 实验。** P0 的落地清单与实测参数量见 §8-P0 的"已完成"标注。
>
> **P0 阶段的实测修正（2026-08-27）**：
> * 参数量表原为 3 类值，已换成本轮的 **2 类**实测值（§5.1/§5.2/§5.3）。
> * `atcnet_s4` 实测 **27,834** 参数（比 ATCNet 的 114,554 小 **4.12 倍**；含 `s4_d_ff=32` 通道混合），
>   `readout_idx = [12,13,14,15,16]`，SSM 时间常数 **[0.27, 16.89] tokens**（Tc=17）。
> * `cell_signature` 的守卫做了比原计划更精确的设计：`early_stopping_patience` **总是**
>   进签名，`max_epochs` **仅在有 scheduler 时**进（见 §6.2 的修订说明）。
> * P0 期间发现并修掉两个既存 bug（§8-P0 附注）。
> **v2 修订（2026-08-27）**：原计划把融合拆成"P2 跑完再做 P3"，该分阶段在实际预算下
> 站不住（见 §0 决定 13 的修订说明）。现在**全部模型——10 个 benchmark + 2 个融合方向
> + 3 个 ATCNet 消融——在同一个 run 里一次提交**，成本只从 356 增到约 450 GPU-h
> （3.7 天 → 4.7 天），换来任意两个模型之间都可做配对检验。
>
> **v3 修订（2026-08-27，P0 代码审查）**：预注册比较里已有
> `atcnet@noattn` vs `atcnet_s4@noattn`（§5.3/§5.4 的单变量对照），但 17 条目名册漏了
> 后者，Holm 家族会静默缩成 7。补为第 18 条，成本约 +17 GPU-h。同时：`atcnet_s4` 的
> L2 不再罚 SSM skip `D`（init=1，核 ~1e-4，罚它会把时间阶段压成恒等）；Friedman /
> Wilcoxon 的"更好"方向随指标翻转；并发 group 不再互相覆盖 `resolved_model_params.json`。
>
> 数字来自 `docs/RESULTS_TRUTH.md`、已完成 run 的 `result.json`、manifest 实测、参考文献原文。
> 开工后同步更新 `0_docs/FILE_CATALOG.md` / `0_docs/operation_log.md` /
> `0_docs/STATUS.md` / `progress.md` / `AGENTS.md`（AGENTS §8.5）。

---

## 0. 已锁定的决定

| # | 决定 | 来源 |
|:--|:---|:---|
| 1 | **seed = 被试划分 seed**（`cross_subject.split_seeds`）；`train.seeds` 固定 `[0]` | 用户 08-27 |
| 2 | `split_seeds: [41, 42]`（先跑两个划分，P4 可扩到 41–45） | 用户 08-27 |
| 3 | 自研 `eegnet` 弃用，统一 `eegnet_official`；`cap_eegnet` 不进表 | 用户 08-27 |
| 4 | 域对抗模型全部删除：`s4erp_adv`、`atcnet_adv` | 用户 08-26 |
| 5 | `dualcd_s4_pos` 不跑（已证伪：45.11%，表征坍塌） | 用户 08-27 |
| 6 | `dualcd_s4_timepatch` **进库** | 用户 08-27 |
| 7 | **WBCIC-SHU 3C 本轮不用** | 用户 08-27 |
| 8 | 训练配方统一；模型结构参数允许按数据集调整，但**只为兼容性，不为调优** | 用户 08-27 |
| 9 | **主报告指标 = Accuracy / F1 / AUROC**，口径参考 ERP-Benchmark | 用户 08-27 |
| 10 | **`early_stopping_patience: 25`** | 用户授权，依据 §6.2 |
| 11 | WBCIC-2C 被试数 = **51** | 用户 08-27 + 论文核实 |
| 12 | S4 实现问题：真 DPLR vs 现状对角实现做实证对比，**延后到 P4**；同时追查"原实现为何这样做" | 用户 08-27 |
| ~~13~~ | ~~ATCNet 消融在 P2 之后做，不并行插队~~ **已修订 → 决定 25** | 用户 08-27，08-27 深夜推翻 |
| 14 | **代码将公开**，需完整、完善、可复用 | 用户 08-27 |
| 15 | 已完成 run 全部保留不覆盖；新方法论一律用新 `run_id` | AGENTS §4 |
| 16 | **被试划分比 = 6:2:2**（对齐 ERP-Benchmark） | 用户 08-27 |
| 17 | **`timepatch` 分箱边界改为 MI 时程** `[0,500,1000,1500,2000,3000,4000]` ms | 用户 08-27 |
| 18 | **训练配方 = 已跑过的那套**：Adam 1e-4 / 无 scheduler / batch 128 / max 500ep | 用户 08-27 |
| 19 | **模型选择监控 = `macro_f1`**（对齐 ERP-Benchmark 的 "best F1 Score"） | 我提 |
| 20 | P0 的工程准备**全做** | 用户 08-27 |
| 21 | 所有实验按 4090 集群规范配置，log 与 error 完整落盘 | 用户 08-27 |
| 22 | **融合方向 A（主线，用户指定）：「空间滤波使用 ATCNet 的」** → `s4erp_dw` | **用户 08-26 第 1 点** |
| 23 | ATCNet 消融要跑并对比（`tcn_filters` / `n_windows` / `attention`） | 用户 08-26 第 4 点 |
| **25** | **全部模型一次提交**：10 benchmark + 融合方向 A（3 变体）+ 融合方向 B（2）+ ATCNet 消融（3）= **18 个条目**，同 run / 同 fold / 同 split / 同配方 | **用户 08-27，推翻决定 13；审查修订 +`atcnet_s4@noattn`** |
| **26** | **融合方向 A 跑 3 个 F1/D 变体**（见 §5.2），不预先猜一个 | 我提，成本仅 +18 GPU-h |
| **27** | **变体的"最优"按验证集 `macro_f1` 选，不按 test 选**；三个变体全部列表 | 我提，合法性要求 |

### 决定 13 为什么被推翻

原分阶段有三条理由，重新审视后只有一条半成立：

| 原理由 | 重新评估 |
|:---|:---|
| ATCNet 消融需要 P0-3 的模型别名机制 | **成立，但只是工程依赖**，P0 做完即解除，不需要等 P2 |
| 消融需要 P2 的 ATCNet 基线做对照 | **错**。同 run / 同 fold / 同 split / 同配方下对照天然存在**且是配对的**，比分两个 run 更强 |
| 融合方向 B 依赖消融结论 | **只对一半**。消融的价值是"省掉方向 B 的算力"；一起提交则两个结果同时出来，直接看方向 B 的结果即可，不需要消融去预测它 |

根本原因：分阶段是在"算力紧张"的假设下做的，但实际总预算只有 356 GPU-h，
加上全部融合与消融也只到约 450 GPU-h（+0.9 天）。**分阶段的收益不存在，代价是丧失配对性。**

---

## 1. 第一性原理：先确定 claim，再设计实验

论文的 claim 决定实验设计，不能反过来。两条互斥候选：

| | Claim M（模型文章） | Claim B（benchmark 文章） |
|:---|:---|:---|
| 主张 | ATCNet×S4ERP 融合模型在跨被试上优于现有方法 | 跨被试 MI 解码的瓶颈是被试而非模型；难被试在零漂移下也不可解码 |
| 需要的证据 | 融合模型**显著**优于最强 baseline | 多数据集 × 多模型 × per-subject 分解 + 被试内对照 |
| 当前证据 | **无**。模型间总极差 6.2pp(3C) / 5.4pp(SHU)，被试标准差 10–15pp | **已很强**：3 个独立队列、被试/模型方差比 45×、sub-011 零漂移仍 chance、3 条域对抗负证据 |

**关键洞察：两条 claim 需要的实验重叠约 90%**，而决定 25 之后它们**共用同一个 run**。
所以不存在"先做 benchmark 还是先做模型"的取舍 —— 一次跑完，按结果分支。

### 1.1 统计功效：为什么必须配对，为什么必须一起提交

要检出模型间 3pp 的差异需要多少被试？

- **非配对**（直接比两个模型均值）：以 SHU 实测被试标准差 ≈10pp 计，n=25 时标准误 2.0pp，
  可检出 ≈ **5.6pp > 模型总极差 5.4pp → 检不出任何差异**。
- **配对**（同批被试、同 fold、同 split，比模型间差值）：被试难度在差值里抵消，
  配对标准差经验上 3–6pp。以 5pp 计：n=51（2C）标准误 0.70pp → 可检出 **2.0pp**；
  n=25（SHU）标准误 1.00pp → 可检出 **2.8pp**。

**配对检验是这项工作能否成立的前提**，而配对要求所有模型跑在完全相同的
(split_seed, fold, subject) 三元组上 —— **这正是决定 25 的科学理由**：
融合模型必须和 baseline 在同一个 run 里，否则"融合更好 X pp"这句话没有统计支撑。

现有代码 fold 由 `split_seed` 确定性生成、所有模型共用，`report_scope: per_test_subject`
已保留逐被试行，**数据条件已满足，不需要额外改代码**。

> 勘误（对早期对话的更正）：我曾把"保存 per-trial 预测"列为配对检验的前置条件，属过度陈述。
> per-subject 配对检验（MOABB / ERP-Benchmark 的标准做法）用现有数据即可。
> per-trial 只是 McNemar（trial 级）、混淆矩阵、事后集成分析的前置。

### 1.2 已经结案的方向：域对抗（三条独立负证据）

| 实验 | 结果 | 含义 |
|:---|:---|:---|
| `adv_821recipe_3c_v1`（11/11） | S4ERP-Adv 论文配方 **68.54%** vs S4ERP 68.26% = **+0.28pp**；我们配方下是 +2.92pp | 对抗项起的是**通用正则**作用，换成隐式正则更强的配方后失效 |
| `atcnet_adv_3class_v1`（22/22） | ATCNet 70.85% → ATCNet-Adv **66.63%** = **−4.22pp** | 强骨干上是净损失 |
| λ 扫描（4 值 × 11 折） | 0.1→2.0 全落在 **0.56pp** 内，远小于折间标准差 13–15pp | 若机制真是不变性学习，对抗强度不可能不敏感 |

理论侧：invariance 类方法只在 covariate shift 下正确；EEG 跨被试含大量 posterior drift
成分（同一 X 在不同被试上可能对应不同 Y），此时清除被试信息 = 清除决策所需信息。
详见 `docs/ANALYSIS_atcnet_and_adversarial.md` §2.2。

### 1.3 已拿到的关键诊断：难被试是"没信号"不是"迁移失败"

`wbci_3c_within_ceiling_v1`（165 cells × 2 模型，已完成）：

| 被试 | 被试内(EEGNet) | 被试内(ATCNet) | 跨被试(ATCNet) | 差 |
|:---|---:|---:|---:|---:|
| sub-011 | 37.6 | 34.7 | 37.0 | −0.6 |
| sub-008 | 56.3 | 40.1 | 73.8 | **+17.5** |
| sub-009 | 59.2 | 58.1 | 73.8 | **+14.6** |
| sub-005 | 79.9 | 71.9 | 68.3 | −11.6 |
| sub-002 | 93.7 | 93.1 | 90.4 | −3.3 |
| 均值(11人) | **70.5** | **64.6** | **70.9** | |

**(1)** sub-011 零域漂移下也只有 34.7%/37.6%（chance=33.3%）→ BCI illiteracy，域自适应救不了。
**(2) 但这个"天花板"不是天花板** —— 4 个被试的跨被试成绩反超被试内（sub-008 高 17.5pp），
因为被试内每 cell 只有 192 训练 trial 而跨被试有 7,199。**它测的是小样本可解码性。**
真天花板要合并同被试全部 session 重跑（P3-B-1）。
附带：被试内 EEGNet(70.5) 反超 ATCNet(64.6)，与"ATCNet 容量大、收敛慢、192 trial 喂不饱"一致。

---

## 2. 数据事实（2026-08-27 实测）

### 2.1 队列关系已查证：2C 与 3C 是互斥队列，不是子集

> 62 healthy participants... Among them, **51 subjects participated in the two-class MI
> experiment, and 11 in the three-class** MI experiment.
> — Yang et al., *Scientific Data* 12:488 (2025), doi:10.1038/s41597-025-04826-y

MOABB `Yang2025`：**2C = subjects 1–51，3C = subjects 52–62**。62 = 51 + 11，无重叠。
加上 SHU 2022 的 25 人，我们有 **3 个独立被试队列**。

- ⚠ Figshare 描述写 "52 subjects"，与论文正文和 MOABB 的 51 矛盾。**已定以 51 为准**（决定 11）。
- ⚠ 我们的 3C manifest 编号 `sub-001..011`，原编号是 **52–62**（已重编号）。若 3C 回归需给映射表。
- 可引用锚点：论文报 within-session SOTA 为 2C **85.3%** / 3C **76.9%**；论文结论自己提出
  *"future development... could also focus on **BCI illiteracy**"* —— 与 §1.3 直接呼应。

### 2.2 两个数据集的实测结构

| | **WBCIC-SHU 2C** | **SHU 2022** |
|:---|---:|---:|
| 被试 | **51**（sub-001..051） | 25 |
| ok session / 总 | **148 / 153**（5 个 failed） | 125 / 125 |
| 每被试 session | 47 人有 3，3 人有 2，1 人有 1（`sub-024`） | 全部 5 |
| 通道 × 点 × 采样率 | 58 × 1000 @ 250 Hz | 32 × 1000 @ 250 Hz |
| 类别 | 2（左/右手抓握） | 2（左/右手） |
| 每 session trial | 恰好 200（全部 148 个） | 87–100（剔坏 trial） |
| 总 trial | **29,600** | **11,988** |
| 每被试 trial（min/均/max） | 200 / 580 / 600 | 437 / 480 / 494 |
| 类别平衡 | **完美 14800:14800，0 个不平衡 session** | 5983:6005，**67/125 session 不平衡** |
| 跨被试实验 | **从未做过** | 已完成 225 cells（旧方法论，保留） |

```bash
python3 -c "
import pandas as pd
for p in ['/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv',
          '/share/workspace2/moto_imagination/SHU/processed/npz_clean/processed_manifest.csv']:
    d=pd.read_csv(p); ok=d[d.status=='ok']
    print(p.split('/')[4], len(ok),'ses', ok.subject_id.nunique(),'subj', ok.n_trials.sum(),'trials')
"
```

**两数据集时间维度完全相同（1000 点 @ 250 Hz），只差通道数。**
SHU 的类别不平衡意味着 `balanced_accuracy` 必须一起报。

### 2.3 决定 8 的技术结论：不需要任何按数据集的结构改动

1. `n_channels` / `n_times` / `n_classes` / `sfreq` 由**实验层注入**，不是模型参数。
2. 依赖时间长度的结构参数只看 `n_times=1000`，两数据集相同：
   ATCNet `Tc=(1000//8)//7=17`、S4 族 `patch_num=(1000-28)//2+1=487`。
3. **经验证据**：`paper_baseline_3c_821.yaml`（58ch）与 `shu_cross_subject_5fold.yaml`（32ch）
   的 `model_params` 块**逐字相同**且两个 run 都跑成功。

**结论：一套完全相同的 `model_params` 覆盖两个数据集**（唯一例外是决定 17 的 timepatch
分箱，按 trial 时长定，两数据集同为 4 s，所以仍统一）。`per_dataset` 机制仍要建（P0-3，
公开代码需要它作为可复用能力），本轮条目留空。

---

## 3. 参考基准：ERP-Benchmark（决定 9/16 的出处，解释了 seeds 41–45）

**Benchmarking ERP Analysis: Manual Features, Deep Learning, and Foundation Models**,
arXiv:2601.00573，§Experimental Setups：

> Batch sizes are 128 for all methods. The training epochs are set to 200 with early
> stopping (**patience=15**) based on the **best F1 Score**. We use **AdamW** with learning
> rates of 1×10⁻⁴, scheduled by **CosineAnnealingLR**. Model performance is evaluated using
> **Accuracy, F1 Score, and AUROC**... with a **6:2:2 ratio of subjects**... Each experiment
> is repeated with **five random seeds (41-45)**.

1. **`split_seeds: [41..45]` 出处就是这篇**，且原文明确 seeds 用于
   *"random subject partitioning across different seeds"* —— **就是被试划分 seed**，与决定 1 一致。
2. **指标 = Accuracy / F1 / AUROC**（决定 9），我们**已全部具备**。
3. `encoders.py` 的 `ShallowNetEmbedding` docstring 写着 "from ERP_Benchmark" ——
   **学长交付的整个 S4/DINO 包源头就是这篇**。这解释了 `d_model=128`、`patch_num` 算法、
   `TemporalBinnedPooling` 默认分箱为何都是 ERP 取向。**这是 §5.4 命名问题与决定 17
   分箱问题的共同根源，也是决定 12 中"追查原实现为何这样做"的具体对象。**

### 3.1 最终对照

| 项 | ERP-Benchmark | 我们 | 状态 |
|:---|:---|:---|:---|
| seeds（被试划分） | 41–45 | **41, 42** | 同语义，先跑 2 个 |
| 被试划分比 | 6:2:2 | **6:2:2** | 对齐 |
| 指标 | Acc / F1 / AUROC | **同** + 5 补充列 | 对齐 |
| 早停监控 | best **F1** | **`macro_f1`** | 对齐 |
| patience | 15 | **25** | 宽松 67% |
| optimizer / scheduler / max_ep | AdamW / Cosine / 200 | **Adam / none / 500** | **偏离**，见下 |
| CV 形式 | Monte Carlo CV | **k-fold**（每被试恰好测一次） | 我们更严格 |

**偏离理由**：`RESULTS_TRUTH` §5 实测两套配方对 ATCNet **+2.46pp**、对 S4ERP **−3.58pp**
（符号相反，会改变排名）；我们已跑过的两个大 run 都用 Adam + 无 scheduler，沿用可与历史
数字做 sanity check 且成本最低。**这条偏离必须写进论文 limitation，并引用 ±2.46/−3.58pp
作为"排名依赖配方"的证据。** 补跑 AdamW+Cosine 作为对照列在 P4。

---

## 4. 协议

两个数据集都是 `kfold_subject`，都保证**每个被试在每个 split 下恰好被测一次**，
per-subject 表格与 mean ± std over subjects 的算法完全同构。

### 4.1 协议 A — WBCIC-SHU 2C（新建）

```yaml
cross_subject:
  protocol: "kfold_subject"
  n_folds: 5                 # np.array_split(51,5) -> 11,10,10,10,10 个测试被试
  val_mode: "subjects"       # 验证 = 留出的训练被试（严格零样本）
  n_val_subjects: 10         # fold0: 30:10:11；fold1-4: 31:10:10  => ≈ 6:2:2
  split_seeds: [41, 42]
train:
  seeds: [0]
```

规模（每被试 580 trial 均值）：训练 17,400–17,980 / 验证 5,800 / 测试 5,800–6,380。

不用 LOSO：51 折会贵 10 倍；MOABB 的 `CrossSubjectEvaluation(n_splits=5)` 就是这个方案。

### 4.2 协议 B — SHU 2022（新 run_id）

```yaml
cross_subject:
  protocol: "kfold_subject"
  n_folds: 5
  val_mode: "subjects"
  n_val_subjects: 5          # 15 : 5 : 5 = 精确 6:2:2
  split_seeds: [41, 42]
```

规模：训练 7,200 / 验证 2,400 / 测试 2,400。

⚠ 旧 run `shu_cross_subject_5fold_v1`（`n_val_subjects=4`、`monitor=accuracy`）
**不能复用** —— 两项都在 `cell_signature` 里，会正确报错。SHU 用新 run_id 全量跑，
旧 run 完整保留作历史对照。

### 4.3 已退出的协议

- **WBCIC-3C LOSO**：决定 7。历史结果（4 个 run）保留在 `RESULTS_TRUTH`。
- **被试内天花板 `wbci_3c_within_ceiling_v1`**：已完成，结论见 §1.3；
  若走 benchmark 路线需在 2C 上补合并全 session 的版本（P3-B-1）。

---

## 5. 完整模型清单：18 个条目

一次提交的全部内容。**`@` 后缀是 P0-3 的模型别名机制**，同一模型的不同配置。

### 5.1 组 1 — Benchmark 基线（10 个）

| # | 条目 | 类别 | 出处 | 参数 @58ch | 参数 @32ch/2类 |
|:--|:---|:---|:---|---:|---:|
**参数量全部为 2 类任务实测值（2026-08-27，P0 完成后逐个构建测得）。**

| # | 条目 | 类别 | 出处 | @58ch | @32ch |
|:--|:---|:---|:---|---:|---:|
| 1 | `eegnet_official` | 已发表 | Lawhern 2018, arl-eegmodels | 3,026 | 2,610 |
| 2 | `eegnex` | 已发表 | Chen 2024 | 58,770 | 57,106 |
| 3 | `eeg_deformer` | 已发表 | Ding 2024（原生 PyTorch，未改动） | 1,610,066 | 1,503,570 |
| 4 | `atcnet` | 已发表 | Altaheri 2023 | **114,554** | 113,722 |
| 5 | `deepconvnet` | 经典 | Schirrmeister 2017 | 302,552¹ | 286,302¹ |
| 6 | `fbcnet` | 经典 | Mane 2021 | 20,711¹ | 13,223¹ |
| 7 | `s4erp` | 自研 | 对角 SSM + flatten，监督 | **1,370,882**² | 944,898 |
| 8 | `dualcd_s4_flatten` | 自研 | + DINO/DualCD | 66,861,186 | 66,009,218 |
| 9 | `dualcd_transformer` | 自研 | Transformer + DINO | 67,900,034 | 67,048,066 |
| 10 | `dualcd_s4_timepatch` | 自研 | + 时间分箱池化（MI 分箱，决定 17） | 4,480,386 | 3,628,418 |

¹ 这两行以前是 **0** —— 它们缺 `describe()`，而 `run_cell` 只从 `describe()` 取
  `n_params`（旧 SHU 表里可以看到两个 0）。P0-2 已补，P1 smoke 会断言全部非 0。
² 逐层：ShallowNetEmbedding 970,496（其中空间卷积 **950,400**，占全模型 **71%**）
  + S4Encoder 275,712 + 分类头 124,674。

**baseline 硬规矩**：只认原作者官方仓库，无完整官方代码就排除并写明理由。
已排除 EEG-Inception / MDGEEG / EEG-DG / DSGNet（见 `code/models/paper_baselines/README.md`）。
**已删除**：`eegnet`（自研）、`cap_eegnet`、`s4erp_adv`、`atcnet_adv`、`dualcd_s4_pos`。

### 5.2 组 2 — 融合方向 A：S4ERP 基座 + ATCNet 空间前端（3 个变体）

**这是用户 08-26 指定的融合主线（决定 22）。** 换掉 S4ERP 的阶段 1（空间滤波）。

`ShallowNetEmbedding` 现在的空间卷积是 dense `Conv2d(128,128,(58,1))` = **950,400** 参数，
占整个 S4ERP 的 **66%**，而 ATCNet 做同一件事（学一组空间滤波器 ≈ CSP）只用
depthwise 的 **1,856** 参数 —— **差 512 倍**。ATCNet 的 depthwise 还多一层先验：
每个时域滤波器（≈频带）配自己的空间滤波器，即 FBCSP 结构；dense 把这个先验扔了。

改成 depthwise 后 `Conv2d(F1, F1*D, (C,1), groups=F1)` 的参数是 `(F1*D)*C`，
**只取决于输出通道数**，与 F1/D 怎么拆无关。但 **F1/D 的拆分决定频带数，是实质性的
架构差异**，且有三个都合法的答案，所以三个都跑（决定 26）：

**实测（58ch / 2 类，2026-08-27）**：

| # | 条目 | F1（频带数） | D（每带空间滤波器） | 前端参数 | 全模型参数 | vs `s4erp` | 说明 |
|:--|:---|---:|---:|---:|---:|---:|:---|
| — | `s4erp`（对照） | 128 dense | — | 970,496 | 1,370,882 | — | 组 1 第 7 项 |
| 11 | `s4erp_dw@f64d2` | 64 | 2 | **25,984** | **426,370** | −68.9% | 对 S4ERP 改动最小，保持 F1·D = d_model = 128 |
| 12 | `s4erp_dw@f16d8` | 16 | 8 | **24,736** | **425,122** | −69.0% | 忠于 ATCNet 的 16 频带，但每带 8 个滤波器 |
| 13 | `s4erp_dw@f16d2p` | 16 | 2 | **6,592** | **406,978** | −70.3% | 完全照抄 ATCNet 的 16/2，加 1×1 投影 32→128 |

**一个顺带测出来的、比参数总量更能说明问题的数字：通道缩放。**
从 32ch 换到 58ch（+26 个电极）时，

| 前端 | 32ch → 58ch 的参数增量 | 增量公式 |
|:---|---:|:---|
| dense（`s4erp`） | **+425,984** | `d_model² · ΔC` = 128 × 128 × 26 |
| depthwise（`s4erp_dw`） | **+3,328** | `(F1·D) · ΔC` = 128 × 26 |

**相差 128 倍**，因为 dense 版本在通道维上还多乘了一个 `d_model`。也就是说
dense 空间卷积的代价不只是"现在大"，而是**随电极数线性放大且系数是 d_model**——
任何更高密度的脑电设置都会让它更糟。这条是方向 A 除了参数量之外的独立论据。

**这三个是同一个融合方案的三种配置，不是三个不同的融合。**
`s4erp` 与它们构成单变量对照（只改空间卷积类型），是"depthwise 空间前端是否更好"最强的证据形式。

**修完之后剩下什么**：S4Encoder 275,712（65%）+ flatten 分类头 124,674（29%）。

> ⚠ **一处早期论证的更正（2026-08-27 审查）**：我曾把这个宽分类头称作"p≫n 病灶"，
> 依据是 ATCNet（495 参数的头，70.85%）vs S4ERP（187K 的头，64.68%）。那是**跨架构的
> 两点比较，混淆严重**。同族内部的证据方向相反（`RESULTS_TRUTH` §4.1）：
>
> | DINO 变体 | 池化 | feature_dim | Acc |
> |:---|:---|---:|---:|
> | flatten | 全保留 | 62,336 | **65.99%** |
> | timepatch | 6 个时间箱 | 1,536 | 61.07% |
> | pos | 注意力池化成 1 个向量 | 128 | 45.11% |
>
> **同族内"越宽越好"。** 真正的规律不是"头窄好"，而是**"表征必须保序"**：
> 注意力池化是置换不变的加权平均，把时间结构抹平了；flatten 全保留；
> ATCNet 的窄头之所以能行，是因为它前面的因果 TCN 已经做了**保序的**压缩。
>
> 所以：**"宽头是病灶"这个说法不成立，不要作为设计依据。** 相应地，
> "给 S4ERP 加一个 SSM 末位读出把头压窄"这类改法**不应该做**——同族证据预测它会掉点。
> 方向 A 与方向 B 仍然互补（一个改空间前端、一个改时间聚合），但互补性不建立在头宽上。

### 5.3 组 3 — 融合方向 B：ATCNet 基座 + S4 时间聚合（2 个）

换掉 ATCNet 的阶段 3（时间聚合）+ 顺带解决它的一个结构矛盾。

**实测（58ch / 2 类，2026-08-27）**：

| # | 条目 | 参数 | 对比 |
|:--|:---|---:|:---|
| — | `atcnet`（对照） | 114,554 | — |
| 14 | `atcnet_s4` | **27,834** | **小 4.12 倍**（−75.7%；含 FFN mixer） |
| 15 | `atcnet_s4@noattn` | **25,642** | **小 4.47 倍**（−77.6%）；`atcnet@noattn` 的单变量对照 |

实测确认（`describe()` 输出，P1 smoke 会断言）：
`Tc = 17`、`readout_idx = [12, 13, 14, 15, 16]`（正是官方模型 5 个窗口的末步位置）、
SSM 时间常数 **[0.27, 16.89] tokens** —— 落在 `Tc = 17` 之内，说明
`log_dt` 重初始化生效了（默认初始化会给出 6.8–546，7/8 个模式退化成滑动平均）。

设计依据（实测/推算）：ATCNet 的 5 个窗口末步恰好是原序列第 12–16 步；
每个 token 覆盖 239 采样点 ≈ 956 ms、步长 224 ms；一个 13-token 窗口张成
**3,644 ms = 4 s trial 的 91%**；TCN 感受野 19 > 窗长 13。
→ **5 个窗口是同一视角的 5 个平移副本（集成），不是多尺度分解。**
而 5 套独立 TCN 权重花了 **83,840 参数（占 ATCNet 的 73.1%）**去做本该平移等变的事。

一个**共享的因果 S4** 天然同时满足权重共享（1/5 参数）与多位置读出（保留集成），
5 个独立 `Linear(32,2)`（共 330 参数）把集成留在读出层。
估算：ConvBlock 19,424 + 共享 S4 约 1,600 + 5 个头 330 ≈ **21,354**。

⚠ **必须重新初始化 `log_dt`**。现有初始化 `rand*log(0.1)-4` 给出时间常数 **6.8–555 步**，
是为 S4ERP 的 487 个 patch 调的。搬到 13 步窗口上，8 个状态里 7 个会退化成滑动平均
—— 不改几乎必然拿到负结果，且会被误读成"S4 不行"。

### 5.4 组 4 — ATCNet 解释性消融（3 个，决定 23）

**不是新模型，是 ATCNet 的配置变体。** 参数已暴露在 `ATCNetConfig`，纯 config。
在决定 25 之后它们从"决策门"降级为"解释性消融"—— 因为方向 B 的结果同时出来了，
不需要它们来预测。但它们回答的问题仍然有价值：

| # | 条目 | 改动 | 实测参数 @58ch | 回答什么 |
|:--|:---|:---|---:|:---|
| — | `atcnet`（对照） | — | 114,554 | — |
| 16 | `atcnet@tcn8` | `tcn_filters: 32→8` | **41,234**（−64%） | TCN 是否过参数化？若掉点 <1pp，则方向 B 的"共享 SSM 替 5 个 TCN"有实证依据 |
| 17 | `atcnet@nwin1` | `n_windows: 5→1` | **38,450**（−66%） | 滑窗集成贡献多少？（假设 H2：若集成是主力，应掉 3–5pp） |
| 18 | `atcnet@noattn` | `attention: mha→null` | **103,594**（−9.6%） | MHA 是否是摆设？（假设 H3：预测掉点 <1pp） |

`@noattn` 只省 9.6% 的参数却是三个消融里唯一不动 TCN 的，所以它同时是
`atcnet_s4@noattn` 的单变量对照（见 §5.3）。

### 5.5 `s4erp` 的 "S4" 名不符实（决定 12）

`s4_layers.py` 构造了完整 HiPPO-LegS 矩阵，但 forward 里 `A_diag = torch.diagonal(A)`
**只取对角线**，下三角结构被丢弃。HiPPO-LegS 的对角恰好是 −(n+1)，所以实际核是

$$K[t] = \sum_{n=1}^{8} C_n B_n\, e^{-n \Delta t \cdot t}$$

即 **8 个可学习衰减指数的加权和 = 多时间尺度 EMA bank**，是对角 SSM（S4D 类），
不是 S4（Gu et al. 2022 的 DPLR）。附带后果：`A_log`/`A_sign` 各 64 个元素中 56 个
永远拿不到梯度。

**本轮处理**：表里标 **"S4ERP (diagonal SSM)"**，方法节如实描述为多尺度指数衰减核，
**不引用 Gu et al. 2022 作为实现依据**。真 DPLR 对比延后到 P4。

### 5.6 `timepatch` 分箱边界（决定 17）

默认 `[0,150,250,350,450,600,750]` ms 是 **ERP/P300 时间窗**（注释写 "for 200Hz, T=170"）。
我们的 trial 是 **4000 ms @ 250 Hz**，用它会把 750–4000 ms 的全部 patch 塞进最后一个箱
（代码 `if not assigned: append(n_bins-1)` 兜底）。

改为 **MI 时程** `[0, 500, 1000, 1500, 2000, 3000, 4000]` ms
（cue 后约 0.5 s 起 ERD，2–3 s 达峰，后段 ERS 回弹），仍 6 箱，
`out_dim = 6 × d_model × 2 = 1,536` 不变。两数据集 trial 时长相同，故统一取值。

---

## 6. 训练配方（已锁定）

### 6.1 指标（决定 9）

| 主表前三列（对齐 ERP-Benchmark） | 补充列 | 状态 |
|:---|:---|:---|
| `accuracy`、`macro_f1`、`auc`(=AUROC) | `balanced_accuracy`（SHU 不平衡必需）、`nll`、`brier`、`ece` | 已实现 |
| | `cohen_kappa` | **P0-4 新增** |

`auc` 口径：二分类用 P(class=1)，多分类用 macro-OVR；两数据集都是二分类，即标准 AUROC。
每 cell 同时报 `best.pt` 与 `last.pt` 两套数字。

### 6.2 patience = 25（决定 10 的依据）

**① `cell_signature` 里没有 `early_stopping_patience`（新发现的隐患）。**
payload 含 optimizer / lr / batch / scheduler / grad_clip / monitor / model_params /
三个被试列表……但 **`early_stopping_patience` 和 `max_epochs` 都不在里面**。
改 patience **不会**触发签名不匹配，续跑会**静默复用**旧 patience 的 cell → 同表混
patience → 比较无效且无报错。→ **P0-9 已完成。**

> **P0 实施时的修订（比原计划更精确）**：两个参数的作用不同，所以守卫是不对称的
> （`cross_subject_protocols._stopping_identity`）：
> * **`early_stopping_patience` 总是进签名** —— 它决定哪个 epoch 成为 `best.pt`，
>   同一条优化轨迹在不同 patience 下会给出不同结果。这才是要堵的洞。
> * **`max_epochs` 只在 scheduler 非 none 时进签名** —— 在我们的 `scheduler: none` 配方下
>   它是纯上限，"提高上限后续跑"是同一条轨迹的严格延长（也正是墙钟被砍后重投的做法）；
>   而在 cosine 下它**就是**退火视界，改它会改变后续每一步的学习率，那才是换了实验。
>   把它无条件写进签名会禁掉一个合法且必要的操作，且没有方法学收益。
> 签名版本号仍是 **v2**（相对 v1 payload 确有变化），旧 run 复用会给出明确报错。

**② 参考文献用 patience=15**，我们取 25 已宽松 67%。

**③ 实测：patience=25 在 SHU 上没有截断任何模型。**

| 模型 | best_epoch 中位 | 最大 | epochs_run 中位 | 早停 |
|:---|---:|---:|---:|:---|
| `fbcnet` | 88 | **247** | 113 | 25/25 |
| `eegnex` | 55 | 101 | 80 | 25/25 |
| `atcnet` | **51** | 101 | 76 | 25/25 |
| `eegnet_official` | 30 | 94 | 55 | 25/25 |
| `eeg_deformer` | 23 | 53 | 48 | 25/25 |
| `s4erp` | 12 | 65 | 37 | 25/25 |
| `deepconvnet` | 9 | 132 | 34 | 25/25 |
| `dualcd_s4_flatten` | 9 | 61 | 34 | 25/25 |
| `dualcd_transformer` | 9 | 45 | 34 | 25/25 |

`fbcnet` 有 cell 改进到第 247 epoch，说明该深的地方会深。AGENTS.md 记的风险
（"821 实测 atcnet best_epoch 中位 406，patience 25 会砍掉最优点"）**没有实现**
—— SHU 上中位是 51。821 的 406 应归因于 3C 数据量小 + macro_f1 监控更噪，
两者都不适用于本轮（2C 数据量是 3C 的 2.4 倍）。

**④ patience 50 贵约 1.68 倍，且贵在最没必要的地方。** `epochs_run ≈ best_epoch + patience`，
patience 主导那些最优点很早的模型 —— 恰好是 3 个 DINO 变体（中位 best_epoch = 9），
占总算力 84%，`epochs_run` 会从 34 涨到 59（+74%）。

### 6.3 最终配方（决定 18/19）

```yaml
train:
  # ---- 已跑过的那套（决定 18）：与 paper_baseline_3c_821_v1 / shu_cross_subject_5fold_v1 同源 ----
  optimizer: "adam"
  lr: 0.0001
  batch_size: 128
  max_epochs: 500                 # 上限，实测从不触及（见下）
  weight_decay: 0.0
  scheduler: "none"
  grad_clip_norm: 0.0
  # ---- 早停与模型选择 ----
  monitor: "macro_f1"             # 决定 19，对齐 ERP-Benchmark 的 "best F1 Score"
  monitor_mode: "max"
  early_stopping_patience: 25     # 决定 10
  # ---- 全条目共享，保证表可比 ----
  normalization: "per_sample_zscore"   # per-trial fit-free，无统计量跨划分
  seeds: [0]                      # 训练种子固定；扫的是 split_seeds
  curves: true                    # 每 epoch train/val/test；test 仅监控，有 guard
  train_eval_max_trials: 2000
  num_workers: 2
  amp: false
  device: "cuda"
  micro_batch_per_model:          # 梯度累积保持等效 batch 128，勿改成直接降 batch
    dualcd_s4_flatten: 32
    dualcd_transformer: 16
    dualcd_s4_timepatch: 32
```

**"不要训练太久"如何保证**：`max_epochs=500` 只是上限，真正的约束是 `patience=25`。
旧 SHU run 的 225 个 cell **全部 25/25 早停**，`epochs_run` 中位 34–113，
观测最大约 272（`fbcnet` 的 247+25）。500 从未被触及，改小不省时间，
反而引入与已跑方案的偏离（且 `max_epochs` 目前不在 `cell_signature` 里，改它是静默的）。

`monitor` 从旧 SHU run 的 `accuracy` 改为 `macro_f1`：两数据集都是接近平衡的二分类
（2C 完美平衡，SHU 5983:6005），两指标数值上几乎重合，**几乎不影响选出的 epoch**，
但口径与 ERP-Benchmark 和 821 run 一致。

---

## 7. 算力预算（决定 25 之后；**2026-08-27 用 smoke 47237/47355 实测重算**）

单 cell 成本 = (单 epoch 秒数) × (收敛 epoch 数)。前者：2C 由 smoke 实测 10 个条目，
其余按下方缩放；后者：全部取旧 SHU run `cross_subject_5fold_v1` 的实测 `epochs_run`
（同配方、同 patience 25），新条目取同族参照物。

**原估算用了统一的 4.5× 缩放（2.46 试次 × 1.81 通道），实测证明它是错的**——
缩放比取决于前端是否吃通道数：

| 模型 | SHU s/epoch | 2C s/epoch | 实测比 |
|:---|---:|---:|---:|
| `eegnet_official` | 2.2 | 11.1 | **×4.96** |
| `s4erp` | 13.9 | 69.9 | **×5.03** |
| `atcnet` | 22.7 | 56.5 | **×2.49** |
| `dualcd_transformer` | 216.5 | 556.8 | **×2.57** |

前端把通道压掉的（ATCNet 的 depthwise、DINO 的 patch embedding）≈ **2.5×**，即纯试次比；
前端 dense 吃通道的（S4ERP 的 `Conv2d(128,128,(58,1))`、EEGNet）≈ **5.0×**。
未实测条目按所属族的比例缩放。

| 条目 | epochs | 2C s/epoch | 2C h/cell | SHU h/cell | 来源 |
|:---|---:|---:|---:|---:|:---|
| `dualcd_transformer` | 36.9 | 556.8 | **5.704** | 2.217 | 实测 |
| `dualcd_s4_timepatch` | 38.8 | 406.6 | **4.382** | 1.705 | 实测 |
| `dualcd_s4_flatten` | 38.8 | 406.2 | 4.378 | 1.704 | 缩放 ×2.57 |
| `atcnet` | 78.5 | 56.5 | **1.232** | 0.494 | 实测 |
| `atcnet@nwin1` | 78.5 | 56.5 | 1.232 | 0.479 | 缩放（保守取满） |
| `atcnet@noattn` | 78.5 | 56.5 | 1.232 | 0.479 | 缩放（保守取满） |
| `s4erp` | 41.3 | 69.9 | **0.802** | 0.159 | 实测 |
| `eegnex` | 81.6 | 31.6 | 0.716 | 0.142 | 缩放 ×5.03 |
| `atcnet@tcn8` | 78.5 | 25.9 | **0.566** | 0.220 | 实测 |
| `eeg_deformer` | 43.2 | 39.3 | 0.472 | 0.094 | 缩放 ×5.03 |
| `fbcnet` | 130.7 | 11.3 | 0.410 | 0.082 | 缩放 ×5.03 |
| `atcnet_s4` | 78.5 | **11.6** | **0.252** | 0.098 | 实测 |
| `atcnet_s4@noattn` | 78.5 | **11.4** | **0.249** | 0.097 | 实测 |
| `s4erp_dw@f64d2` | 41.3 | 20.9 | **0.240** | 0.093 | 实测 |
| `s4erp_dw@f16d8` | 41.3 | 20.9 | 0.240 | 0.093 | 缩放 |
| `eegnet_official` | 66.3 | 11.1 | **0.204** | 0.041 | 实测 |
| `deepconvnet` | 46.9 | 14.4 | 0.187 | 0.037 | 缩放 ×5.03 |
| `s4erp_dw@f16d2p` | 41.3 | 13.3 | **0.153** | 0.059 | 实测 |

| run | cells | GPU-h（实测重算） | 原估算 |
|:---|---:|---:|---:|
| `wbci_2c_cross_subject_bench_v1`：18 × 5 折 × 2 split | **180** | **~227** | ~384 |
| `shu_cross_subject_bench_v1`：18 × 5 折 × 2 split | **180** | **~83** | ~85 |
| **合计** | **360** | **~310** | ~469 |

4 卡并行（QOS `gres/gpu=4` 硬上限）→ 墙钟约 **77 小时 ≈ 3.2 天**（原估 4.9 天）。

**偏差 −34%，在验收阈值（±50%）内且方向有利，因此不重新排产，P2 按原方案提交。**
偏差几乎全部来自 2C：原估算把 ATCNet 族和 DINO 族也按 4.5× 缩放，
而它们实际只有 2.5×。SHU 侧 83 vs 85 h 几乎不变（那本来就是实测）。

3 个 DINO 变体仍占约 **算力的 75%**（14.5 of 19.4 h/cell），不裁剪 —— 它们是 Claim B
的重要论据（67M 参数换来垫底表现）。

**融合方向 B 的额外发现**：`atcnet_s4` 是 **11.6 s/epoch**，而 `atcnet` 是 56.5 —— 
**快 4.9×**，同时参数少 4.12×（27,834 vs 114,554）。这是论文里可以直接用的
效率论据，原估算（~2.8 h/cell，即比 ATCNet 略贵）方向就是反的。

### 7.1 4090 集群配置（决定 21，已核实）

| 项 | 值 |
|:---|:---|
| 节点 | login01 + gpu01–05，每节点 8× RTX 4090D 24 GB，**无 SSH，仅 Slurm** |
| 分区 | `gpu2node`(gpu01,02) / `gpu3node`(gpu03,04,05)，**association 钉死，不能写多分区** |
| QOS `32cpu4gpu64g` | `MaxWall 48h`、`cpu=32`、**`gres/gpu=4` 硬生效**、`mem=64G` 不生效 |
| 单 job | `-c 8`、`--gres=gpu:1`、**`--mem=40G`（不要动，实测 MaxRSS 15.6 G）** |
| 并发 | 最多 4 个 job |
| 环境 | `mi_torch_cu118`（torch 2.7.1+cu118） |
| 卡满 | 第 4 个 job PENDING `(Resources)` → `scontrol update jobid=<id> partition=gpu3node` |
| 日志 | `#SBATCH -o logs/slurm/%x-%j.out` / `-e logs/slurm/%x-%j.err` |
| 配额 | home 512 GiB；IO 重的活放 storge |
| 规矩 | 不在 login 节点跑重活；先 smoke 再全量；无 CUDA 时 fail fast |

⚠ **`MaxWall 48h` < `dualcd_transformer` 在 2C 上的单模型总时长**（~10 h × 10 cells = 100 h），
必须跨 job 接力。`last.pt` 续跑 + `result.json` 完成标记已支持，但需自动重投脚本（P0-5）。

---

## 8. 执行步骤

### P0 — 工程准备 + 新模型代码 ✅ 已完成（2026-08-27）

全部 13 项完成，`pytest tests/ -q` **134 passed**，无 lint 错误。

**P0 期间发现并修掉的两个既存 bug**（都在 `git stash` 前就存在于 HEAD，非本次引入）：

1. **`S4ERP` 上有一个多余的 `update_adv_lambda` 方法。** 它引用 `self.grl` 和
   `self.adv_warmup_epochs`，而这两个属性只有 `S4ERPAdversarial` 有；adapter 的
   `training_step` 又用 `hasattr(model, "update_adv_lambda")` 鸭子类型判断，
   于是**plain `s4erp` 调用 `training_step` 会直接 AttributeError**。
   它一直潜伏是因为 `s4erp` 的 `uses_custom_loss` 为 False，训练器走的是普通 CE 路径，
   从不调用 `training_step`。已删除该方法，并把 adapter 的判断从 `hasattr` 改成
   显式的 `_ADVERSARIAL_VARIANTS` 集合 —— 和 `is_dualcd` 陷阱同一类问题，一并堵掉。
2. **`s4erp_adv` 违反 `{logits, features, confidence}` 契约**（多返回 `subject_logits`），
   契约测试因此一直是红的。既然它已弃用且不进 benchmark，测试改为只对"仍维护的变体"
   断言契约，并新增一个测试把这条偏离**显式记录**下来，而不是悄悄跳过。

| # | 动作 | 落地位置 | 状态 |
|:--|:---|:---|:---|
| **P0-9** | `cell_signature` 加入早停设定（见 §6.2 的不对称修订）+ 签名版本号 v2 | `cross_subject_protocols.py` `_stopping_identity` / `SIGNATURE_VERSION`；`cross_session_e2e_protocols.py` 同步 | ✅ |
| **P0-3** | 参数单一来源（`defaults` / `per_dataset` / `variants`）+ 模型别名 `name@variant` | 新建 `code/models/model_specs.py`；`registry.build_model` 解析别名；`runners.py` 调 `resolve_all` 并写 `resolved_model_params.json`；cell 目录名用 `safe_tag` | ✅ |
| **P0-1** | 5 个模型改为**弃用但保留**（见下方说明）+ 堵掉 `is_dualcd` 静默跳过 | `registry.DEPRECATED_MODELS` / `eeg_foundation.adapter.DEPRECATED_VARIANTS`；`training_step` 的 DualCD 分支现在遇到对抗参数会**报错**而非静默丢弃 | ✅ |
| **P0-11** | 组 2：`DepthwiseShallowNetEmbedding`（新类，不改现有类）+ `S4ERPDW` + 3 个变体 | `encoders.py` / `models.py` / `adapter.py`（`dw_F1`/`dw_D`）/ `configs/models/s4erp_dw.yaml` | ✅ |
| **P0-12** | 组 3：`atcnet_s4` 共享因果 SSM + **`log_dt` 按 Tc=17 重初始化** + 因果掩码 MHA | 新建 `code/models/atcnet/atcnet_s4.py`（`atcnet_torch.py` 未改动）+ `adapter.ATCNetS4Classifier` + `configs/models/atcnet_s4.yaml` | ✅ |
| **P0-2** | `deepconvnet`/`fbcnet` 补 `describe()` + `default_lr()` | `deepconvnet.py` / `fbcnet.py`；实测 302,552 / 20,711（原为 0） | ✅ |
| **P0-4** | 加 Cohen's Kappa，并进结果列 | `experiments/metrics.py`；`CROSS_SUBJECT_RESULT_COLUMNS` 加 `cohen_kappa` / `last_cohen_kappa` | ✅ |
| **P0-10** | `timepatch` 分箱 → MI 时程 `[0,500,1000,1500,2000,3000,4000]` ms | **由 P0-3 自动生效**：yaml 里本来就有 MI 分箱，是实验 config 的内联 `model_params` 把它盖掉了 | ✅ |
| **P0-6** | 保存 per-trial 预测 | `run_cell` 写 `predictions.npz`（subject/session/y_true/y_pred/probs × best+last，约 100 kB/cell） | ✅ |
| **P0-5** | 提交脚本：4 卡分组 + 48h 自动重投链 + sacct 快照 + `failures.json` + 环境指纹 | 新建 `scripts/slurm/{bench_gpu.sbatch,submit_bench.sh}`；`runners._environment_fingerprint()`（git commit / torch / cuDNN / GPU / SLURM_JOB_ID） | ✅ |
| **P0-13** | 两个实验 config，`model_params` 刻意留空 | `configs/experiments/{wbci_2c,shu}_cross_subject_bench.yaml`，两者 `--dry-run` 通过、17 条目齐全 | ✅ |
| **P0-7** | summarize 重写：多 split_seed / 按 val 选变体 / 配对检验 / Friedman+Nemenyi / CD 图 / 跨数据集主表 | 新建 `code/summaries/cross_subject_bench.py` + 重写 `scripts/summarize_cross_subject.py`；已在已完成的 SHU run 上验证跑通 | ✅ |
| **P0-8** | 公开准备 | 新建 `LICENSE`（含第三方与数据条款）/ `requirements.txt` / `environment.yml` / `pyproject.toml` / `REPRODUCE.md` / `tests/models/test_fusion_and_specs.py`（41 个测试） | ✅ |

#### P0-1 的一处刻意偏离：改成"弃用但保留"，而不是删除

计划原文写的是"registry **删** `eegnet`/`cap_eegnet`/`s4erp_adv`/`atcnet_adv`/`dualcd_s4_pos`"。
实施时改成了**保留实现、标记弃用、从 benchmark 名单剔除**，理由有三条：

1. **这五个都产出过 `docs/RESULTS_TRUTH.md` 里引用的数字。** 删掉代码等于让那些
   已发表/已记录的数字不可复现 —— 对一个即将公开的仓库是明确的坏实践。
2. **`eegnet` / `cap_eegnet` 被 Phase 0–2c 的历史 config 引用。** AGENTS 明确要求
   "Phase 0–2c 结果可复现"，删掉会直接破坏它。
3. 用户的意图（不再参与实验、不再混淆）由 `BENCHMARK_MODEL_NAMES` 白名单 +
   构建时的弃用警告完全满足，且更安全。

落地：`registry.DEPRECATED_MODELS` 与 `eeg_foundation.adapter.DEPRECATED_VARIANTS` 各自
携带**弃用理由和替代品**，`build_model` 构建时 WARNING。
`configs/models/*.yaml` 同理保留。

**⚠ 尚未提交**：当前工作区在 P0 之前就有 18 个 modified + 一批 untracked
（含 `registry.py`、`eeg_foundation/{adapter,models}.py`、`atcnet/adapter.py`），
最后一个 commit 是 `bf49e24`。P0 的改动叠加在这之上。
**建议在 P1 之前提交，建立干净基线** —— 需要用户明确授权。

### P1 — Smoke（1 卡，约 3 小时）

**必须覆盖全部新代码**（一起提交的代价是 bug 要在 smoke 阶段抓出来）。
逐条验收清单与可直接粘贴的检查脚本见 [`REPRODUCE.md`](../REPRODUCE.md) §4：

```bash
sbatch -J bench_smoke scripts/slurm/bench_gpu.sbatch \
  code/configs/experiments/wbci_2c_cross_subject_bench.yaml \
  --folds-subset 0 \
  --models eegnet_official,atcnet,s4erp,dualcd_transformer,dualcd_s4_timepatch,s4erp_dw@f64d2,s4erp_dw@f16d2p,atcnet_s4,atcnet_s4@noattn,atcnet@tcn8 \
  --max-epochs 5 \
  --out outputs/experiments/wbci_shu/wbci_2c_bench_smoke \
  --ckpt-dir checkpoints/wbci_shu/wbci_2c_bench_smoke
```

**验收清单**（全过才进 P2）—— **2026-08-27 全部通过**：

| # | 项 | 结果 |
|:--|:---|:---|
| 1 | 全部 cell `ok`，无异常 | ✅ **20/20**（10 条目 × 2 split × fold 0），`failures.json` 为空 |
| 2 | `n_params` 全部非 0，且与 §5.2/§5.3 一致 | ✅ `s4erp_dw@f64d2` = 426,370；`atcnet_s4` = **27,834**；`@noattn` = 25,642 |
| 3 | per-trial 文件写出，Kappa 在指标里 | ✅ 20/20 有 `predictions.npz`，`probs_best` 形状对齐；`cohen_kappa` 在每行 |
| 4 | 6:2:2 与 `split_seeds` 正确、测试被试零交集 | ✅ 每 cell val=10 被试，train/val 与 test 交集为空 |
| 5 | `timepatch` 的 6 个箱都非空 | ✅ 每箱 patch 数 = 57 / 62 / 63 / 62 / 125 / 118 |
| 6 | `atcnet_s4` 时间常数落在 Tc 量级 | ✅ **τ ∈ [0.27, 16.89] tokens**，Tc=17；readout `[12..16]` |
| 7 | `dualcd_transformer` 显存够用 | ✅ `micro_batch=16 × accum=8 → 128`，MaxRSS 9.25 G / 申请 40 G，无 OOM |
| 8 | 实测耗时替换 §7 估算，差 >50% 需重排产 | ✅ 见 §7 重算：**310 vs 469 GPU-h（−34%）**，阈值内且方向有利，不重排产 |

**smoke 过程中抓到并修掉的两个 bug**（正是 smoke 存在的理由）：

| bug | 症状 | 修复 |
|:---|:---|:---|
| cell 目录未创建 | `np.savez_compressed` 不建父目录，训练成功后**每个 cell** 都在写 `predictions.npz` 时 `FileNotFoundError` | `run_cell` 开头 `mkdir(parents=True)`（`98f4b33`） |
| `timescales()` 张量不带 device | `torch.arange` 建在 CPU，而 `log_dt` 在 cuda；`describe()` 每次存 best checkpoint 都调用 → **fusion B 4/4 cell 训练后崩溃** | 传 `device=dt.device`；补 GPU 门控的全名册 `forward + loss + describe()` 回归测试（`2f606b0`） |

第二个 bug 在 CPU 上**原理上不可见**（两个操作数都在 CPU），这是 CPU plumbing 跑通、GPU 仍然全挂的原因。

### P2 — 全量统一 run（4 卡，约 3.2 天）

| run_id | 内容 |
|:---|:---|
| `wbci_2c_cross_subject_bench_v1` | **18 条目** × 5 折 × split 41,42 = **180 cells** |
| `shu_cross_subject_bench_v1` | **18 条目** × 5 折 × split 41,42 = **180 cells** |

产出：per-subject 表、per-model 表、跨数据集主表（Acc/F1/AUROC + 补充列 + 参数量 + GPU-h）、
配对检验矩阵、CD diagram、三曲线图。

**决定 27 的执行**：组 2 的三个变体，按跨折平均**验证集** `macro_f1` 选出最优者作为
主结果的代表，**三个全部列在表里**。绝不按 test 挑。

### P3 — 分析与论文分支（原 P4）

**P3-M（融合有效）**：`s4erp_dw` 或 `atcnet_s4` 在配对检验下显著优于对应基座
→ 走模型文章。补：扩到 `split_seeds: [41..45]`、考虑补 3C LOSO 作第三队列。

**P3-B（融合无效）**：转 benchmark 文章。补：
1. **P3-B-1：2C 被试内天花板**，必须**合并同被试全部 session**（约 580 trial）做 5 折
   —— 3C 版本已被证明数据饥饿（§1.3）。⚠ 现有 `within` runner 是 within-**session**，
   合并 session 的 within-**subject** 协议**可能要新写**，届时确认。
2. §1.2 三条域对抗负结果的正式呈现。
3. 引用数据集论文自己的 BCI illiteracy 提议（§2.1）作为呼应。

### P4 — 延后项（原 P5）

1. **追查 S4 对角化来源**：查 ERP-Benchmark（arXiv:2601.00573）与 Al-Masud et al.
   *"Benchmarking ECG FMs"*（2026）官方代码，确认是有意的 S4D 简化还是移植疏漏。
   答案决定下一条的表述（"修正了 bug" vs "补上了低秩项"）。
2. **`s4erp_dplr` vs `s4erp`**：补上低秩项，看真 S4 对 MI 是否更好。独立可发表的消融。
3. 扩到 `split_seeds: [41..45]`（与 ERP-Benchmark 完全对齐）。
4. 补跑 AdamW + CosineAnnealingLR 配方，把"排名依赖配方"作为独立发现
   （`RESULTS_TRUTH` §5 已有 ATCNet +2.46pp / S4ERP −3.58pp 的证据）。

---

## 8.6 全部实验清单审计

| 实验 | 出处 | 落在 | 状态 |
|:---|:---|:---|:---|
| WBCIC-2C 跨被试 benchmark | 我提，用户批 | **P2 组 1** | 待跑 |
| SHU 跨被试 benchmark（新 run_id） | 我提，用户批 | **P2 组 1** | 待跑 |
| `dualcd_s4_timepatch` 补齐 + MI 分箱 | 决定 6 / 17 | **P2 组 1** | 待跑 |
| **`s4erp_dw`：空间滤波换 ATCNet depthwise，3 变体** | **决定 22（用户 08-26 第 1 点）** | **P2 组 2（融合主线）** | 待跑 |
| `atcnet_s4`：共享因果 S4 替 5 个 TCN | 我提 08-26 | **P2 组 3** | 待跑 |
| `tcn_filters 32→8` | 决定 23（用户 08-26 第 4 点） | **P2 组 4** | 待跑 |
| `n_windows=1` | `ANALYSIS` §1.3 假设 H2 | **P2 组 4** | 待跑 |
| `attention=null` | `ANALYSIS` §1.4 假设 H3 | **P2 组 4** | 待跑 |
| 2C 被试内天花板（合并全 session） | 我提，§1.3 证明 3C 版数据饥饿 | **P3-B-1** | 条件执行 |
| `s4erp_dplr` vs `s4erp` | 决定 12 | **P4-2** | 延后 |
| 追查 S4 对角化来源 | 决定 12 | **P4-1** | 延后 |
| 扩到 `split_seeds: 41–45` | 决定 2 | **P4-3** | 延后 |
| AdamW+cosine 配方对照 | `RESULTS_TRUTH` §5 | **P4-4** | 延后 |
| 3C LOSO | 决定 7 | P3-M 可能回归 | 本轮不用 |

**已明确取消（不要再提）**：ATCNet-Adv 改 λ 调度重跑、CDAN 条件对抗、
任何 domain-adversarial 变体 —— 决定 4 + §1.2 三条负证据。

**~~两项候选~~ 已于 2026-08-27 决定不做，正式划掉**：

| # | 候选 | 出处 | 决定 |
|:--|:---|:---|:---|
| ~~U1~~ | ~~源被试筛选 / 加权~~ | `ANALYSIS` §2.4 候选 4 | **不做**（用户 08-27）。它会改变训练集构成，无法混进 P2 主表，只能作独立 arm；本轮不开这个 arm |
| ~~U2~~ | ~~Euclidean Alignment arm~~ | `CROSS_SUBJECT_PROTOCOL_MEMO` §4-Q4 | **不做**（用户 08-27）。它是预处理层 arm，会让实验网格 ×2；本轮不开 |

两者都不再是待办。若将来重启，注意它们都**改变实验网格**，必须作为独立 run_id
而不是并入 benchmark 主表。

---

## 9. 未决问题：无

§0 的 27 条决定覆盖了全部设计选择；§8.6 的 U1 / U2 已于 2026-08-27 决定不做。
P0 已完成，**下一步是 P1 smoke**。

**执行期间我会主动报告（不是问题，是我会盯的风险）**：

| 风险 | 触发条件 | 我的动作 |
|:---|:---|:---|
| 组 2/3/4 成本估算偏差 | P1 实测与 §7 估算差 >50% | 先报修正预算，再决定是否排产 P2 |
| 新模型有 bug | P1 smoke 未过验收清单 | 修完重跑 smoke，不带病进 P2 |
| `atcnet_s4` 的 `log_dt` 初始化 | 时间常数仍是 6.8–555 步 | 阻断，必须改对（否则负结果会被误读成"S4 不行"） |
| `dualcd_transformer` 撞 `MaxWall 48h` | 单 job 跑不完 10 个 cell | P0-5 的自动重投链应处理；失败则手动分批 |
| `fbcnet` 深度收敛 | 2C 上 best_epoch 接近 500 上限 | 报告并讨论是否单独放宽 `max_epochs` |
| `sub-024` 只有 1 个 session | 200 trial vs 别人 600 | per-subject 表标注，论文写明 |
| 显存不足 | 58ch 下 DINO 变体超 24 GB | 调 `micro_batch_per_model`（保持等效 batch 128），不降 `batch_size` |

---

## 10. 文件清单

**将新建**：
`code/configs/experiments/{wbci_2c_cross_subject_bench,shu_cross_subject_bench}.yaml`、
`code/configs/models/{s4erp_dw,atcnet_s4}.yaml`（含 3 个 `s4erp_dw` 变体段）、
`scripts/slurm/submit_bench.sh`、
`LICENSE`、`requirements.txt`、`environment.yml`、`pyproject.toml`、`REPRODUCE.md`、
新模块单元测试。

**将修改**：
`code/models/registry.py`（删 5 个 + 加 2 个新模型）、
`code/models/eeg_foundation/encoders.py`（**加** `DepthwiseShallowNetEmbedding`，不改现有类）、
`code/models/eeg_foundation/models.py`（**加** `S4ERPDW`）、
`code/models/eeg_foundation/adapter.py`（VARIANT_NAMES + 派发 + 清理 `is_dualcd` 陷阱）、
`code/models/atcnet/{atcnet_torch,adapter}.py`（加 `atcnet_s4`）、
`code/models/{deepconvnet,fbcnet}.py`（`describe()` + `default_lr()`）、
`code/experiments/session_protocols.py`（Kappa）、
`code/experiments/cross_subject_protocols.py`（`cell_signature` + per-trial 保存）、
`code/runners.py`（`per_dataset` 合并 + 模型别名）、
`scripts/summarize_cross_subject.py`、`code/configs/models/*.yaml`。

**只读引用**：`docs/RESULTS_TRUTH.md`、`docs/ANALYSIS_atcnet_and_adversarial.md`、
`docs/CROSS_SUBJECT_PROTOCOL_MEMO.md`（原 `4_experiments/`）、`AGENTS.md`。

**参考文献**：
- Yang et al., *Sci Data* 12:488 (2025) — WBCIC-SHU，doi:10.1038/s41597-025-04826-y
- ERP-Benchmark, arXiv:2601.00573 — 指标 / seeds 41–45 / 6:2:2 / S4-DINO 包源头
- Lou et al., *IEEE JBHI* 2026 — DSGNet，doi:10.1109/jbhi.2026.3689121（3C 对标，本轮不用）
- Gu, Goel & Ré, ICLR 2022 — S4 原文（**注意 §5.5：我们的实现不是它**）
- Altaheri 2023 ATCNet；Lawhern 2018 EEGNet；Chen 2024 EEGNeX；Ding 2024 EEG-Deformer；
  Schirrmeister 2017 DeepConvNet；Mane 2021 FBCNet

**开工后同步更新**（AGENTS §8.5）：`0_docs/FILE_CATALOG.md`、
`0_docs/operation_log.md`、`0_docs/STATUS.md`、`progress.md`、`AGENTS.md`。
