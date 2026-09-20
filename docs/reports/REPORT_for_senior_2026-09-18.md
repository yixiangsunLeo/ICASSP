---
title: "给学长的报告：Spiking EEGNeX 并行方案（2026-09-18 凌晨，v2）"
tags:
  - "#modality/eeg"
  - "#method/snn"
created: "2026-09-18"
updated: "2026-09-18"
status: "archived"
---

# 给学长的报告：Spiking EEGNeX 并行方案 · 做了什么、为什么、在跑什么（2026-09-18 凌晨，v2）

> **路径说明（2026-09-18 17:30 迁移后）**：文中 `eeg-mi-online/…` 与 `SYX/eeg-mi-online/…` 现为 `SYX/ICASSP/…`（本仓库根）。
> 先导结果与主实验状态的最新版见 `docs/STATUS.md` 与 `docs/PLAN_spiking_eegnex_icassp2027.md` §0.5。
>
> 这是与学长"改 EEGNeX → 加点别的 → 套到 SNN"并行的一版尝试，**最终采不采用由学长定**。
> 一切都跑在原来的 `eeg-mi-online` 框架、同样的被试划分上，学长自己的改法注册成一个 model entry
> 就能进同一张表、和这里的所有模型逐被试配对比较。截稿：ICASSP 2027 已延期到 **9/24 20:00（北京）**。

---

## 1. 一句话

把 EEGNeX 逐层脉冲化，**让 EEG 的采样轴直接当脉冲时间轴**（不重复输入 T 次），配上无标签的被试级对齐
（EA + AdaBN），在 87 人 / 3 个队列上与已有的 18 个 ANN 逐被试配对比较。方案先经四位独立"专家"评审
（SNN 算法 / EEG-BCI 信号 / 统计方法 / ICASSP 领域主席），按评审修了 11 处设计后再开跑。
SHU 先导实验（4 GPU）正在跑，预计 9/18 中午出结果，之后提交 2C / 3C 主实验。

## 1.1 评审说了什么、我改了什么（重点）

**最重要的一条（领域主席）**：Wu Dongrui 组 2025 年在 *Neural Networks* 已发表"PLIF 直接跑在 1000 个采样点上 + EA 做跨被试 MI"（Zhang et al.，代码公开），NiSNN-A（TNNLS 2025）已做 54 人 LOSO + 发放率能耗。
所以"时间原生脉冲化"本身**不能当作首创**。我们**仍然独有**、且 6 天内能拿到证据的：
① MI-BCI 里第一个与 **18 个强 ANN（含 ATCNet / EEG-Deformer / EEGNeX）逐被试配对、统计严格**的 SNN 比较（87 人、3 队列）；
② **"无标签对齐对 SNN 的增益是否大于对 ANN"**这个交互效应（阈值非线性对尺度敏感 → 应该更吃对齐），以前没人量化过。
→ 论文框架改为 **"非劣性 + 对齐交互"**（预注册非劣边界 1pp，2C 为确认队列，SHU/3C 为复现队列），能耗以 EEGNet 为参照写成"EEGNeX 的精度、EEGNet 的能耗"，乘数不进标题。预计接收概率：按原方案 4/10，改框架后 ~6/10。

**神经元层的 6 处缺陷（SNN 专家）**，都会让固定配方（Adam 1e-4 / patience 25）静默失败，已全部修掉：
τ 范围过大让 3/8 个编码通道天生哑火（τ ≥ 8 步的漏积分器对零均值振荡输出 ≈ 0）→ 逐层范围 [1.5,4] / [1.5,8]；
深层预激活初始时在阈值下 2–3.5σ → 送入神经元的 BN 初始化 β=+0.5（池化后 γ=2）+ 稳态发放率正则；
二值读出 + FC max_norm(0.25) 把 logit 卡在 ~1.1 → 读出层改为非脉冲漏积分器（膜电位读出），二值读出留作消融；
脉冲上的 dropout 重标定 → 只做掩码；深层改软复位；删掉 chunk 编码。

**信号层（EEG 专家）**："1-bit 编码保相位所以空间滤波仍可行"不能当定理——改成假设，用**编码器位置消融**检验（`s_hplif_afe`：空间滤波前不发脉冲）；
EA 之后**不能**再逐通道 z-score（会把刚白化的协方差再打乱）→ 改为逐 trial 标量归一化；新增 8–30 Hz 参考协方差的 EA 变体；
EA 对 2C 的预期增益只有 +0.5–2pp（30 个源被试已经学到部分协方差不变性），别指望文献里的 +4–5pp；
"posterior drift"的说法与我们自己 Phase-2c 的结论矛盾，已删。

**统计（方法专家）**：配对设计有效且功效比我想的高（2C 上同侪模型的逐被试配对 SD 只有 1.4–2.5pp → 可检出 0.6–1.0pp）；
H0 必须写成非劣性（单侧 CI vs 边界）；H1 只保留**一个**主检验（S-EEGNeX-S@hplif，EA+AdaBN vs 无，accuracy，2C），其余全部标探索性；
"对齐后的 SNN vs 未对齐的 18 个 ANN"不成立 → 主表分两块（未对齐块 / 对齐 2×2 块），并给 EEGNet 也跑对齐臂；单训练 seed 写进 limitation（预算允许则给主对照补 2 个 seed）。

四份评审原文保存在会话记录里，方案 v2 的 §0 有逐条对照表：`eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md`。

---

## 2. 从第一性原理出发的三个判断（详见 `eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md`）

**判断 1：跨被试的瓶颈是被试偏移，不是模型容量。** 我们自己的 18 模型 × 3 队列结果：2C 上前 8 名相差 1.0pp、
被试间标准差 12pp；67 M 参数的模型垫底、3 k 参数的 EEGNet 距第一 1pp；所有模型 train–test gap 15–33pp；
域对抗在强骨干上是净损失。→ 有效的操作是**把每个被试的输入拉回同一参考系**，而不是换更大的网络。

**判断 2：EEGNeX 之所以最强，是因为它是可学习的 FBCSP + 多时间尺度包络。** 两层 64 点时域卷积 = 陡滤波器组；
depthwise 空间卷积 = 每频带各自的空间滤波；ELU + avgpool = 整流 + 平滑 = 带功率包络；两个膨胀卷积 = ERD/ERS 的两个时间尺度。
它在 3 个队列上都是第一或并列第一，且只有 58 k 参数、GPU 时间是 ATCNet 的 1/4。

**判断 3：LIF 神经元在物理上就是这条流水线需要的非线性。**
- LIF = 整流 + 漏积分 + 阈值 = **包络检测器**，脉冲计数 ∝ 带功率（FBCSP 的 log-var 特征由神经元算出来）；
- 第一层 LIF 跑在 250 Hz 采样率上 = **1-bit 电平交叉编码器**：幅度进脉冲率、相位进脉冲时刻，后面 64 点卷积作用在脉冲串上等价于重建滤波，
  所以空间滤波（依赖电极间相位）仍可行；
- 事件驱动：EEGNeX 993 M MACs 里 95% 在 conv2；conv2 改吃脉冲后代价变为 `950 M × 发放率` 次累加。
  冒烟实测发放率 ~5% → **能耗比 ANN 低 ~20×**（标准的"重复输入 T=4 次"编码只有 ~10×）。

**SNN 的软肋反过来是可检验的机制假设 H1**：阈值是尺度敏感的非线性，测试被试的电流分布一偏，发放率整体偏 →
**无标签对齐对 SNN 的增益应大于对 ANN**。这是 SNN 专属、以前没人量化过的结论，且能用发放率的被试间分布直接画出来。

---

## 3. 具体做了什么（代码，全部在 `SYX/eeg-mi-online/`）

| 文件 | 内容 |
|:---|:---|
| `code/models/spiking/neurons.py` | 自写 LIF / PLIF / **HetPLIF（每通道可学习 τ，逐层对数均匀初始化：编码层 [1.5,4]、深层 [1.5,8] 步）**，ATan 代理梯度，硬/软复位，非脉冲 LI 读出模式，mask-only `SpikeDropout`，TorchScript 融合循环；无 spikingjelly 依赖 |
| `code/models/spiking/models.py` | **S-EEGNeX** 与 **S-EEGNet**：与官方 ANN 完全相同的权重形状（参数只多几个 τ）。两种编码：`repeat`（文献标准 T=4 直接编码）、**`stream`（时间原生：三层神经元分别跑 1000 / 250 / 63 步；'same' 填充与 EEGNeX 一致，因果版为消融）**。LI 膜电位读出、送入神经元的 BN 初始化、编码器位置可选（`conv1` / `spatial`）。逐层 MAC/AC 表 + 由实测发放率算能耗 |
| `code/models/spiking/adapter.py` + `registry.py` | 接入项目的 `{logits, features, confidence}` 契约；`describe()` 报参数/τ/能耗；`pop_spike_stats()` 供 runner 逐被试记录发放率；`training_step` = CE + 稳态发放率正则 |
| `code/configs/models/seegnex.yaml`, `seegnet.yaml` | 变体：`hplif`（D 版）、`s_lif / s_plif / s_hplif`（S 版三种神经元）、单因子消融 `s_hplif_{afe,spk,causal,hard,noinit,noreg,all}` |
| `code/experiments/cross_subject_protocols.py` | ① 归一化语法 `ea_session[_mu]+per_sample_global`：**逐 session Euclidean Alignment**（复用 Phase-2b 的 `EuclideanAlignment`，10% 收缩；`_mu` = 参考协方差取 8–30 Hz 副本），之后只做逐 trial 标量归一化；无标签，无统计量跨被试；② **AdaBN**：评估时用测试被试自身无标签数据重估 BN 统计量，另记 `adabn_*` 列（不碰训练、不进 cell 签名）；③ 逐被试记录 `spike_stats`（发放率 + 能耗） |
| `code/training/e2e_trainer.py`, `code/runners.py` | `eval.adabn` 开关的传递 |
| `scripts/make_snn_configs.py` | 生成 12 个实验 config：{SHU, 2C, 3C} × {zs, ea} × {lr 1e-4, 1e-3}，`cross_subject` 块与 bench **完全一致**（同折同被试 → 与 18 个 ANN 配对） |
| `scripts/slurm/submit_snn.sh` | 按模型分组提交、48 h 接力 |
| `scripts/analyze_snn_round.py` | 预注册配对检验（Wilcoxon + Holm）、**H1 交互检验**（逐被试差之差）、能耗表、过拟合表、Friedman；bench 的 18 个 ANN 作 `+bench` 条目一并配对 |
| `tests/models/test_spiking.py` | 13 个单元测试：脚本/Python 循环数值一致、LIF 对 10 Hz 振荡的编码、HetPLIF 逐层初值、EA 白化（含带限参考）、全部变体前后向 + 自定义损失、因果版因果性、LI 读出/软复位/BN 初始化、mask-only dropout、S 版能耗 ≤ D 版。全套 158 测试通过 |

3C processed 数据已重生成（`outputs/processed/wbci_shu_3c_mat_clean/`，作业 70221）。

---

## 4. 冒烟测试（SHU fold 0，2 epoch，作业 70222，exit 0）

| 条目 | s/epoch（SHU） | 参数 | 发放率 (lif1/lif3/lif5) | 能耗比 ANN/SNN |
|:---|---:|---:|:---|---:|
| eegnex（ANN） | 5.7 | 57,106 | — | 1 |
| seegnex@hplif（D，T=4） | 21 | 57,186 | 0.068 / 0.037 / 0.040 | **9.7×** |
| seegnex@s_hplif（S，时间原生） | 63 | 57,186 | 0.051 / 0.032 / 0.041 | **22×** |
| seegnet@plif | ~10 | 1,458 | 0.058 / 0.041 | 0.99×（EEGNet 的干路在空间卷积前没有非线性，99% MACs 仍是模拟的——反过来说明 EEGNeX 的结构更适合脉冲化） |

2 个 epoch 的精度当然没有意义；这一步只验证：全部条目可训、有梯度、AdaBN/发放率/能耗都写进 `result.json`。

---

## 5. 正在跑的（4 GPU，QOS 上限；v2 代码，02:20 提交）

SHU 先导（split 41，5 折），5 个作业（4 个并发，第 5 个排队）：

| 作业 | 臂 | 条目 | 目的 |
|:---|:---|:---|:---|
| pz3 (70306) / pz4 (70308) | z-score × lr 1e-3 / 1e-4 | eegnex, eegnet_official, seegnex@hplif, seegnex@s_hplif, seegnet@plif | **按验证集 macro-F1 选 SNN 学习率** |
| pe3 (70310) | EA(+逐 trial 标量) × 1e-3 | 同上 | EA 的方向与幅度（ANN 与 SNN 都有） |
| pm3 (70313) | EA-μ（8–30 Hz 参考协方差）× 1e-3 | eegnex, seegnex@s_hplif, seegnet@plif | 选 EA 变体 |
| pa3 (70315) | z-score × 1e-3 | s_hplif_afe（编码位置）、s_hplif_spk（二值读出）、s_hplif_noinit（普通 BN 初始化） | 三个单因子消融，其中 afe 直接检验"1-bit 编码在空间滤波前是否损失精度" |
| pq3 (70339) | EA + 逐通道 z-score × 1e-3 | eegnex, seegnex@s_hplif, seegnet@plif | 经验臂：EEG 专家说 EA 后不该再逐通道 z-score，但最早跑完的 4 折 eegnex 显示这个组合比纯 z-score 高 ~1pp、比 EA+标量高，所以留下来让数据说话 |

之前用修订前代码跑的先导已取消；其中 eegnex 的 8 个 z-score cell 与模型无关、直接复用。这些先导 cell 都会成为最终 SHU 表的一部分（之后只需补 split 42 与其余变体）。

主实验（先导后提交，预计 9/18 中午）：2C + SHU + 3C，split 41/42，`{eegnex, eegnet_official, seegnex@hplif, seegnex@s_hplif, seegnet@plif}` × `{zs, 选定的 EA}`；
`s_lif / s_plif`（H3 神经元消融）只在 SHU 上跑；BCI IV 2a LOSO 锚点（loader 半天，截止 9/20 早）。估算 ~250 GPU-h ≈ 4 卡 2.6 天 → 9/21 出齐，9/22–24 写作。

**取消方法**：`scancel -u yuan`（全部）或 `scancel <jobid>`；作业清单 `eeg-mi-online/logs/slurm/snn_*_job_ids.txt`。

---

## 6. 需要学长拍板 / 可能不同意的地方

1. **论文框架**：评审建议从"提出时间原生脉冲化"改为"**非劣性 + 对齐交互**"（因为 Wu 组 2025 已发表时间原生 PLIF + EA）。这是最大的方向性决定，学长定。
2. **transductive 对齐**：EA 与 AdaBN 都用测试被试的**无标签**数据（各自 session 的协方差 / BN 统计量）。文献标准做法，但必须写明；为公平，EEGNeX 与 EEGNet 两个 ANN 都跑同样的对齐臂；主表分"未对齐块 / 对齐块"两块。
3. **学习率**：bench 配方 1e-4；SNN 文献 1e-3。用 SHU 先导按验证集选，不看测试集。
4. **S 版比 D 版慢 3×**（顺序循环 1000 步是 launch-bound；SHU 上 63 s/epoch vs ANN 5.7 s）。预算够；要扩更多变体可再做 CUDA graph 优化。
5. **单训练 seed、2 个划分 seed**：沿用 bench 口径；预算允许给 2C 的主 2×2 对照补 2 个 seed，否则写 limitation。
6. **BCI IV 2a**：三位评审都提；EEG-FM-Bench 项目已有 pkl，接入需写 loader（~半天），我计划 9/19 做，截止 9/20 早。
7. **能耗表述**：以 EEGNet ANN / S-EEGNet 为参照（"EEGNeX 的精度、EEGNet 的能耗"），报 SynOps 为主、pJ 为辅，不写"20×"进标题。

---

## 7. 文件索引

- 方案：`eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md`
- 梳理简报（数据集/模型/结果清单）：`SYX/MEETING_BRIEF_2026-09-18.md`
- 实验 config：`eeg-mi-online/code/configs/experiments/snn_*.yaml`
- 结果目录：`eeg-mi-online/outputs/experiments/{shu,wbci_shu}/snn_*_v1/`
- 分析：`python scripts/analyze_snn_round.py --dataset shu --run zs=<dir> --run ea=<dir> --bench results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1 --out outputs/analysis/snn_round/shu`
