---
title: "Spiking EEGNeX for cross-subject MI — ICASSP 2027 方案（v2，已按四位专家评审修订）"
created: "2026-09-18"
updated: "2026-09-18 19:10"
status: "v2_main_run_running"
deadline: "2026-09-23 23:59 AoE = 2026-09-24 20:00 北京时间（官网明示不再延期）"
---

# Spiking EEGNeX for Cross-Subject Motor Imagery — 方案 v2

> 学长的原话：「那个 eegnex 我看看能不能改一下 / 加点别的 / 再套到 snn 上」。
> 本方案是一条与学长并行的版本，完全跑在同一框架、同一被试划分上，所以学长的改法可以作为
> 另一个 entry 加进同一张表、和这里的所有模型逐被试配对比较。
>
> **v2 说明**：v1 交给四位专家（SNN 算法 / EEG-BCI 信号 / 统计方法 / ICASSP 领域主席）独立评审后修订。
> §0 是评审结论与本轮采纳的变更；§1–§7 是 v1 正文，其中被推翻的表述用 ~~删除线~~ 标出并给出替换。

---

## 0. 评审结论与 v2 变更（2026-09-18）

### 0.1 四份评审的一致结论

| 评审 | 关键判断 | 分数 |
|:---|:---|:---|
| ICASSP 领域主席 | 贡献 (1)(4)(5) 已有先例：**Zhang, Wang, An, Zheng, Wu, *Neural Networks* 191:107741 (2025)**——PLIF 直接跑在 1000 个原始采样点上 + `--EA` 开关做跨被试 MI，代码公开；**NiSNN-A (TNNLS 2025)** 已做 54 人 LOSO + 发放率能耗。仍然成立且 6 天内可支撑的：MI-BCI 里**第一个**与强 ANN 逐被试配对、统计严格的 SNN 比较；对齐 × 网络类型的**交互效应**。建议改成"非劣性 + 对齐交互"框架 + BCI IV 2a 锚点 | 按 v1 写 4/10，改框架后 ~6/10 |
| SNN 算法专家 | 评估框架远强于 SNN-MI 文献；但神经元层有 6 处会让固定配方**静默失败**的缺陷（见 0.2）；τ ≥ 8 步的漏积分器对零均值振荡输出 ≈ 0，永不过阈——"LIF = 包络检测器"的顺序写反了（整流发生在阈值处，不在积分里）；能耗算法正确，S 版 15–20× 合理但模型是**混合体**（conv1 / depthwise / conv4 仍是模拟层）| 6/10 → 修后 8/10；新颖性 5/10 |
| EEG-BCI 信号专家 | "1-bit 编码保相位 → 空间滤波仍可行"作为定理**不成立**（过阈相位依赖幅度、半波、逐电极非线性破坏线性解混），应改为假设并用**编码器位置消融**检验；能耗对照选错了（EEGNeX 是 EEGNet 的 32× MACs 换 +0.8pp，正确说法是"EEGNeX 精度、EEGNet 能耗"）；"posterior drift"与 H1 及 Phase-2c 结论（类内散布膨胀、非中心塌缩）自相矛盾；EA 后**不能**再逐通道 z-score；EA 对 2C 的预期增益只有 +0.5–2pp | 6/10 → 修后 8/10；对齐增益预期 5/10 |
| 统计方法专家 | 配对设计有效且比 v1 假设的功效高 3–4×（2C 同侪模型配对 SD 1.4–2.5pp → MDE 0.6–1.0pp）；但 H0 是非劣性却用了双侧优效检验；H1 有 ~450 条未固定的检验路径且受基线水平混杂；"对齐后的 SNN vs 未对齐的 18 个 ANN"不成立；单训练 seed 且所有 cell 共享同一初始化 | 6/10 → 修后 8/10 |

### 0.2 采纳的代码变更（全部已实现、通过 158 个测试、GPU 冒烟中）

| # | 变更 | 来源 |
|:--|:---|:---|
| 1 | **HetPLIF 逐层 τ 初值范围**：编码层 [1.5, 4] 步（6–16 ms @250 Hz），深层 [1.5, 8] 步。v1 的 [1.5, 16] 会让 3/8 个编码通道天生哑火 | SNN A1/A2 |
| 2 | **送入神经元的 BN 初始化** β = +0.5（池化后再进神经元的 γ = 2）：否则深层预激活在阈值下 2–3.5σ，Adam-1e-4 + patience 25 会把它停在 chance | SNN A3 |
| 3 | **读出层改为非脉冲漏积分器（LI 膜电位读出）**；二值读出保留为消融 `s_hplif_spk`。原因：8×63 bit、~4% 发放率 → 只剩 ~20 个有效特征，加上 FC 的 max_norm(0.25) 把 logit 卡在 ~1.1 | SNN A4 / EEG A5 |
| 4 | **深层软复位**（rate ≈ 线性于输入），编码层保持硬复位（每周期一发、最低能耗） | SNN B2 |
| 5 | **脉冲上的 dropout 只做掩码不重标定**（`SpikeDropout`），否则下游 BN 的训练方差是测试的 2 倍 → 阈值把 30% 的尺度差放大成数倍发放率漂移 | SNN A5 |
| 6 | **删除 chunk 编码**（窗口边界破坏卷积感受野、不是流式）；主 S 模型改用与 EEGNeX 相同的 'same' 填充（唯一差别就是神经元），因果版 `s_hplif_causal` 作消融 | SNN A6 / EEG A4 |
| 7 | **编码器位置消融 `s_hplif_afe`**（conv1→conv2→depthwise 全模拟，池化后才发脉冲）：检验"1-bit 编码在空间滤波前是否损失精度"，代价是能耗故事消失 | EEG B1 |
| 8 | **稳态发放率正则** λΣ(r̄ₗ − 0.1)²，λ = 1e-3（Zenke & Vogels 2021 的活动边界）：防止某层哑火/饱和 | SNN §2.2(c) |
| 9 | ~~**EA 之后用逐 trial 标量归一化 `per_sample_global`**（不再逐通道 z-score）~~ → **已被 §0.5 / D-021 推翻**：先导数据显示 EA + 逐通道 z-score（`eaz`）优于 EA + 标量，主实验对齐臂 = `eaz`；`per_sample_global` 只用于先导 `ea` 臂。新增 `ea_session_mu`（参考协方差取自 8–30 Hz 副本）也被先导否定（−1.3pp）| EEG B5/B6 |
| 10 | 新增 `eegnet_official` 到所有臂（对齐臂的第二个 ANN 锚点）；`curves: false`（先导期间不打印测试曲线；每次 eval 都要重跑 1000 步循环）| EEG B4 / 统计 B7 |
| 11 | `describe()` 的能耗上界改为显式 `dense=True`；`snn_sops` 列 | SNN §6 |

### 0.3 采纳的框架与统计变更

- **论文框架**：~~"时间原生脉冲化 + 异质 τ + 能耗"~~ → **"非劣性 + 对齐交互"**：S-EEGNeX 在 2C 上与 EEGNeX 非劣（预注册边界 δ = 1pp = 2C 前 8 名 ANN 的总极差），配对对照 17 个其余 ANN；无标签对齐对 SNN 的增益大于对 ANN（差之差 + 95% CI）；SynOps/能耗以 **EEGNet ANN 与 S-EEGNet 为参照**（"EEGNeX 的精度、EEGNet 的能耗"），乘数不进标题/摘要。
- **必须引用并明确对比**：Zhang et al. 2025 (*Neural Networks*, Wu 组)、NiSNN-A (TNNLS 2025)、HR-SNN (TCDS 2024)、LENet / SRIF（转换法）、Yin–Corradi–Bohté 2021、Perez-Nieves 2021、PLIF (ICCV 2021)、He & Wu 2020、Junqueira 2024、Wimpff 2024、Horowitz 2014 + Dampfhoffer 2023。
- **预注册家族（2C，Holm，α = 0.05，5 项）**：H0-NI（S@hplif vs eegnex，未对齐，δ = 1pp，单侧 bootstrap 下界）；H1（差之差 Wilcoxon，**唯一主检验**：S-EEGNeX-S@hplif，EA+AdaBN vs 无，accuracy，best-val，2C，以 H0 成立为前提）；H2-NI（S vs D，δ = 1pp）；H3（Page's L 有序检验 LIF ≤ PLIF ≤ HetPLIF）；B1（S-EEGNeX vs S-EEGNet）。SHU / 3C 为复现队列（同样 5 项，队列内 Holm，摘要不依赖它们）；其余全部标为探索性。
- **两块主表**：未对齐块（18 个 ANN + 未对齐 SNN，配对 CD 图，Bonferroni–Dunn 以 S-EEGNeX 为对照，k = 19）；对齐 2×2 块（{eegnex, eegnet_official, S-EEGNeX} × {无, EA, EA+AdaBN}）。**绝不**把对齐后的 SNN 与未对齐的旧 ANN 直接比较。
- **检验实现**：n ≤ 50 用精确 Wilcoxon、n = 51 用置换；Hodges–Lehmann 中位差 + BCa bootstrap CI；报告 n_eff、胜/平/负；SHU 主指标改 balanced accuracy（67/125 个 session 不平衡 + AdaBN 对类先验敏感）；覆盖断言 `n_cells == n_folds × n_splits`。
- **训练 seed**：若预算允许，2C 主 2×2 对照加 2 个训练 seed（≈ 30–60 GPU-h）；否则写进 limitation（单 seed、共享初始化）。
- **BCI IV 2a LOSO 锚点**（9 人，2 类 LH/RH + 4 类）：loader 半天，截止 9/20 早；≈ 5 GPU-h。
- **删除的表述**：~~"posterior drift"~~ → "被试特异的增益/协方差偏移 + 类内散布膨胀"；~~"anytime / streaming"~~ → "sample-synchronous，'same' 填充下 look-ahead ≤ 128 ms；因果版为消融"；~~"LIF = 包络检测器"~~ → "LIF₁ 在阈值处整流 → conv2 平滑非负脉冲串 = 包络估计 → LIF₂"；~~"FBCSP log-var 等价"~~ → "阈上单调幅度编码，阈下有死区"；~~"1-bit 编码保相位所以空间滤波仍可行"~~ → 假设，由 `s_hplif_afe` 消融检验，并画脉冲–相位锁定直方图。
- **H2 改写**：预期 D ≥ S 精度、S ≫ D 能耗；写成"S 在 D 的 1pp 内、SOPs 少 ~4×"。
- **H4 降级**为观察（train 精度低的模型 gap 自然小）。

### 0.4 v2 先导设计（SHU，split 41，5 折；已提交 2026-09-18 02:20）

| 作业 | 臂 | 条目 |
|:---|:---|:---|
| pz3 / pz4 | z-score × lr {1e-3, 1e-4} | eegnex, eegnet_official, seegnex@hplif, seegnex@s_hplif, seegnet@plif |
| pe3 | EA(+标量) × lr 1e-3 | 同上 |
| pm3 | EA-μ(8–30 Hz 参考) × lr 1e-3 | eegnex, seegnex@s_hplif, seegnet@plif |
| pa3 | z-score × lr 1e-3 | 消融：s_hplif_afe（编码位置）、s_hplif_spk（二值读出）、s_hplif_noinit（普通 BN 初始化） |
| pq3（补提，作业 70339/70340） | EA + 逐通道 z-score（`eaz`）× lr 1e-3 | eegnex, seegnex@s_hplif, seegnet@plif —— 经验臂：最早跑完的 eegnex cell 显示这个组合比纯 z-score 高 ~1pp，与 EEG 专家"EA 后不该再逐通道 z-score"的判断相反，留下让数据说话 |

决策规则（**只看验证集 macro-F1**）：lr 取 SNN 验证分更高者；EA 臂在 ea / eamu / eaz 中选验证分更高者进入主实验；`afe` 若显著优于主模型则编码器理论被证伪、主模型换为 afe 并如实报告能耗后果。

### 0.5 先导结果（2026-09-18 16:50，SHU split 41，25 人，120 cell 全部完成）与主实验决定

数字 = 逐被试准确率均值；Δ 为逐被试配对差（Wilcoxon）。出处（2026-09-18 19:00 用带臂映射的脚本重算，逐行可查）：
`outputs/analysis/snn_round/shu_pilot_lr1e4/paired_tests.csv`（lr 1e-4：S / D vs EEGNeX、S vs D、S-EEGNet vs EEGNet）、
`outputs/analysis/snn_round/shu_pilot_lr1e3/{paired_tests,interaction_H1}.csv`（lr 1e-3：zs vs eaz 的 EA 效应、H1 交互、AdaBN、afe/spk/noinit 消融）；
`shu_pilot/` 是 5 臂合表，只用其 `per_model.csv` / `energy.csv`。注意 EA 变体之间按验证集 macro-F1 选 eaz，但验证集上 zs 仍高于所有 EA 臂
（eegnex 0.675 vs 0.669；S 0.658 vs 0.646）——"EA 是否优于 zs"由 2C 判；afe/spk/noinit 消融只在 lr 1e-3 跑过，配对对象是 lr 1e-3 的 S-EEGNeX-S。

| 问题 | 结果 | 决定 |
|:---|:---|:---|
| lr | 1e-4 优于 1e-3：eegnex +1.1pp、S-EEGNeX(S) +0.7、(D) +1.3；验证集 macro-F1 同向（eegnex 0.681 vs 0.675，S 0.661 vs 0.658） | **主实验 lr = 1e-4**（= bench 配方，与 18 个旧 baseline 完全同配方） |
| EA 变体（lr 1e-3） | EA+逐通道 z-score（eaz）：ANN +1.39pp、SNN +0.87pp；EA+标量：0 / −0.98；EA-μ：−1.26 / −0.15。验证集同序（eegnex 0.669 / 0.655 / 0.650） | **对齐臂 = eaz**。EEG 专家"EA 后不可逐通道 z-score"的判断被数据否定，写进论文的 preprocessing 讨论 |
| SNN vs ANN（1e-4，z-score） | D 版 −0.48pp（CI [−1.84, +0.88]，p=0.17）；S 版 −1.38pp（p=0.039）；S vs D −0.90pp（p=0.07）；S-EEGNet vs EEGNet −0.94pp | 非劣性在 n=25 上不可判定（MDE ≈ 2pp），由 2C 决定；预期 D 非劣、S 略低 |
| H1 交互（eaz，1e-3） | ANN 增益 +1.39，SNN 增益 +0.87，交互 −0.51pp（p=0.35） | SHU 上无支持；2C 为确认队列，结果按观察报告 |
| AdaBN | ANN@eaz +0.63pp（p=0.038）；z-score 臂 −0.07；SNN +0.1–0.3 | 保留为评估开关，效应小 |
| 编码器位置 | 模拟前端（afe）比 conv1 后发脉冲 **−1.82pp（p=0.027）** | 1-bit 编码放在空间滤波前**不损失、反而更好** → 编码器假设成立，主模型不变 |
| 读出 / 初始化 | 二值读出 +0.50（n.s.）；普通 BN 初始化 −0.30（n.s.） | 主模型保持 LI 读出 + BN 初始化（预注册，不按测试集改） |
| 能耗（训练后实测发放率 11–20%） | S 版 148 µJ vs EEGNeX 2540 µJ（**17×**）；D 版 7×；S-EEGNet 0.99×；EEGNet ANN 78 µJ | "EEGNeX 的精度、~2× EEGNet 的能耗、比 EEGNeX 低 17×" |
| 复现性 | eegnex@zs(1e-4, split 41) 64.65 vs bench 65.49（2 splits） | 同配方，差异在单划分噪声内；split 42 跑完再比 |

**主实验（17:00 提交，8 组共 20 个作业，每组 2–3 个 48 h 接力）**：lr 1e-4，臂 {zs, eaz}，split 41/42：
2C `{seegnex@s_hplif, seegnex@hplif}` / `{eegnex, eegnet_official, seegnet@plif}` 各一作业 × 2 臂（c2z/c2e）；
3C LOSO 5 个模型 × 2 臂（c3z/c3e）；SHU 补 split 42 + `s_lif/s_plif`（H3）× 2 臂（shz/she）。
估算 ~156 GPU-h；集群当日很满，实际墙钟取决于排队。

---

## 0. 约束与资产

| 项 | 值 |
|:---|:---|
| 截稿 | 9/24 20:00（北京），剩 ~6.5 天；4 页正文 + 1 页参考文献 |
| 算力 | QOS 上限 4 GPU（RTX 4090D 24 GB）并发、48 h/作业；集群当前 40 卡仅 ~4 张空闲，需排队 |
| 现成资产 | 3 个跨被试队列（87 人）× 18 个 ANN 条目的完整配对结果（同折同划分，`split_seeds 41/42`）；EEGNeX 官方 Keras 逐层移植的 PyTorch 版；被试级 k-fold/LOSO 框架；配对 Wilcoxon+Holm / Friedman+Nemenyi 汇总 |
| 缺的 | 所有 checkpoint / 逐 trial 预测（已删）；3C processed 数据（7 分钟重生成） |

**所有新模型的 ANN 参照 `eegnex` 会在新 run 里重跑**（同折），不依赖旧 checkpoint。

---

## 1. 第一性原理分析

> **以下 §1–§7 为 v1 原文（2026-09-18 00:40）。** 假设表（§3）、时间表（§4）、论文骨架（§6）与全部措辞以 §0.3–§0.5 为准；
> 被评审推翻的表述在下文用 ~~删除线~~ 标出并给出替换。

### 1.1 问题的结构：方差在被试，不在模型

自己的证据（`docs/RESULTS_TRUTH.md`、两批 bench 表）：

1. 2C 上前 8 名模型相差 1.0pp，被试间标准差 11.5–12pp；模型方差 / 被试方差 ≈ 1/45。
2. 所有模型严重过拟合训练被试：3C 上 train 80–98% vs test 65–70%（gap 15–33pp），且 gap 与参数量几乎无关。
3. 67 M 参数的 DINO 模型垫底，3 k 参数的 EEGNet 距第一 1pp → **容量不是瓶颈，跨被试分布偏移才是**。
4. 域对抗（学不变表示）在强骨干上是净损失（ATCNet −4.2pp）→ "抹掉被试信息" 的路线不对：~~跨被试偏移里含大量 posterior drift，不是纯 covariate shift~~ → 替换为「被试特异的增益 / 协方差偏移 + 类内散布膨胀」（与 Phase-2c 结论一致；"posterior drift" 与 H1 自相矛盾，见 §0.3）。
5. 3C 内 sub-011 被试内也只有 34.7%（chance 33.3%）→ 一部分被试根本没有可解码信号（BCI illiteracy），任何模型都救不了；模型比较必须配对、逐被试。

**推论**：能提升跨被试精度的操作，应当是**把每个被试的输入拉回同一参考系**（对齐），而不是让模型更大或更"不变"。

### 1.2 EEGNeX 为什么强：它是可学习的 FBCSP + 多尺度时程

`EEGNeX_8_32`：conv(1×64)×8 → BN → ELU → conv(1×64)×32 → BN → depthwise 空间卷积(C×1)×2 → BN → ELU → avgpool(4) → dilated conv(1×16, d=2) → BN → dilated conv(1×16, d=4) → BN → ELU → avgpool(4) → flatten → FC。

- 两层 64 点时域卷积 = 更陡的滤波器组（比 EEGNet 的单层滤波器频率选择性更好——μ/β 带的边界很重要）；
- depthwise 空间卷积 = 每个频带各自的空间滤波器（FBCSP 的结构先验）；
- ELU + avgpool = 整流 + 平滑 = **带功率包络**；
- 两个膨胀卷积（感受野 31 + 61 个池化后样本 ≈ 0.5–1 s）= ERD/ERS 的**多时间尺度时程**；
- flatten 保留时间位置（RESULTS_TRUTH §4.1 证明把时间抹平会崩）。

在 3 个队列上它都是第一或并列第一，并且只有 58 k 参数、GPU 时间是 ATCNet 的 1/4。**改它，是在改一个已知有效的先验结构，而不是从零开始。**

### 1.3 SNN 从物理上能给这条流水线带来什么

把 EEGNeX 的非线性换成 LIF 神经元，不是"换个激活函数"，有三层含义：

1. ~~**LIF = 整流 + 漏积分 + 阈值 → 就是包络检测器。**~~ → 替换为「LIF₁ 在阈值处整流 → conv2 平滑非负脉冲串 = 包络估计 → LIF₂」（SNN 评审：整流发生在阈值处、不在积分里；τ ≥ 8 步的漏积分器对零均值振荡输出 ≈ 0）。带通后的 EEG 是 ~10 Hz 振荡；LIF 在每个正半周充电、过阈值发放、复位——阈上脉冲计数随幅度单调增加，阈下有死区（~~FBCSP 的 `log(var)` 特征在这里由神经元硬件级地算出来~~ → 「阈上单调幅度编码，不是 log-var 等价」）。
2. **第一层 LIF 在采样率上运行 = 1-bit 电平交叉编码器（Σ-Δ / level-crossing ADC）。** 它每个周期在固定相位发一个脉冲：**幅度进了脉冲率、相位进了脉冲时刻**。后面的 64 点时域卷积作用在脉冲串上等价于重建滤波器，~~因此空间滤波（需要电极间相位关系）**仍然可行**~~ → 这只是**假设**（EEG 评审：过阈相位依赖幅度、半波、逐电极非线性会破坏线性解混），由编码器位置消融 `s_hplif_afe` 检验——先导结果：模拟前端反而 −1.82pp（p=0.027），假设得到支持。这正是神经形态 EEG 前端的编码方式。
3. **事件驱动 → 乘累加(MAC)变累加(AC)。** EEGNeX 的 993 M MACs 里 95% 在 conv2（作用于 conv1 输出）。如果 conv1 之后就是脉冲，conv2 的代价变为 `950 M × 发放率` 次 AC。振荡每周期一发，250 Hz 采样下发放率 ~0.04–0.1 → conv2 只需 40–95 M 次 AC。用 45 nm 口径（MAC 4.6 pJ / AC 0.9 pJ）估算，**整机能耗降 ~15–20×**；标准的"重复输入 T 次"直接编码只有 ~5×。

### 1.4 SNN 的软肋，反过来是本文的机制性发现

阈值是**尺度敏感**的非线性：同一权重下，输入电流整体放大 20%，发放率会非线性地变。ANN 的 ELU 近似线性、BN 又逐通道归一，对尺度/协方差偏移相对鲁棒；SNN 的 BN 统计量来自训练被试，测试被试的电流分布一偏，发放率整体偏移 → 读出层看到的是另一种"字典"。

**可检验假设 H1：无标签的被试级对齐（EA / AdaBN）对 SNN 的增益显著大于对 ANN 的增益（交互效应）。** 这是一条 SNN 专属、以前没人量化过的结论，且可以用发放率的被试间分布直接可视化（对齐前后每层平均发放率的被试间方差）。

---

## 2. 方法设计

### 2.1 两种脉冲化（同一套权重形状，参数量 = EEGNeX + 少量 τ）

| | **S-EEGNeX-D**（direct，文献标准口径） | **S-EEGNeX-S**（temporal-native / ~~streaming~~ → sample-synchronous，'same' 填充下 look-ahead ≤ 128 ms；本文主推） |
|:---|:---|:---|
| 时间轴 | 仿真步 T=4，每步输入同一 trial（常量电流） | **EEG 的采样轴就是脉冲时间轴**：LIF₁ 跑 1000 步 @250 Hz，池化后 LIF₂ 250 步，LIF₃ 63 步 |
| 卷积 | 'same' 填充（与 EEGNeX 完全一致） | **因果**填充 + 因果池化（流式可运行，`@same` 作消融） |
| 编码 | conv1+BN 算一次，常量电流注入 T 步 | conv1+BN 算一次，逐样本注入（1-bit 编码器） |
| 读出 | FC 逐步输出取均值（rate readout） | FC 读 flatten 后的脉冲计数图（与 EEGNeX 头相同） |
| conv2 代价 | 950 M × T × r₁ AC | 950 M × r₁ AC（**少 T 倍**） |
| 用途 | 与 SNN 文献可比的基线 | 精度 + 能耗 + 流式三合一的主模型 |

层映射（两者共用）：`conv1 BN LIF₁ | conv2 BN | dw BN pool LIF₂ drop | conv3 BN | conv4 BN pool LIF₃ drop | fc`。
LIF 放在（线性的）avgpool **之后**，保证所有下游层输入都是 0/1。EEGNeX 原本没有激活的地方（conv2 后、conv3 后）不加神经元（`spike_all` 作为"全脉冲化"消融）。

### 2.2 神经元：LIF / PLIF / **HetPLIF**（异质时间常数）

`H[t] = V[t-1] + (X[t] − V[t-1]) / τ`，`S = Θ(H − v_th)`，硬复位，ATan 代理梯度（α=2），detach reset。
- LIF：τ=2 固定；PLIF：每层一个可学习 τ；**HetPLIF：每个通道一个可学习 τ，初值在 [1.5, 16] 步内对数均匀分布**。
- 在 S 版里 τ 的单位是真实时间：LIF₂ 层 62.5 Hz 上 τ∈[24 ms, 260 ms]，覆盖"锁相"到"包络积分"两端。**HetPLIF 是 EEGNeX 两个膨胀卷积（d=2, 4 = 两个固定时间尺度）的神经形态对应物：一组连续可学的时间尺度。**
- 假设 H3：HetPLIF ≥ PLIF ≥ LIF。

### 2.3 "加点别的"：无标签被试级对齐，两层

| 层 | 方法 | 作用 | 口径 |
|:---|:---|:---|:---|
| 输入 | **Euclidean Alignment**（He & Wu, TBME 2020）逐 **session** 白化：`X' = R^{-1/2} X`，R 为该 session 全部 trial 的平均协方差（10% 收缩） | 消空间协方差偏移；输出协方差均值 = I，尺度也被归一 | 只用该 session 自身的无标签 trial；训练/验证/测试被试各自对齐，**无统计量跨被试** |
| 网络内 | **AdaBN**：测试被试的所有 BN 层用该被试无标签 trial 重估 μ/σ，仿射参数不动 | 把每层进阈值的电流重新对中 → **发放率稳态（homeostasis）** | 训练完成后按被试执行，不重训；对 ANN 和 SNN 同样做，H1 就是比较两者的增益 |

两者都是 transductive-unsupervised（用测试被试无标签数据），是跨被试 MI 文献的标准做法；论文里明确写出，并给出"只用前 k 个 trial 校准"的曲线作为在线可行性证据（时间允许时）。

### 2.4 能耗核算

每层记录：dense MACs、运行次数、输入是否为脉冲、来源神经元的实测平均发放率（逐测试被试记录进 `result.json`）。
`E_SNN = Σ_analog MACs·4.6 pJ + Σ_spike MACs·r·0.9 pJ`，`E_ANN = Σ MACs·4.6 pJ`。报告每 trial 能耗与比值。

---

## 3. 假设驱动的实验设计

| # | 假设 | 判据 | 所需 entry |
|:---|:---|:---|:---|
| H0 | S-EEGNeX 在 87 人 3 队列上与 EEGNeX 精度差 ≤ 1pp（非劣） | 配对 Wilcoxon 的 Δ 与 95% CI | eegnex, seegnex_stream@hplif |
| H1 | 对齐对 SNN 的增益 > 对 ANN | 逐被试 Δ(EA)−Δ(EA) 的配对检验；发放率被试间方差 | eegnex / seegnex_stream 各 × {zscore, EA} × {AdaBN off/on} |
| H2 | S 版精度 ≥ D 版，SOPs 少 ~T 倍 | 配对检验 + 能耗表 | seegnex@hplif vs seegnex_stream@hplif |
| H3 | HetPLIF ≥ PLIF ≥ LIF | 配对检验 | seegnex_stream@{lif,plif,hplif} |
| H4 | SNN 的 train–test gap < ANN（脉冲量化正则） | 已有 `curves: true`，取 best epoch 的 train/val/test | 同上 |
| H5 | 能耗 ≥ 10× 降低（S 版） | 实测发放率代入 | 同上 |
| 基线 | S-EEGNet（文献里最常见的 SNN-MI 骨干） | 同表 | seegnet@plif |

**协议**：与 bench 完全相同的 `kfold_subject` 6:2:2、`split_seeds [41, 42]`、单训练 seed、Adam、batch 128、patience 25（监控 macro-F1）、最多 500 epoch → 每个新 entry 与 18 个旧 entry 在相同 (split, fold, subject) 上配对。
**学习率**：SNN 文献惯用 1e-3，bench 配方是 1e-4。在 SHU 上用 `split_seed 41` 做先导（两档 lr 各一个 run_id），**按验证集 macro-F1** 选，绝不看测试集；选定后固定用于 2C / 3C。
**统计**：逐被试配对 Wilcoxon（approx）+ Holm（预注册家族），Friedman + Nemenyi CD 图，效应量 d_z 与 95% CI；三个队列分别报，不合并。
**报告口径**：Accuracy / Macro-F1 / AUROC（+ balanced acc, kappa）；best-val checkpoint 为主、last 并列。

### 3.1 数据集

| 队列 | 用法 | 备注 |
|:---|:---|:---|
| WBCIC-SHU 2C（51 人） | 主表 | 唯一能检出 ~2pp 差异的队列 |
| SHU 2022（25 人） | 主表 + 先导 | 地板效应，差异要谨慎读 |
| WBCIC-SHU 3C（11 人，LOSO） | 第三队列 / 三分类 | 需重生成 processed 数据；配方改为 bench 配方并重跑 eegnex 作对照 |

BCI IV 2a：评审默认期待，但接入本框架需新 loader；列为 P2（有余力再做）。

---

## 4. 算力与时间表

单 cell 成本估计（以 2C 上 `eegnex` 实测 31.6 s/epoch、中位 82 epoch 为基准）：D 版 T=4 ≈ 2.5×，S 版（三层顺序 LIF，1000/250/63 步的 Python 循环，JIT 融合后）≈ 2×。

| 阶段 | 内容 | GPU-h | 墙钟（4 卡） |
|:---|:---|---:|:---|
| P1 冒烟 | 全部 entry × fold 0 × 3 epoch，SHU | ~1 | 1 h |
| P2 先导 | SHU，split 41，5 折：{eegnex, seegnex@hplif, seegnex_stream@hplif, seegnet@plif} × {zscore, EA} × {lr 1e-4, 1e-3} = 16 config × 5 cell | ~25 | ~7 h |
| P3 主实验 | 2C + SHU + 3C，split 41/42：{eegnex, seegnex@hplif, seegnex_stream@{lif,plif,hplif}, seegnet@plif} × {zscore, EA} = 12 entry；AdaBN 为评估时开关不加训练成本 | 2C ~150 / SHU ~40 / 3C ~35 | ~2.5 天 |
| P4 消融（若有余力） | `@same`、`spike_all`、`T=8`、soft reset | ~40 | 0.5 天 |
| 分析 | 能耗、发放率-对齐图、CD 图、~~anytime 曲线~~（措辞已删，见 §0.3） | CPU | 0.5 天 |
| 写作 | 4 页 | — | 9/22–9/24 |

关键路径：**9/18 代码 + 冒烟 + 先导提交 → 9/19 主实验提交 → 9/21 晚结果 → 9/22–24 写作**。

---

## 5. 风险与预案

| 风险 | 概率 | 预案 |
|:---|:---|:---|
| S 版 Python 逐步循环太慢 | 中 | TorchScript 融合；仍慢则 LIF₁ 改在 avgpool 后（250 步）并如实报告编码位置 |
| SNN 精度掉 >3pp | 中 | 先导阶段用验证集选 lr / 神经元；D 版作兜底主模型 |
| EA 让 ANN 涨得一样多（H1 不成立） | 中 | 仍然是有效结论：SNN 达到 SOTA-ANN 水平 + 10× 能耗 + 流式；H1 改为报告为观察 |
| 集群排不到卡 | 高 | 先导用 4 卡；主实验按 2C→SHU→3C 优先级分组，任何时刻保持 4 个作业在队列 |
| transductive 对齐被审稿人质疑 | 中 | 明确写出；补"前 k trial 校准"曲线 |
| 单训练 seed | 已知 | 论文 limitation；沿用 bench 口径（2 个划分 seed） |

---

## 6. 论文骨架（4 页）

1. Intro：跨被试 MI 的瓶颈是被试偏移；SNN 的能耗/流式优势与阈值脆弱性；贡献三条。
2. Method：EEGNeX→S-EEGNeX 的层映射（图 1）；temporal-native 编码与 1-bit 编码器解释；HetPLIF；EA+AdaBN 作为"发放率稳态"。
3. Experiments：3 队列 87 人、18 ANN 基线配对；表 1 主表（Acc/F1/AUROC + 参数 + 能耗）；图 2 对齐 × ANN/SNN 交互；图 3 发放率被试间分布；消融小表。
4. Conclusion + limitation（transductive、单 seed、配方依赖）。

---

## 7. 与学长方案的关系

- 学长改 EEGNeX 的任何版本，注册成一个 model entry 就能进同一个 run，和这里所有 entry 配对比较；
- 本方案的所有实验 config / 提交脚本 / 汇总脚本对他的 entry 同样适用；
- 最终用哪一版由学长决定，本方案的结果至少提供：SNN 基线数字、对齐增益、能耗口径、统计脚本。
