---
title: "PROPOSAL — AdaHGNN（EEG_AdaMSHyper）面向跨被试运动想象的适配方案（待审阅，未实现、未跑）"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/hypergraph"
  - "#method/domain_generalization"
created: "2026-09-20"
updated: "2026-09-20 17:40"
status: "proposal-for-review"
---

# AdaHGNN → 运动想象跨被试：第一性原理适配方案

> **状态**：方案，**没有实现、没有提交任何作业**。AdaHGNN 相关作业 71803–71808 已于今天取消（全部 PENDING，Elapsed 0:00，从未训练）；此后未再提交。主实验 71034 / 71213 / 71222 未动。
> 原模型审计的每一条都标了来源：**[代码推导]** = 从 `project_V4/models.py` 逐行推出；**[待单测]** = 实现时用 CPU 单测确认。
> 本文没有任何测试集数字；所有"预期"都是设计估计，不是结果。

## 0. 核心结论（一页）

1. **原版 `EEG_AdaMSHyper` 是为 AVED 听注意任务设计的**（32 通道、128 Hz、1 s 窗、每被试同分布 LOSO + DANN）。搬到我们的运动想象（MI）数据（58/32 通道、250 Hz、4 s、跨被试、逐 trial z-score）上，**不改一字就有 4 处硬性不匹配**：显存（58 通道 × 1000 点 × 每通道 256 张特征图，batch 128 时激活约 37 GB，4090 D 24 GB 装不下）、时间尺度（卷积核 / 时移量按 128 Hz 硬编码）、空间池化把 **C3 与 C4 平均在一起**（正好抹掉左右手 MI 的侧化信号）、"自适应超图"在代码里**不可微**（结构 = 随机固定分组）。
2. **MI 的判别信息在哪里**：对侧感觉运动区（C3/C4 及邻近 FC/CP）mu（8–13 Hz）/ beta（14–30 Hz）节律的 **事件相关去同步（ERD）**，双脚在中线（Cz/FCz）。经典最优特征 = **先空间滤波（CSP / Laplacian），再取带功率**。原模型是"**每通道先取功率、再混合通道**"，数学上无法实现空间滤波（功率的线性组合 ≠ 线性组合的功率），这是它在 MI 上的**天花板**。
3. **方案**：分四层，逐层可关可开，最后由验证集（只用训练被试中划出的验证被试，**不看测试集**）决定采用哪一层：
   - **T0 必需**（否则跑不起来 / 明显违背物理量纲）：宽度 128→32、卷积核按 250 Hz 重标定、去掉/重标定时移模块、58 通道尺度表。
   - **T1 无争议适配**（不新增机制）：**解剖学 + 分半球** 的多尺度层级（58→13 区→4 超区；32→14→4），**保留节点身份的读出**（不再对区域取均值），dhg 稠密等价实现（数值一致、快 10 倍以上）。
   - **T2 核心增益**：(a) 让超图关联矩阵**真正可微、可学**；(b) **共享时间滤波器组**（跨通道共享，58× 参数减少）；(c) **把第一层超图消息传递搬到非线性之前**，变成"**学习到的超图 Laplacian 空间滤波器**" \(\tilde X=(I-\gamma A_1)X\)——这是 CSP/Laplacian 思想在超图框架里的原生实现，也是我预期收益最大的一项。
   - **T3 训练期可选**：原版的跨被试 Dirichlet mixup（改成**同类别**混合）、软标签 DANN、NHC 结构约束。本项目已有三条域对抗负证据，默认关，作为一个可选臂。
4. **推荐配置 AdaHGNN-MI**（T0+T1+T2）≈ **0.1 M 参数**（与 EEGNeX 58 k / ATCNet 115 k 同量级），激活约 7 GB，2C 一个 cell 估计 15–30 min。**同时保留"原版只做 T0"的 F+ 版本**作为"学长原模型"对照行。
5. **验证方案**：2C split 41 前 3 折，5 个阶梯变体（V0 原版-F+ → V4 全适配），按**验证集 macro-F1** 选定 → 冻结 → 三队列全量（31 cell/变体）。全程测试集不参与任何选择。
6. **需要拍板的 6 个问题**在 §8。方案通过后：实现 + 单测约半天，先导 3–4 GPU·h，全量约 10–20 GPU·h，**09-22 可出齐**。

## 1. 目标与约束

| 项 | 内容 |
|:---|:---|
| 目标 | 把学长的 `EEG_AdaMSHyper`（`/share/home/yuan/LY/DeepHypergraph/project_V4`，2026-01-16）适配到本项目三个跨被试 MI 队列，**在同一套 zs 协议下**与 18 个 ANN + 4 个外部基线 + SNN 逐被试配对比较，并把准确率提到该架构能达到的最高 |
| 不可改（协议） | `kfold_subject` 5 折 6:2:2、split 41+42（3C LOSO）；Adam 1e-4 / batch 128 / wd 0 / 无 scheduler / max 500 / patience 25 / 监控验证 macro-F1 / seed 0；输入 `[B, C, 1000]` @ 250 Hz，`per_sample_zscore`；**测试被试的任何数据不参与模型选择**；不做 EA / AdaBN（zs 臂） |
| 可改（模型） | 网络结构、结构超参、模型内部的训练期技巧（通过 `uses_custom_loss` / `training_step` hook，不改 trainer） |
| 数据规模 | 2C：训练被试 ≈ 31 人 ≈ 18 k trial / 折；SHU：≈ 15 人 ≈ 7 k；3C：≈ 8 人 ≈ 7 k（LOSO）。**小样本 + 被试偏移**：18 个 ANN 的结论是 58 k 参数的 EEGNeX 最好、67 M 参数最差，train–test gap 15–33 pp |
| 时间 | 截稿 09-24 20:00；今天 09-20 17:40 |

### 1.1 zs 是什么（回答之前的问题）

`zs` = `per_sample_zscore`：**每个 trial、每个通道**，用它自己 1000 个采样点的均值和标准差做 z-score，\(x \leftarrow (x-\mu_{t})/\sigma_{t}\)。fit-free、不跨 trial、不跨被试、不做 EA、不做 AdaBN。这是所有 18 个 ANN、4 个外部基线和 SNN 主实验共用的输入口径（`code/models/eeg_foundation/adapter.py::normalize_trials`）。`eaz` = EA + z-score，已按 D-035 取消。

## 2. 原模型审计：从第一性原理看它为什么不适合直接跑 MI

原版前向（`_official/models.py`，sha256 `46846808…`，与实验室文件逐字节一致）：

```
x[B,C,T] → LearnableTemporalShift(fs=128 硬编码, ±0.1 s)
        → 每个通道一个独立 TemporalStem: 1→256 逐点卷积 → BN → 拆两支各128 → depthwise conv k=63 / k=31 → BN → 求和 → GELU → AvgPool(8) → Dropout(0.5)
        → 堆成 [B,C,P,128] + 可学习时间位置编码 → 在 P 上取均值 → 节点特征 [B,C,128]
        → 三个尺度：{MultiScale 超图结构（嵌入相似度→top-k→阈值→0/1）→ HypergraphConvDHG(Linear→NHC 约束→HGNNPConv: Linear→mean 聚合→ReLU→Dropout) → 节点均值 → 连续下标均分池化到下一尺度}
        → concat 三个尺度均值 [B,384] → Linear128→BN→ReLU→Dropout→Linear(n_cls)  （另有 GRL + 领域分类头）
```

| # | 发现 | 依据 | 对 MI 的后果 |
|:--|:---|:---|:---|
| A1 | **显存**：每通道 stem 在 T=1000、B=128 下为反向保存 ≈ (256+256+2·128+2·128+128+128)=1280 张 `[128,1000]` fp32 图 ≈ 655 MB；58 通道 ≈ **37 GB**，32 通道 ≈ 20.5 GB。4090 D 只有 24 GB | [代码推导]（解析估计；AVED 原设定 T=128、B=60 只要 1/17） | 58 通道原版 batch 128 **必 OOM**；32 通道临界。要么 micro-batch（BN 统计量变了，D-034 明确不给 BN 模型用），要么减宽度 |
| A2 | **"自适应超图"不可微**：`adj = softmax(relu(3·sim))` → top-k 掩码 → `torch.where(adj>0.5,1,0)` → `nonzero` 取下标。阈值与取下标切断梯度；`embedhy` / `embednod` 全程收不到梯度 | [代码推导]（`models.py` L214–L228）；[待单测]：backward 后 `.grad` 为 None | 结构 = **随机初始化决定的固定分组**。而且嵌入 ~N(0,1)、维度 128 → sim 标准差 ≈ 11.3，×3 后 softmax 几乎 one-hot → **每个电极恰好属于 1 条超边**，58 节点 / 40 超边约剩 31 个非空组，多为 1–3 个电极的随机小组。"自适应"在这份代码里只是名字 |
| A3 | **连续下标池化平均了左右同名电极**：按我们的 58 通道顺序，58→29 的 29 对里 **16 对是左右同名对**：F1F2、F3F4、F5F6、F7F8、**C1C2、C3C4、C5C6**、T7T8、**CP1CP2、CP3CP4、CP5CP6**、TP7TP8、P3P4、P5P6、P7P8、O1O2。32 通道 32→8 的第 4 组 = {C3, C4, T3, T4}；32→2 = 前 4 组 vs 后 4 组（前后而非左右） | [代码推导]（`config.py::_generate_mappings` + `configs/datasets/{wbci_shu,shu}.yaml` 通道顺序） | **左右手 MI 的判别量就是 C3 vs C4 的 ERD 侧化**。尺度 2、3 在池化后完全没有侧化信息，只剩尺度 1 能判别。类名 `AnatomicalSpatialPooling` 在原任务里也不是解剖分区，只是均分 |
| A4 | **时间量纲按 128 Hz 硬编码**：核 63/31 点在 128 Hz 是 492/242 ms，在 250 Hz 变成 252/124 ms（124 ms 的滤波器频率分辨率 ≈ 8 Hz，分不开 mu 与 beta）；时移上限 `int(0.1×128)=12` 点 = 48 ms；patch 8 = 32 ms（原 62.5 ms） | [代码推导]（`models.py` L139、L142） | 滤波器组物理带宽变了；时移模块对 MI 无意义（AAD 里是对齐语音包络延迟，MI 没有刺激锁定的通道间延迟） |
| A5 | **每通道独立 stem**：58 × 12.3 k ≈ 715 k 参数（占总参数 1.87 M 的 38%），32 通道 394 k。时间滤波器在生理上是**通道无关**的（mu/beta 频带处处相同，差别只在幅度与空间分布） | [代码推导] | 58 倍冗余参数化；小样本跨被试下每通道滤波器容易学到被试特异模式。本项目证据：参数越大越差 |
| A6 | **空间滤波发生在非线性之后**：先对每通道做 GELU→时间平均（可证明 \(\mathbb E[\mathrm{GELU}(x)]=\sigma^2/\sqrt{2\pi(1+\sigma^2)}\) 对 \(x\sim\mathcal N(0,\sigma^2)\) 单调，即**每通道带功率**），再跨通道混合 | [代码推导] | 功率的线性组合 ≠ 线性组合的功率：**无法实现 CSP / Laplacian / 双极导联**这类先线性组合通道再取功率的空间滤波器。而这正是 MI 解码最强的先验（McFarland 1997：Laplacian > CAR > 耳参考；EEGNet/EEGNeX/ATCNet 全部是"时间卷积→depthwise 空间卷积→非线性→池化"）。这是原模型在 MI 上的**结构性天花板** |
| A7 | **读出对节点取均值**（置换不变）：三个尺度各取节点均值再 concat。通道身份只通过"每通道独立 stem 权重"隐含地进入 | [代码推导] | 一旦 stem 共享（A5 的修正），均值读出就**彻底丢掉左右信息**；读出必须保留节点身份 |
| A8 | **原训练法不可移植**：每通道 min-max 归一化、AdamW 1e-4 + wd 1e-3、`ReduceLROnPlateau` 监控**测试被试**准确率并按测试准确率选 checkpoint、batch 60、50 epoch、跨被试 Dirichlet mixup（p=0.5、k=3、α=2，**标签取源被试**，允许把左手 trial 和别人的右手 trial 混合后仍标左手）、软标签 DANN、NHC×0.1 | [代码推导]（`train.py`、`dataset.py`、`main.py`） | 我们的协议禁止测试集参与选择；共享配方固定。mixup/DANN 可作为模型内部 hook 复现，但 mixup 应改成**同类别**混合 |
| A9 | 实现效率：58 个 stem Python 循环；每个 forward 用 Python 循环把超图复制 B=128 次再建 `dhg.Hypergraph`（3 个尺度），并有 `.cpu().numpy()` 同步 | [代码推导] | 训练慢、GPU 利用率低；稠密等价实现（共享关联矩阵上的 \(D_v^{-1}HD_e^{-1}H^\top\)）数学相同 |

**结论**：A1/A4 是"跑不起来 / 量纲错"，A3/A7 是"把 MI 的判别量抹掉"，A2 是"卖点没起作用"，A5/A6 是"缺少 MI 最重要的归纳偏置"。只修 A1/A4 就跑，我预期它在 2C 上会明显低于 EEGNeX（83.4%）——这只是预期，需要 F+ 版本实测来证实。

## 3. 设计方案（分层，可逐层开关）

### T0 · 必需（不改机制，只让模型在我们的量纲下成立）

| 改动 | 内容 | 理由 |
|:--|:---|:---|
| T0.1 宽度 | `d_model` 128 → **32**（stem 中间通道 64；节点特征 32） | A1：激活 37 GB → ≈ 9 GB（原版结构）/ ≈ 7 GB（共享 stem 版）。MI 需要的带通滤波器数量 8–32 个（EEGNet 8、EEGNeX 8/32/64、ATCNet 16）；`d_model` 是原 Config 字段，属于配置而非结构改动 |
| T0.2 卷积核 | 63/31 → **125/63**（保持原设计 0.5 s / 0.25 s 的物理长度；250 Hz 下频率分辨率 2 Hz / 4 Hz） | A4；能分开 mu（8–13）与 beta（14–30） |
| T0.3 时移模块 | **去掉**（等价于 shift=0 初始状态；可选保留并把 `sampling_rate` 设为 250） | A4；MI 无通道间刺激延迟，去掉省一次 gather；保留则是无害的多余自由度 |
| T0.4 patch | 8 → **16**（64 ms，原设计 62.5 ms） | A4；读出为均值时几乎无影响，只影响 dropout 粒度和 T1.3 的时间注意力粒度 |
| T0.5 58 通道尺度表 | 原 Config 只支持 32/64；补 58 | 已在现有 adapter 中（`[58,29,1]`），T1.1 会换成解剖层级 |

### T1 · 无争议适配（修 A3 / A7 / A9，不新增学习机制）

**T1.1 解剖学、分半球的多尺度层级**（替换连续下标均分；同一个 `AnatomicalSpatialPooling` 类，只换映射表）

58 通道（2C / 3C），尺度 2 = **13 区**：

| 区 | 电极 | 区 | 电极 |
|:--|:--|:--|:--|
| PF 前额 (7) | Fpz Fp1 Fp2 AF3 AF4 AF7 AF8 | Mid 中线 (3) | **Fz FCz Cz**（双脚 MI 的 ERD 位置，3C 关键） |
| F-L (4) | F1 F3 F5 F7 | F-R (4) | F2 F4 F6 F8 |
| FC-L (4) | FC1 FC3 FC5 FT7 | FC-R (4) | FC2 FC4 FC6 FT8 |
| C-L (4) | **C1 C3 C5** T7 | C-R (4) | **C2 C4 C6** T8 |
| CP-L (4) | CP1 CP3 CP5 TP7 | CP-R (4) | CP2 CP4 CP6 TP8 |
| P-L (6) | P3 P5 P7 PO3 PO5 PO7 | P-R (6) | P4 P6 P8 PO4 PO6 PO8 |
| Occ (4) | POz Oz O1 O2 | | 合计 58 ✓ |

尺度 3 = **4 超区**：SM-L = FC-L+C-L+CP-L (12)、SM-R (12)、Mid (3)、Rest = PF+F-L+F-R+P-L+P-R+Occ (31)。

32 通道（SHU），尺度 2 = **14 区**：PF {FP1 FP2}、F-L {F3 F7}、F-R {F4 F8}、FC-L {FC1 FC5}、FC-R {FC2 FC6}、C-L {C3 T3}、C-R {C4 T4}、CP-L {CP1 CP5}、CP-R {CP2 CP6}、P-L {P3 T5 PO3}、P-R {P4 T6 PO4}、Mid {FZ CZ PZ}、Occ {OZ O1 O2}、Ear {A1 A2}（耳垂，几乎无脑电，单列以免污染别的区）。尺度 3 = SM-L (6)、SM-R (6)、Mid (3)、Rest (17)。

理由：A3。左右分开是 MI 的最低要求；中线单列服务 3C 双脚类；池化仍是区内均值（与原类一致）。

**T1.2 保留节点身份的读出**：\(z=[\,\mathrm{mean}_{n}(X^{(1)})\;;\;\mathrm{flatten}(X^{(2)})\;;\;\mathrm{flatten}(X^{(3)})\,]\)，维度 \(D+13D+4D=18D\)（58 通道）/ \(19D\)（32 通道），D=32 时 576 / 608 → Linear 128 → BN → ReLU → Dropout 0.5 → Linear(n_cls)。理由：A7；分类器要能算 SM-L 与 SM-R 的差。尺度 1（58 节点）仍取均值以控制参数（可选：注意力池化）。

**T1.3（可选，低风险）时间加权池化**：patch 均值 → 在 P=62 个 patch 上用 softmax 权重（62 个参数，初始化为均匀 = 原版均值）。理由：0–4 s 窗含 cue 后 0–0.5 s 的无 ERD 段，让模型自己学会降权；初始化等于原版，不会更差。

**T1.4 稠密等价实现**：共享关联矩阵 \(H\in\{0,1\}^{N\times E}\) 上一次 einsum \(A=D_v^{-1}HD_e^{-1}H^\top\)，代替"复制 B 份建 `dhg.Hypergraph`"。数学与 `dhg.nn.HGNNPConv`（默认 `W_e=1`，mean 聚合）完全相同；实现时写单测：同一 0/1 关联矩阵下与 dhg 输出的最大绝对误差 < 1e-5。理由：A9。

### T2 · 核心增益（新增 / 改变学习机制，预期收益最大，也是与原版差别最大的部分）

**T2.1 可微、可学的超图结构**
- 嵌入初始化改为 \(\mathcal N(0, 1/d)\)（相似度 O(1)，softmax 不饱和；原版 N(0,1) 在 d=128 下饱和为 one-hot，见 A2）。
- \(H_{\text{soft}}=\mathrm{softmax}_e(\alpha\cdot\mathrm{relu}(E_v E_e^\top))\)，\(\alpha=3\) 保留；top-k（k=5）掩码保留但用 **straight-through**：前向用掩码后重归一化的稀疏 \(H\)，梯度经 \(H_{\text{soft}}\) 回到嵌入。
- 聚合矩阵 \(A=D_v^{-1}HD_e^{-1}H^\top\) 用加权 \(H\)（HGNN 允许实值关联）。
- 理由：A2。让"自适应"名副其实：结构由任务梯度驱动（预期会学出 C3/FC3/CP3 一类的功能分组），而不是 seed 决定的随机分组。NHC 约束（原版 ×0.1）在结构可微后才有意义，作为 T3.3 可选。

**T2.2 共享时间滤波器组**（跨通道共享权重）
- 原版每通道独立 stem → 改为对所有通道相同的滤波器：`Conv2d(1, F, (1,125))` + `Conv2d(1, F, (1,63))`（两支相加，与原 radix=2 两支求和一致）→ BN(F)。F = 32。参数 ≈ 6 k（原 715 k）。
- 通道特异性由三处保留：每节点可学习仿射（C×F，≈ 3.7 k）、超图关联（每节点自己的分组权重）、T1.2 的按节点读出。
- 理由：A5。生理上滤波器通道无关；跨被试小样本下 58× 参数减少直接降方差。这是 EEGNet/EEGNeX/ATCNet 共同的选择。

**T2.3 信号域的超图 Laplacian 空间滤波（关键）**
- 把**第一层超图消息传递搬到非线性之前**，作用在带通后的时间序列 \(S\in\mathbb R^{B\times F\times C\times T}\) 上：
  \[\tilde S = (I-\gamma\,A_1)\,S,\qquad \gamma=\mathrm{sigmoid}(g)\in[0,1],\ g\in\mathbb R^{C}\ (\text{每节点一个，初始 } \gamma=0.5)\]
  \(\gamma=0\) 退化为原始通道；\(\gamma=1\) 是"通道减去它所在超边邻域的均值" = **小 Laplacian 参考**（McFarland 1997 证明其 mu 节律信噪比优于 CAR 与耳参考）。邻域由 T2.1 学到的超图决定 → **"自适应超图学的是 Laplacian 参考邻域"**，这是超图在 MI 里最自然、最可解释的角色。
- 然后才是功率：GELU（原版；或平方，做消融）→ AvgPool(16) → T1.3 时间加权 → 每节点仿射 → 节点特征 \([B,C,F]\)。
- 之后按原版走三个尺度的 HGNNP 卷积（Linear→聚合→ReLU→Dropout）+ 解剖池化 + T1.2 读出。
- 理由：A6。这是让该架构能够表达"先空间滤波再取功率"的**唯一**改动（其余层都在功率域）。它只增加 C 个标量参数。

### T3 · 训练期可选臂（模型内部 hook，不改 trainer / 协议）

| 项 | 内容 | 依据与态度 |
|:--|:---|:---|
| T3.1 同类别跨被试 mixup | 复现原版 Dirichlet 混合（p=0.5、k=3、α=2），但**只在同一 batch 内、同标签、不同被试**的 trial 之间混合 | 原版把不同类别混合后仍用源标签（A8）——对 MI 是标签噪声；同类别版本是无损的数据增广。本项目对 mixup 无证据，值得一个臂 |
| T3.2 软标签 DANN | GRL + 领域头，alpha 按原版 sigmoid 日程（以前 50 epoch 计），领域 = 训练被试 | 本项目三条负证据（S4ERP-Adv +0.28、ATCNet-Adv −4.22、λ 扫描无差别）。**默认关**；若学长坚持原版思想可开一个臂。实现约束：`cross_subject_protocols.py:670` 只在模型名含 `adv` 时才把被试 id 放进 batch，所以该臂别名须形如 `adahgnn@mi_adv` |
| T3.3 NHC 结构约束 ×0.1 | 原版的超边间距离/相似度 + 节点重构 L1 | 仅在 T2.1 可微后有意义；作为 T2.1 的开关一起消融 |

### 推荐配置 AdaHGNN-MI（T0 + T1 + T2，T3 关）前向定义

```
x[B,C,1000] (zs)
 → S = BN( Conv(1,125)(x) + Conv(1,63)(x) )               # 共享滤波器组, F=32          [B,F,C,T]
 → S~ = (I − γ ⊙ A1) S   (A1 来自可微超图 H1, γ∈[0,1]^C)  # 学习到的 Laplacian 空间滤波  [B,F,C,T]
 → GELU → AvgPool(16) → Dropout(0.5) → softmax-时间加权求和 → 每节点仿射     # 带功率     [B,C,F]
 → 尺度1: HGNNP(H1): Linear→A1聚合→ReLU→Dropout           X1 [B,58,F]  → mean_n → D
 → 解剖池化 58→13 → 尺度2: HGNNP(H2)                        X2 [B,13,F]  → flatten → 13D
 → 解剖池化 13→4  → 尺度3: HGNNP(H3)                        X3 [B,4,F]   → flatten → 4D
 → concat (18D=576) → Linear 128 → BN → ReLU → Dropout 0.5 → Linear n_cls
```

预算（D=F=32，58 通道）：滤波器组 ≈ 6 k、超图嵌入 ≈ 4 k、三层 HGNNP ≈ 6 k、仿射 ≈ 4 k、分类器 ≈ 74 k → **≈ 0.1 M 参数**；反向保存的激活 ≈ 7 × 0.95 GB ≈ **7 GB**（B=128）；每 epoch 计算量 ≈ 数百 GFLOP，2C 估 10–20 s/epoch → 一个 cell 15–30 min（数据加载为主）。F=64 版 ≈ 0.19 M 参数、13 GB，作消融。

F+（原版，只做 T0）：官方类不改代码，`d_model=32`、连续下标池化、无时移（或 `sampling_rate=250`）；参数 ≈ 0.2 M（58 个 stem × 3.3 k）；激活 ≈ 9 GB；因 58 个 Python 循环 stem + 每步建 dhg 超图，速度慢数倍，一个 cell 估 1–3 h（不确定，冒烟后修正）。

## 4. 验证方案（只用验证被试，测试集不参与）

| 步骤 | 内容 |
|:--|:---|
| 先导队列 | **2C split 41，折 0/1/2**（每折 10 个验证被试 → 30 个验证被试；2C 是唯一能分辨 ~1 pp 的队列，SHU 是地板效应） |
| 阶梯变体 | V0 = F+（原版+T0）；V1 = V0+T1；V2 = V1+T2.1；V3 = V2+T2.2；V4 = V3+T2.3（= AdaHGNN-MI）；可选 V5 = V4+T3.1（mixup），V6 = V4+T3.2（DANN） |
| 选择规则（预先声明） | 以**验证集 macro-F1（3 折均值）**为唯一依据；一个层级被采用需满足 ≥ +0.5 pp，或它是量纲/物理修正（T0、T1.1、T1.2）且不劣于 −0.5 pp。**不看测试指标**（`curves: false`，测试曲线不打印） |
| 冻结 | 选定后写进 `code/configs/models/adahgnn.yaml` 的 `variants`，记 D-xxx，再跑三队列全量：2C 10 + SHU 10 + 3C 11 = 31 cell / 变体；建议 **V4 与 V0 都跑全量**（V0 = "原模型"行，V4 = "适配"行） |
| 统计 | 探索性条目，不进预注册五族；在同一 run 目录内与全部条目逐被试配对（`analyze_snn_round.py` 自动纳入）；Bonferroni–Dunn 的 k 相应 +1/+2 |
| 单测（实现时） | (i) `_official` 文件 sha256 不变；(ii) 稠密聚合 vs `dhg.HGNNPConv` 在 0/1 关联下误差 < 1e-5；(iii) 超图嵌入**有**梯度（对照原版无梯度）；(iv) 解剖映射覆盖每个电极恰好一次、左右不混；(v) 三队列 forward/backward 形状与 JSON `describe()`；(vi) 协议 yaml 与 `snn_*_zs_lr1e4.yaml` 逐键相同 |

## 5. 预期与风险（是估计，不是结果）

| 项 | 预期 | 风险 / 应对 |
|:--|:---|:---|
| V0（原版 F+） | 2C 明显低于 EEGNeX；SHU/3C 接近底部 | 若冒烟显示 1 个 cell > 3 h，全量只跑 2C 或减 `d_model` 到 16 |
| V1（解剖层级 + 节点读出） | 修 A3/A7 应有可见提升 | 参数从均值读出变 flatten，分类器变大；由 dropout 0.5 + BN 控制 |
| V2（可微结构） | 小到中等；主要是"卖点成立"而非精度 | 若不稳定，回退为固定解剖初始化的结构（仍可微） |
| V3（共享 stem） | 降方差，跨被试通常有利 | 若验证集变差，改为"按 13 区共享"的折中 |
| V4（Laplacian 空间滤波） | **最大的一项**；把模型带进 EEGNet/EEGNeX 所在的"先空间滤波再功率"类 | γ 全学到 0 则说明无用（可检查）；正权重聚合只能做区域参考，不能做有符号 CSP——若还想更强，下一步是允许 Θ 在通道维混合（那就接近 depthwise 空间卷积，背离超图身份，本轮不做） |
| 命名与归属 | V4 与原版差别实质性；论文里需写成 "AdaHGNN-MI（adapted from EEG-AdaMSHyper）" 并说明改动 | 由学长决定用哪一行、怎么署名 |
| 时间 | 实现 + 单测 0.5 天；先导 15 cell ≈ 4–8 GPU·h；全量 31–62 cell ≈ 10–25 GPU·h | 71034/71213/71222 今晚收尾后 4 卡可用；09-22 出齐可行 |

## 6. 与前序工作的关系

- D-034 外部基线协议原样沿用；AdaHGNN 的 cell 仍写进 `snn_*_zs_lr1e4_v1`，与 18 ANN + 4 外部 + SNN 同折配对。
- D-037（导出）→ D-038（撤回改写、恢复官方类、装 `dhg`、取消 71803–71808）→ 本方案 D-039（用户改指令：先出适配方案，不跑）。
- 本项目已证伪的方向（域对抗、融合 A/B）在 T3 中只作可选臂；主线不依赖它们。
- 与 `PLAN_spiking_eegnex_icassp2027.md` 无冲突：AdaHGNN 只是对照表新增行，不改主实验。

## 7. 时间表（方案通过后）

| 时间 | 事项 |
|:--|:---|
| 09-20 晚 | 学长/用户审阅本方案；拍板 §8 |
| 09-21 上午 | 实现 `code/models/external/adahgnn/mi_adapter.py`（V0–V4 作为 `adahgnn@{orig,v1,…,mi}` 变体，官方文件不动）+ 6 项单测 + CPU 微型前向；`make_adahgnn_configs.py` 生成先导 yaml（`--folds-subset 0,1,2 --split-seeds 41` 走命令行，不新建 run_id） |
| 09-21 中午 | 先导提交（每变体一进程，2 卡并行，`SMOKE_FIRST`）；下午看验证 macro-F1 |
| 09-21 晚 | 冻结变体（D-xxx）→ 三队列全量提交（V4 + V0） |
| 09-22 | 出齐 → `analyze_snn_round.py` → `build_dataset_tables.py` 加行 → 报告 |

## 8. 需要拍板的问题

1. **接受 T2 的三项改动吗？**（可微结构 / 共享滤波器 / Laplacian 空间滤波）——这是精度的主要来源，也是与原版差别最大的部分。若只允许 T0+T1，我预期精度上限有限（A6）。
2. **原版行（V0 = F+）要不要跑全量？** 我建议跑（论文/汇报里"原模型 vs 适配"一目了然），代价约 10–20 GPU·h，取决于冒烟速度。
3. **T3 要不要开臂？** 我建议 mixup（同类别）开 1 个臂、DANN 不开（三条负证据）；若学长坚持 DANN 思想，可开 `@mi_adv`。
4. **先导队列**：2C 前 3 折（我建议）还是 SHU 全 5 折（更快、但地板效应分辨不出差别）？
5. **模型命名**：AdaHGNN-MI / EEG-AdaMSHyper-MI / 学长定名。
6. **`d_model`/F**：32（默认，省显存）还是 64（消融）？

## 9. 文件清单

- 本方案：`docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`
- 官方原文件（未改）：`code/models/external/adahgnn/_official/{models.py,config.py}`（sha256 见 `PROVENANCE.md`）
- 现有忠实包装（官方类 + zs 协议，未跑）：`code/models/external/adahgnn/adapter.py`、`code/configs/models/adahgnn.yaml`、`code/configs/experiments/bl_*_adahgnn_zs_lr1e4.yaml`、`tests/models/test_adahgnn.py`
- 已取消作业：71803–71808（`logs/slurm/bl_adahgnn_job_ids.txt`，`sacct` 全部 CANCELLED、Elapsed 0:00）
- 相关决定：`docs/DECISIONS.md` D-037 / D-038 / D-039
- 原始来源：`/share/home/yuan/LY/DeepHypergraph/project_V4/{models.py,config.py,train.py,dataset.py,main.py}`
