---
title: "AGENTS.md — ICASSP 2027 项目灵魂记忆"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/snn"
  - "#method/domain_generalization"
created: "2026-09-18"
updated: "2026-09-20 20:30"
status: "active"
---

# AGENTS.md — 唯一权威灵魂记忆（先读这个）

> 任何 Agent / 人进入本目录先读本文件（5 分钟），再按 §0 的顺序读状态文件。
> 本文件只放**长期有效**的身份、已验证事实、规则与契约。「现在做到哪、什么在跑」在 `docs/STATUS.md`；
> 逐日流水在 `progress.md`；每个影响口径的决定在 `docs/DECISIONS.md`。**结果数字不进本文件**，
> 只给出它们在哪个表里。`CLAUDE.md` 与 `.cursor/rules/` 只是入口和机器可读切片，都指向这里。

## 0. 文件路由（进门先看）

| 你想知道 | 看 |
|:---|:---|
| 现在做到哪、哪些作业在跑、下一步、截稿倒计时 | **`docs/STATUS.md`** |
| 逐日发生了什么（最新在上） | `progress.md` |
| ICASSP 论文方案（v2，四位专家评审后；§0.5 = 先导结果与主实验决定） | **`docs/PLAN_spiking_eegnex_icassp2027.md`** |
| 之前跑过什么、哪个模型最强、哪些方向已证伪（3 队列 × 18 ANN） | **`docs/PRIOR_RESULTS.md`** |
| WBCIC-SHU 3C 的权威数字与勘误 | `docs/RESULTS_TRUTH.md` |
| 18 个 ANN 基线的协议与 27 条锁定决定（SNN 轮完全沿用） | `docs/PLAN_cross_subject_benchmark_and_fusion.md` |
| ATCNet 消融 / 域对抗为什么结案 | `docs/ANALYSIS_atcnet_and_adversarial.md` |
| 每个影响口径的决定（编号 D-xxx） | `docs/DECISIONS.md` |
| 目录与代码分层、哪些代码是活的 | `docs/ARCHITECTURE.md` |
| 投稿格式、截稿、ORCID、模板、检查单 | `docs/ICASSP2027_SUBMISSION.md`；**稿件底稿 = 根目录 `old_reference.tex`**（团队 2026 论文改的，D-033），样式文件与 `refs.bib` 在 `paper/` |
| 复现命令（bench 与 SNN 轮） | `REPRODUCE.md`；换机 `SETUP.md` |
| **换到 3090 / 新机器（给那边的 AI 执行）** | **根目录 `HANDOFF.md`** → `docs/handoff/` |
| 发给学长的两份文档（09-18） | `docs/reports/` |
| 跨被试协议的文献依据（6:2:2、seeds 41–45 的出处等） | `docs/CROSS_SUBJECT_PROTOCOL_MEMO.md`（7 个问题）→ `docs/CROSS_SUBJECT_PROTOCOL_RESEARCH.md`（全文，带置信度标签） |

## 1. 项目身份

| 维度 | 内容 |
|:---|:---|
| 目标 | **一篇 ICASSP 2027 论文**（4 页正文 + 可选第 5 页只放参考文献）。截稿 **2026-09-23 23:59 AoE = 北京时间 09-24 20:00**（官网 09-18 核对已从 09-16 延期到 09-23；此前记录称"不再延期"，未再核实；CMS Paper Kit 页面截至 09-18 仍写 09-16——投稿前登录 CMS 确认）。ICASSP 2027：2027-05-16~21，Toronto |
| 题目方向（**候选，待学长拍板**） | 方案 v2 提议：**Spiking EEGNeX for cross-subject motor-imagery EEG**——把 EEGNeX 逐层脉冲化（时间原生编码：EEG 采样轴 = 脉冲时间轴），配无标签被试级对齐（EA + AdaBN），在 87 人 / 3 个队列上与 18 个强 ANN **逐被试配对**比较；建议的框架是「非劣性 + 对齐交互 + 能耗」（因为 Zhang et al., *Neural Networks* 2025 已发表 PLIF 跑在 1000 采样点 + EA，不能 claim 首创）。**选哪些实验、怎么立论、写不写这套框架，由学长决定；学长定之前 Agent 不写论文正文。** |
| 学长指令（2026-09-17） | 「那个 eegnex 我看看能不能改一下 / 加点别的 / 再套到 snn 上」；「我们跑了几个数据集、对应跑了几个模型，梳理完开会」；「在 SYX 下新建 ICASSP，我们到这个里面做」。学长会自己改一版 EEGNeX；本仓库的 S-EEGNeX 是**并行版本**，**最终采用哪版由学长定**。学长的模型注册成一个 model entry 即进入同一张表、与所有条目逐被试配对 |
| 数据集（3 个独立跨被试队列） | WBCIC-SHU 2C（51 人，58ch，左/右手）· SHU 2022（25 人，32ch）· WBCIC-SHU 3C（11 人，58ch，左/右/脚，LOSO）。2C 与 3C 是**互斥被试队列**（1–51 vs 52–62）。全部 250 Hz、0–4 s、1000 点 |
| 环境 | conda `mi_torch_cu118`（Python 3.10，torch 2.7.1+cu118）；Slurm 分区 `gpu2node` / `gpu3node`，QOS 每用户 4 GPU / 48 h 一作业；登录节点不跑重活 |
| 仓库 | 本目录 = git 仓库根。历史远端 `origin` = `git@github.com-stevencummings:stevencummings341-prog/EEG.git`（旧名自动重定向）。**可迁移远端** `icassp` = `https://github.com/yixiangsunLeo/ICASSP`（private；3090 克隆这个；`result.json` / 分析表经它同步）。未经用户授权不得 commit / push |
| 血缘 | 本仓库就是 2026-06 起的 `eeg-mi-online`（跨 session → 跨被试 → benchmark → SNN），2026-09-18 迁到本路径并剪掉跨 session 时代的遗留（去向见 §10）。姊妹项目 `../Benchmark`（EEG 基础模型 benchmark，ICASSP 之后的主线） |

## 2. 必须知道的已验证事实（浓缩版；数字与表在 `docs/PRIOR_RESULTS.md`、`docs/PLAN_spiking_eegnex_icassp2027.md` §0.5）

1. **ANN 基线已齐**：18 个模型条目在 2C 与 SHU 上各 180 cell 全部完成（`results_snapshot_2026-09-13/experiments/{wbci_shu/wbci_2c_cross_subject_bench_v1, shu/shu_cross_subject_bench_v1}`），3C 上 7 模型 × 2 套配方（`docs/RESULTS_TRUTH.md`）。**EEGNeX 在三个队列都是第一或并列第一**（58 k 参数，GPU 时间是 ATCNet 的 1/4）。
2. **瓶颈是被试偏移不是容量**：2C 前 8 名相差 1.0pp、被试间 SD ≈ 12pp；67 M 参数模型垫底；所有模型 train–test gap 15–33pp；3C sub-011 被试内也只有 34.7%（chance 33.3%）= BCI illiteracy。→ 模型比较必须**逐被试配对**。
3. **已结案、不要再提的方向**：ATCNet×S4ERP 融合 A/B（显著更差）、ATCNet 三个消融（不显著）、域对抗（三条负证据）。
4. **SNN 先导（SHU，split 41，120 cell，2026-09-18 完成）**：lr **1e-4** 优于 1e-3（= bench 配方）；对齐臂选 **EA + 逐通道 z-score（`eaz`）**（EA+标量 = 0，8–30 Hz 参考协方差 −1.3pp，EEG 专家的理论被数据否定）；D 版（T=4）比 EEGNeX −0.48pp（n.s.），S 版（时间原生）−1.38pp（p=0.04）；S 版能耗比 EEGNeX 低 **17×**、D 版 7×；编码器位置消融：模拟前端（`afe`）**更差** −1.82pp → 1-bit 编码放在 conv1 后成立；H1「对齐更帮 SNN」在 SHU 上**无支持**（交互 −0.5pp，p=0.35），由 2C 判。
5. **旧 checkpoint / 逐 trial 预测已删**（2026-09-13 清理 41 G + 11 G），所以所有 ANN 参照（`eegnex`、`eegnet_official`）在 SNN 轮里**同折重跑**，不依赖旧权重。
6. **SHU 2022 是地板效应**（chance band 51.4–53.7%，67/125 session 类别不平衡 → 主指标 balanced accuracy）；**2C 是唯一能检出 ~1–2pp 差异的确认队列**；3C 只有 11 人，LOSO。
7. 所有 SNN 轮 config 的 `cross_subject` 块与 bench **逐字相同**（`kfold_subject` / 5 折 / 6:2:2 / `split_seeds [41, 42]` / `train.seeds [0]`），因此每个新条目都与 18 个 ANN 在相同 (split, fold, subject) 上配对。对比模型在 2C / SHU 另跑 seed 43（D-036）；3C 是 LOSO，不加第三个划分。

## 3. 硬边界

1. 只在本仓库根目录内写文件；外部 raw 目录（`/share/workspace2/moto_imagination/{WBCIC_SHU,SHU}`）**只读**，禁止改名/删除/写入。
2. 外部路径只出现在 `code/configs/paths.local.yaml`（gitignore；本集群只有这一份 local 文件——`code/configs/datasets/*.yaml` 是 `/CHANGE/ME` 占位，实验 YAML 用 `paths.local.yaml` 的逻辑键 `processed_manifest` / `shu_processed_manifest` / `wbci_3c_processed_manifest`）；Python / 实验 YAML 里不写绝对路径。
3. **不编造结果**；未落盘（`result.json` 不存在）的东西只能写 planned / running，不能写 done。
4. 不覆盖已完成的 `*_v1` 输出与 checkpoint；换配置必须换 `--out` / `run_id`（`cell_signature` 守卫会报错而不是静默续跑）。
5. **不 `scancel` 正在跑或排队的作业**、不改动 `code/`、`scripts/slurm/`、`code/configs/experiments/snn_*.yaml`、`code/configs/paths.local.yaml`，除非用户明确要求——主实验作业（zs 臂：71021–71023、71030–71031、71034–71035、71230–71234；eaz 臂已按用户决定于 09-19 01:50 全部取消，D-035）与外部基线作业（71184–71185、71206–71225）正在依赖它们（清单 `logs/slurm/{snn_*,bl}_job_ids.txt`）。
6. 不 commit / push，除非用户明确要求。例外：3090 机器按 `HANDOFF.md` Phase 12 用 `scripts/sync_results_git.sh` 提交 `result.json` 与分析表（禁止二进制、禁止 `paths.local.yaml`、禁止 force-push）。剪遗留已在 D-041 落地；仍不要 `git clean -fd` / `git push --force`。新克隆以 `yixiangsunLeo/ICASSP` 为准，不要再 clone 旧 `EEG.git` 当工作树。
7. 登录节点不跑训练 / 预处理；分析脚本（`scripts/analyze_snn_round.py`、`scripts/summarize_cross_subject.py`）是 CPU 轻活，可在登录节点跑。
8. 兼容软链接 `/share/home/yuan/SYX/eeg-mi-online -> ICASSP` 在主实验作业全部结束前**不得删除**（20 个作业的 `WorkDir` / `Command` / `StdOut` 写的是旧路径；3C manifest 已改为新路径，不再依赖软链接）。

## 4. 实验规则

1. 按**被试**划分，绝不按 trial 泄漏；验证集只从训练被试里切；测试被试的数据**不参与任何模型选择**（`e2e_trainer.py` 有守卫；`curves: false` 时连测试曲线都不打印）。
2. 归一化 / 对齐只能是 per-trial fit-free 或**只用该 session 自身无标签 trial**（EA 逐 session 白化、AdaBN 用测试被试无标签 trial 重估 BN 统计量）；**无统计量跨被试**。EA / AdaBN 是 transductive-unsupervised，论文必须写明，并且 ANN 参照跑**同样的对齐臂**。
3. **预注册家族（2C，Holm，α=0.05，5 项）**：H0-NI（S@hplif vs eegnex，未对齐，δ=1pp，单侧 bootstrap 下界）；~~**H1 = 唯一主检验**~~（差之差 Wilcoxon：S-EEGNeX-S@hplif，EA+AdaBN vs 无，accuracy，best-val，2C）——**本次投稿不跑 eaz 臂，H1 无数据、不写（D-035，用户 09-19 决定；论文 = zs 协议下的对比表 + 能耗）**；H2-NI（S vs D，δ=1pp）；H3（Page's L：LIF ≤ PLIF ≤ HetPLIF）；B1（S-EEGNeX vs S-EEGNet）。SHU / 3C 为复现队列（同 5 项，队列内 Holm）；其余全部标探索性。**不按测试集改主模型**（先导中二值读出 +0.5pp n.s.，仍保持预注册的 LI 读出）。
4. **两块主表，绝不混比**：未对齐块（18 ANN + 未对齐 SNN，Friedman + Nemenyi CD，Bonferroni–Dunn 以 S-EEGNeX 为对照 k=19）；对齐块（{eegnex, eegnet_official, S-EEGNeX} × {无, EA, EA+AdaBN} = 3×3；H1 用其中 {EEGNeX, S-EEGNeX} × {无, EA+AdaBN} 的 2×2 子集做差之差）。**不把对齐后的 SNN 与未对齐的旧 ANN 直接比。**
5. 检验实现：n ≤ 50 精确 Wilcoxon、n = 51 置换；Hodges–Lehmann 中位差 + BCa bootstrap CI；报 n_eff、胜/平/负；覆盖断言 `n_cells == n_folds × n_splits`；跨模型比较只在同一 `run_id`（同划分）内。
6. 配方 = bench 配方：Adam 1e-4 / batch 128 / 无 scheduler / max 500 / patience 25 / monitor macro-F1 / 单训练 seed 0 / 划分 seed 41+42。**对比模型**在 2C / SHU 上另加划分 seed 43（D-036）；**3C 保持 LOSO（split_seed 0），不加 43**。SNN 本轮不加 43。单训练 seed 写进 limitation（预算允许给 2C 主 2×2 补 2 个 seed）。
7. 能耗口径：逐层 dense MACs × 实测发放率（45 nm：MAC 4.6 pJ / AC 0.9 pJ），以 **EEGNet ANN 与 S-EEGNet 为参照**（「EEGNeX 的精度、~2× EEGNet 的能耗、比 EEGNeX 低 17×」），乘数**不进标题和摘要**。
8. 每个 cell 记录 seed、split、模型 YAML、配方、git commit、环境指纹、硬件；每 cell 只存 `best.pt` + `last.pt`，可断点续跑，`result.json` 为完成标记。
9. 先 smoke 再 full；同一条提交命令重复执行 = 断点续跑（完成的 cell 跳过），永远安全。

## 5. 目录路由

| 路径 | 放什么 | 规则 |
|:---|:---|:---|
| `code/` | 训练框架（唯一人工入口 `python code/run.py --config ...`）；`code/models/spiking/` = 本轮新模型；`code/configs/experiments/snn_*.yaml` = 本轮 24 个网格配置（主实验用 6 个）。2026-09-18 已按主实验命令的 import 闭包剪枝（D-031），剩下的文件都在闭包内或被保留脚本/测试引用 | 分层见 `docs/ARCHITECTURE.md`；**主实验期间不动**（闭包内文件在作业启动时才被读取） |
| `scripts/` | `make_snn_configs.py`（生成 snn 配置）、`analyze_snn_round.py`（预注册检验 + H1 + 能耗 + 与 bench 配对；臂映射 `--aligned-arm`）、`summarize_cross_subject.py`、`plot_three_curves.py`、预处理入口；`scripts/slurm/{_common.sh, bench_gpu.sbatch, submit_snn.sh, submit_bench.sh, shu_gpu.sbatch, shu_cpu.sbatch}` | 能进 `code/` 的功能不留在 scripts；主实验期间**允许**在 `scripts/` 根下新增分析/画图脚本（不在作业闭包内），`scripts/slurm/` 不动 |
| `tests/` | pytest（剪枝后 106 项 + 1 skip；`tests/models/test_spiking.py` 13 项） | 新模型 / 新协议必带测试 |
| `outputs/experiments/{shu,wbci_shu}/snn_*_v1/` | 本轮 run 产物（`cells/**/result.json`、`failures__*.json`、`resolved_model_params__*.json`） | 只读；复跑用新 run_id |
| `outputs/analysis/snn_round/<dataset>/` | `analyze_snn_round.py` 的表（`REPORT.md`, `per_model.csv`, `paired_tests.csv`, `interaction_H1.csv`, `energy.csv`, `subject_matrix.csv`） | 表由脚本生成，禁止手改 |
| `outputs/processed/wbci_shu_3c_mat_clean/` | 3C processed 数据（2.0 G，作业 70221 重生成） | 3C 作业依赖；不删 |
| `checkpoints/{shu,wbci_shu}/<run_id>/<cell>/{best,last}.pt` | 权重 | 不删；不重复复制 |
| `logs/slurm/` | `<job>-<id>.out/.err`、`sacct_*.txt`、`snn_<tag>_job_ids.txt` | 作业清单在这里 |
| `results_snapshot_2026-09-13/` | **只读归档**：18 ANN bench。git 只跟踪 `README` + `tables/` + `cells/*/result.json`（给 `--bench`）；predictions / 曲线仍在盘上 | 不改 cells 内容 |
| `HANDOFF.md`、`docs/handoff/` | 给 3090 / 新机器 AI 的执行手册 | 换机步骤变了才改 |
| `docs/` | 方案、状态、决定、先前结果、架构、投稿说明、`reports/`（给学长）、`history/`（会话摘录） | 见 §8 |
| `paper/` | ICASSP 2027 官方模板（`spconf.sty`, `IEEEbib.bst`）、`refs.bib`、`figures/`、`tables/`、早期骨架 `main.tex`；**稿件底稿本身是根目录 `old_reference.tex`**（用户放置，D-033），其三张队列表由 `scripts/build_dataset_tables.py --splice` 生成 | 数字只能来自 `outputs/analysis/**/*.csv`；见 §7 |

## 6. 代码规则

- 人工入口只有 `python code/run.py --config code/configs/experiments/<name>.yaml [--models a,b@variant --split-seeds 41 --tag-suffix X --device cuda]`；GPU 走 `scripts/slurm/submit_snn.sh <config> <tag> "<group1>" ["<group2>"...] [-- extra]`（每组一卡，`N_LINKS` 个 48 h 接力）。
- 模型别名 `name@variant`（`code/models/model_specs.py`：`defaults` / `per_dataset` / `variants` 单一来源）；未定义别名一律抛错。本轮条目：`eegnex`, `eegnet_official`, `seegnex@{hplif, s_lif, s_plif, s_hplif, s_hplif_afe, s_hplif_spk, s_hplif_noinit, s_hplif_causal, s_hplif_hard, s_hplif_noreg, s_hplif_all}`, `seegnet@plif`；**外部基线轮（D-034）**：`tcformer`, `labram`, `eegpt`, `cbramod`（`code/models/external/`，registry 懒加载；`per_dataset` 只放 `channel_set`；权重在 `weights/`，协议 `docs/PROTOCOL_external_baselines.md`），3C 另补 `atcnet`, `eeg_deformer`, `fbcnet`；**AdaHGNN（D-037 / D-038 / D-040）**：`adahgnn`（未改的 `EEG_AdaMSHyper`，注册但不跑）、**`adahgnn@mi` / `adahgnn@mi_mix`（AdaHGNN-MI，学长模型的 MI 适配，`code/models/external/adahgnn/mi_model.py`；无预训练权重；协议同 D-034 zs；论文行按验证集 macro-F1 选臂）**。
- 模型特有训练行为走 hook（`uses_custom_loss` / `training_step` / `after_optimizer_step`），不 fork trainer。`code/training/trainer.py` **冻结**；新能力进 `e2e_trainer.py`。所有模型共享 `{logits, features, confidence}` 契约；SNN 额外 `describe()`（参数 / τ / 能耗）与 `pop_spike_stats()`。
- 加模型 → `code/models/<name>/` + `code/configs/models/<name>.yaml` + `registry.py` 注册 + 测试；加实验 → `scripts/make_snn_configs.py` 生成（不要手写那 24 份 YAML）。
- 弃用模型（`s4erp_adv`, `atcnet_adv`, `dualcd_s4_pos`, `eegnet`(自研), `cap_eegnet`）目前仍在 `registry.py` 的模块级 import 里（= 作业 import 闭包内），**作业结束后**连同 yaml 与相关测试一起删除；它们产出的数字都保存在 `results_snapshot_2026-09-13/` 的 `result.json` + `resolved_config.yaml` 中，不依赖代码存活。

## 7. 论文规则

0. **先决条件：论文题目、框架、进论文的实验清单由学长决定。** 学长拍板前 `paper/main.tex` 只维护格式骨架与空表格，Agent 不写摘要 / 正文 / 贡献，不替学长挑实验。下面 1–5 条是拍板之后写作时的约束。
1. 稿件底稿是根目录 `old_reference.tex`（团队 ICASSP 2026 论文按 2027 kit 对齐；`paper/main.tex` 为早期骨架），用官方 `spconf.sty`（`paper/README.md` 有编译说明）。**4 页正文（含图、可含参考文献）+ 可选第 5 页只放参考文献 / 资助 / 伦理合规声明**；PDF ≤ 5 MB；文件名 = 第一作者姓 `.pdf`；单盲（publish-ready，署名）；**每位作者需 ORCID**；IEEE 要求**在致谢中披露 AI 生成内容的使用**（编辑 / 语法润色除外）。
2. 每个数字都要能指回 `outputs/analysis/snn_round/<dataset>/*.csv` 或 `results_snapshot_2026-09-13/**/tables/*.csv` 的一行；表格由脚本生成后再排版，不手算。
3. 框架以学长拍板为准（方案 v2 的建议是「非劣性 + 对齐交互 + 能耗」，仅供参考）；无论哪种框架，必须引用并明确对比：Zhang et al. 2025 (*Neural Networks*, Wu 组, PLIF+EA)、NiSNN-A (TNNLS 2025)、HR-SNN (TCDS 2024)、LENet / SRIF（转换法）、Yin–Corradi–Bohté 2021、Perez-Nieves 2021（异质 τ）、PLIF (Fang, ICCV 2021)、He & Wu 2020 (EA)、Junqueira 2024、Wimpff 2024、Horowitz 2014 + Dampfhoffer 2023（能耗口径）、EEGNeX (Chen 2024)、EEGNet (Lawhern 2018)、ATCNet、EEG-Deformer、数据集论文（Yang 2025 *Sci Data*；Ma 2022；DSGNet JBHI 2026 用作 3C 对标）。
4. 被删除、不得再写的表述：「posterior drift」、「anytime / streaming」（改「sample-synchronous，'same' 填充 look-ahead ≤ 128 ms」）、「LIF = 包络检测器」（改「LIF₁ 在阈值处整流 → conv2 平滑 → 包络估计」）、「1-bit 编码保相位所以空间滤波可行」（是假设，由 `afe` 消融支持）、「20× 能耗」进标题。
5. limitation 必写：transductive 对齐、单训练 seed / 共享初始化、配方依赖、SHU 地板效应、S 版 launch-bound 慢 3×。

## 8. 文档纪律（只维护这几份活文档）

| 文档 | 何时改 |
|:---|:---|
| `AGENTS.md` | **规则 / 契约 / 长期事实**变了才改；不记状态数字 |
| `docs/STATUS.md` | 作业状态、完成 cell 数、下一步、截稿倒计时变了就改（每次会话结束前更新一次） |
| `progress.md` | 每次实质动作追加一条（最新在上，含日期时刻） |
| `docs/DECISIONS.md` | 每个影响结果口径的决定：编号、背景、决定、后果 |
| `docs/PLAN_spiking_eegnex_icassp2027.md` | 方案变了改；§0.5 记先导 / 主实验结论 |

其余 `docs/*.md` 是参考卡片。**不再维护** FILE_CATALOG / operation_log 之类的镜像文档——目录变化看 `git status`。根目录与 `docs/` 下 Markdown 带 YAML frontmatter；文件名无空格、snake_case；CSV 表头英文 snake_case。报告结构：核心结论 → 目标 → 方法 → 协议 → 结果 → 分析 → 与前序关系 → 下一步 → 文件清单。

## 9. Slurm 运行纪律

- 提交前 `DRY_RUN=1` 看命令；`squeue -u $USER -o '%.10i %.14j %.2t %.11M %R'` 看状态；`gpu2node` 满时对 **PENDING** 作业 `scontrol update jobid=<id> partition=gpu3node`（一次只能一个分区）。
- 进度：`find outputs/experiments/<ds>/<run_id>/cells -name result.json | wc -l`；失败看 `<run>/failures__<tag>.json`，不在 Slurm 日志里翻。
- 接力链：每组 `N_LINKS` 个作业 `afterany` 串联，后一链重跑同一命令即续跑。
- 分析：本轮 **只有 zs**（D-035），用 `python scripts/analyze_snn_round.py --dataset <shu|wbci_shu> --run zs=<dir> --allow-empty-family --bench results_snapshot_2026-09-13/experiments/<...>_bench_v1 --out outputs/analysis/snn_round/<name>`。**不要**再传 `--run eaz=` / `--aligned-arm`。若以后重开对齐臂：`--run eaz=<dir> --aligned-arm eaz` 不能省（脚本靠它把 `eaz` 映射成预注册角色 `ea`，对不上会退出）。
- 先导数字的出处：`outputs/analysis/snn_round/shu_pilot_lr1e4/`（lr 1e-4：S/D vs EEGNeX、S vs D、S-EEGNet vs EEGNet）与 `shu_pilot_lr1e3/`（lr 1e-3：zs vs eaz 的 EA 效应、H1 交互、AdaBN、afe/spk/noinit 消融）的 `paired_tests.csv` / `interaction_H1.csv`；`shu_pilot/` 是 5 臂合表（只看 `per_model.csv`）。

## 10. 交接信息

- **路径变更（2026-09-18）**：`SYX/eeg-mi-online` → `SYX/ICASSP`（同一 git 仓库，`mv` 重命名，历史完整）；旧路径留兼容软链接给正在跑的作业。剪掉的遗留（跨 session Phase 0–2c 结果区 `1_..5_*`、`0_docs/`、`backup/`、`inbox/`、`models_eeg_foundation/`、旧路线文档、一次性脚本、快照里的跨 session run）在 `/share/home/yuan/SYX/_archive/eeg-mi-online_legacy_2026-09-18/`，git 历史与 `/share/home/yuan/SYX_backup_2026-09-13.tar.gz` 也都有。
- 3C processed manifest（`outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv`）的 `npz_path` 已改写为新路径。
- **可迁移 GitHub（D-041）**：`https://github.com/yixiangsunLeo/ICASSP`（private）。代码 + 灵魂文件 + 小结果文件走这里；2C/SHU/3C 的 npz 与预训练权重 **不进 git**，3090 按 `HANDOFF.md` 拷盘并 `scripts/repath_manifests.py`。训练仍写 `outputs/experiments/{wbci_shu,shu}/snn_*_zs_lr1e4_v1/`。
- 数据（4090）：`/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/`（2C）、`/share/workspace2/moto_imagination/SHU/processed/npz_clean/`（SHU）、`outputs/processed/wbci_shu_3c_mat_clean/`（3C）。BCI IV 2a（BNCI2014001）的 benchmark pkl 在 `/share/workspace2/moto_imagination/eegbench_benchmark_pkl/`（由 `../Benchmark` 生成），接入本框架需写 loader。
- 姊妹项目 `../Benchmark`（EEG 基础模型 benchmark，`AGENTS.md` 自成体系）：ICASSP 截稿后再回去做。
- 其他 GPU 项目共用集群配额，提交前看 `squeue` 总量。

## 11. 禁止事项

- 不把结果数字、历史叙事、临时 TODO 堆进本文件。
- 不在 2C 结果出来前把「非劣」「H1 成立/不成立」写成结论；不看测试集调主模型。
- 不删 `outputs/`、`checkpoints/`、`results_snapshot_2026-09-13/`、`logs/`——大不是删的理由。
- 不改第三方 / 官方 baseline 的数学（`code/models/paper_baselines/` 只认作者官方代码）。
- 不在 `SYX/` 顶层再放文件；ICASSP 的东西只进本目录，benchmark 的东西只进 `../Benchmark`。
