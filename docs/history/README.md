---
title: "history — 关键会话摘录与学长原话"
created: "2026-09-18"
updated: "2026-09-18"
status: "archived"
---

# history

只读归档：把「只留在聊天里」的知识落盘，供后续 Agent / 人回看。**活文档不在这里**（规则 `AGENTS.md`，状态 `docs/STATUS.md`）。

## 学长原话（微信截图，`assets/`）

| 日期 | 原话 | 含义 |
|:---|:---|:---|
| 2026-09-17 | 「我们之前不是跑了一些运动想象的跨被试的结果吧」「那个 eegnex 我看看能不能改一下」「加点别的」「再套到 snn 上」「师弟那些代码你还在吧」「我们跑了几个数据集。对应跑了几个模型」「把这几个梳理完成之后跟我开个会议」 | 论文方向 = 改 EEGNeX → 加对齐/别的 → 脉冲化；先梳理已有跨被试结果（→ `docs/PRIOR_RESULTS.md`、`docs/reports/MEETING_BRIEF_2026-09-18.md`） |
| 2026-09-18 | 「/share/home/yuan/SYX 把这里面的东西整理一下吧」「然后新建一个 /share/home/yuan/SYX/ICASSP」「我们到这个里面做」 | 本目录的由来（D-027） |

学长自己会改一版 EEGNeX；本仓库的 S-EEGNeX 是并行版本，最终采用哪版由学长定（`AGENTS.md` §1）。

## 文件

| 文件 | 内容 |
|:---|:---|
| `2026-09-17_18_session_icassp_plan_pilot_and_main_run.md` | 09-17 23:48 → 09-18 17:05 会话的文字导出（用户消息 + Agent 叙述，无工具噪音）：学长要求 → 方案 v1 → 四位专家评审 → v2 代码 → SHU 先导 → 主实验提交 |
| `expert_reviews_2026-09-18/01_icassp_area_chair_review.md` | 领域主席视角：三条贡献已有先例（Zhang 2025 / NiSNN-A），建议改「非劣性 + 对齐交互」框架、加 BCI IV 2a 锚点 |
| `expert_reviews_2026-09-18/02_statistics_methodology_review.md` | 统计：配对功效重算（MDE 0.6–1.0pp）、非劣性检验形式、H1 forking paths、两块主表、单 seed |
| `expert_reviews_2026-09-18/03_snn_algorithm_review.md` | SNN：神经元层 6 处静默失败缺陷（τ 范围、BN 初始化、读出、复位、dropout、chunk）、能耗算法核对 |
| `expert_reviews_2026-09-18/04_eeg_bci_signal_review.md` | EEG-BCI：1-bit 编码保相位不是定理、能耗参照应为 EEGNet、EA 后归一化、EA 增益预期、posterior drift 措辞 |
| `assets/*.png` | 学长微信截图原件（**已 gitignore，只留本机盘上，不随仓库推送**；文字转录见上表） |

四份评审如何被采纳：`docs/PLAN_spiking_eegnex_icassp2027.md` §0.1–§0.3、`docs/DECISIONS.md` D-001–D-019。
更早（2026-06 ~ 09-13）的会话未导出；那段时期的结论已沉淀在 `progress.md`、`docs/PLAN_cross_subject_benchmark_and_fusion.md`、`docs/RESULTS_TRUTH.md`。
