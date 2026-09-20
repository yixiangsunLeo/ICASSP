---
title: "CLAUDE.md — 兼容入口（指向 AGENTS.md）"
tags:
  - "#modality/eeg"
created: "2026-06-12"
updated: "2026-09-20"
status: "active"
---

# CLAUDE.md

本文件只是给自动读取 `CLAUDE.md` 的工具用的入口。

- **在 3090 / 新机器上：** 先完整执行根目录 **`HANDOFF.md`**（那边 AI 的主程序），再读 `AGENTS.md`。
- **在 4090 / 本目录：** **唯一权威规则文件是同目录的 `AGENTS.md`**，请先读它。

读完 `AGENTS.md` 后按需读：`docs/STATUS.md`（现在做到哪、什么在跑、截稿倒计时）→ `progress.md` 顶部一条 →
`docs/PLAN_spiking_eegnex_icassp2027.md`（方案 v2；§0.5 先导结果与主实验决定）→ `docs/PRIOR_RESULTS.md`（之前跑过的 3 队列 × 18 ANN）→
`docs/ARCHITECTURE.md`（代码分层）。

最关键的三条（细节见 `AGENTS.md`）：

- 截稿 **2026-09-24 20:00 北京**。4090 上不要 `scancel` 别人的作业、不要改正在被作业读取的 `code/` 与 `scripts/slurm/`。
- 不编造结果；测试集不参与模型选择；数字只能来自脚本生成的表。
- 做了事记 `progress.md`，状态变了改 `docs/STATUS.md`，决定记 `docs/DECISIONS.md`。3090 按 `HANDOFF.md` Phase 12 同步 `result.json`；其余不 commit 除非用户要求。
