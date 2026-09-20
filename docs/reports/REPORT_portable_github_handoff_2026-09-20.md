---
title: "REPORT — 可迁移 GitHub 与 3090 交接（D-041）"
created: "2026-09-20"
status: "active"
---

# 核心结论

4090 读不到的 workspace 不能靠 GitHub 解决。GitHub 只带代码、灵魂文件和小体积 `result.json`。3090 上的 AI 必须按根目录 `HANDOFF.md` 重建环境、拷 9.6 G processed、改写 manifest 绝对路径，训练仍写入原来的相对目录，再用 `scripts/sync_results_git.sh` 把结果推回来。

# 目标

用户要求：把可迁移部分推到 `https://github.com/yixiangsunLeo/ICASSP`；交接文件让那边 AI 能独立把项目跑起来；灵魂文件一并过去；结果保存到同一相对路径并经 GitHub 同步回本树。

# 方法

三路专家审计（环境依赖 / 路径与数据 / git 可提交集合）后落盘交接与脚本，不改正在跑的训练闭包。

# 协议

- 不进 git：npz、权重、`paths.local.yaml`、`*.pt`、logs。
- 进 git：代码、`.cursor/rules/`、`AGENTS.md`、`HANDOFF.md`、`result.json`、分析 csv/md、snapshot 的 tables 与 bench `result.json`。
- 训练 `run_id` / `output_dir` 不改。
- 路径只通过本机 `paths.local.yaml` + `scripts/repath_manifests.py`。

# 结果

交接入口：`HANDOFF.md`（Phase 0–13）。附录：`docs/handoff/`。脚本：`repath_manifests.py`、`handoff_preflight.sh`、`sync_results_git.sh`。决定：D-041。

# 分析

训练加载 `npz_path` 原文，不重定根——这是换机第一失败点，所以预检会数 148/125/33 个 ok npz。`cell_signature` 不含主机名，4090 已完成的 cell 在 3090 会被跳过。

# 与前序关系

沿用 D-035（只 zs）、D-034 / D-040 的 run 目录约定。不取消 4090 作业。

# 下一步

1. 推送到 `yixiangsunLeo/ICASSP`（private）。
2. 用户在 4090 按 HANDOFF Phase 3 rsync 数据。
3. 3090 AI 跑到 `PREFLIGHT OK` 再训。

# 文件清单

- `HANDOFF.md`
- `docs/handoff/00_IDENTITY.md` `INSTALL.md` `DATA.md` `TRAIN.md` `RESULTS_SYNC.md` `PATHS_LOCAL.template.yaml`
- `scripts/repath_manifests.py` `scripts/handoff_preflight.sh` `scripts/sync_results_git.sh`
- `.gitignore`（结果放行）
- `docs/DECISIONS.md` D-041
