---
title: "Code Configs"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# Code Configs

## 1. Purpose

新配置层，分 datasets、models、experiments。

## 2. Portable paths

- 本机真实路径：`paths.local.yaml`、`datasets/*.local.yaml`（gitignore）。
- 仓库模板：`*.example.yaml` + 占位 `paths.yaml` / dataset yaml（`/CHANGE/ME/...`）。
- 实验 YAML 用逻辑键：`processed_manifest` / `shu_processed_manifest`，不要写死集群绝对路径。
- 换机步骤见根目录 `SETUP.md`。

## 3. Update Rules

参数优先写 YAML；代码不硬编码数据路径和超参。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `docs/ARCHITECTURE.md`: 全项目结构、代码分层、每个模块是活的还是遗留（2026-09-18 起取代 `0_docs/`，不再维护 FILE_CATALOG，见 `docs/DECISIONS.md` D-029）。
