---
title: "Dataset Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Dataset Layer

## 1. Purpose

统一数据集适配层，负责 Session 抽象、manifest/session 索引、通道对齐。

## 2. What Belongs Here

- 与本目录职责直接相关的文件。
- 能帮助后来的人理解实验、代码或结果的索引说明。
- 必要时放 README、manifest、报告或轻量配置；大型数据和 checkpoint 不放在文档目录。

## 3. Update Rules

不要在 `load()` 中偷偷做重预处理；raw-to-npz 必须显式实验执行。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `docs/ARCHITECTURE.md`: 全项目结构、代码分层、每个模块是活的还是遗留（2026-09-18 起取代 `0_docs/`，不再维护 FILE_CATALOG，见 `docs/DECISIONS.md` D-029）。
