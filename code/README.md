---
title: "Code Framework"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-09-18"
status: "active"
---

# Code Framework

## 1. Purpose

模块化代码框架（数据集/模型/方法/实验分层）。人工入口是 `code/run.py`。

## 2. What Belongs Here

- 与本目录职责直接相关的文件。
- 能帮助后来的人理解实验、代码或结果的索引说明。
- 必要时放 README、manifest、报告或轻量配置；大型数据和 checkpoint 不放在文档目录。

## 3. Update Rules

新增数据集/模型/方法/实验必须落在对应层，同时更新 config 与最近的 README；分层与活/遗留标注见 `docs/ARCHITECTURE.md` §2。
**主实验作业运行期间（2026-09-18 起）不要改动本目录**，见 `AGENTS.md` §3.5。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `docs/ARCHITECTURE.md`: 全项目结构、代码分层、每个模块的状态。
- `code/models/spiking/`: 本轮（ICASSP 2027）新增的 S-EEGNeX / S-EEGNet。
