---
title: "Experiment Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-07"
status: "active"
---

# Experiment Layer

## 1. Purpose

实验协议层，组合 dataset、model、method、trainer、metrics。

## 2. What Belongs Here

| 文件 | 泛化轴 | 划分单位 |
|:---|:---|:---|
| `cross_subject_protocols.py` | **跨被试（唯一主线）**：被试级 LOSO / k-fold / holdout；归一化语法 `per_sample_zscore` / `ea_session[_mu]+…`（逐 session EA）；评估时 AdaBN；逐被试 `spike_stats`；`cell_signature` | **subject**；可选 session 级 train/val |
| `session_protocols.py` | 只为 `TrainSpec` / `evaluate_predictions` 被上面复用而保留（跨 session 协议本身已归档） | — |
| `metrics.py` / `data_quality.py` | 共用指标与质检 | — |

2026-09-18 剪枝：`session_multisource_protocols.py`、`session_alignment_protocols.py`、`prototype_drift*.py`、`session_tta.py`、
`cross_session_e2e_protocols.py`（跨 session 时代 Phase 2a–3 与 SHU 被试内跨 session）已移到
`../_archive/eeg-mi-online_legacy_2026-09-18/code/experiments/`。它们不在主实验作业的 import 闭包里（用实际命令跑 CPU 微型 run 记录
`sys.modules` 核实）。`code/runners.py` 里对应的 runner 函数暂留，作业结束后删除。

## 3. Update Rules

新增协议必须说明数据划分、防泄漏规则、输出 CSV 字段。

跨被试协议的三条铁律（`cross_subject_protocols.py` 里有断言 + `tests/foundation/` 有对应测试）：
测试被试绝不出现在 train/val；验证集来自留出训练被试（`val_mode=subjects`）、训练被试的 trial
切片（`val_mode=trials`），或**同批非测试被试按 session 切**（`val_mode=sessions`：train
ses-01+02 / val ses-03，对齐 SHUv5 / WBCIC-SHU 3C 论文）；归一化必须是 per-trial fit-free 的。

**三曲线（2026-08-10）**：`train.curves: true` 时，协议层额外构建两个**只做监控**的 loader
（训练集固定子集在 eval 模式 + 测试被试），交给 `e2e_trainer` 每 epoch 记录进 `history` 的
`train_eval` / `test`。**test 绝不参与模型选择**——`best.pt`、早停、`best_score` 只读 val，
守卫在 `code/training/e2e_trainer.py`。绘图见 `scripts/plot_three_curves.py`。

**cell 目录命名**（`scripts/plot_three_curves.py` 能解析前两种）：
`<model>__fold<k>__seed<s>`（单划分 / LOSO）、`<model>__split<ss>__fold<k>__seed<s>`（扫多个 `split_seeds`）。
注意 `split_seeds` 改的是**被试划分**，`train.seeds` 改的是**训练随机种子**，两者不要混；往已有 run 目录续跑时命名模式必须一致（D-030）。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `docs/ARCHITECTURE.md`: 全项目结构、代码分层、每个模块是活的还是遗留（2026-09-18 起取代 `0_docs/`，不再维护 FILE_CATALOG，见 `docs/DECISIONS.md` D-029）。
