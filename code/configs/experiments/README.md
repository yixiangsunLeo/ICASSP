---
title: "Experiment Configs"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-09-18"
status: "active"
---

# Experiment Configs

## 1. Purpose

可运行的实验配置。`experiment.name` 决定用哪个 runner（`code/runners.py` 的 `PHASE_RUNNERS`）；
本轮只用 `foundation_cross_subject`（被试级 k-fold / LOSO，`code/experiments/cross_subject_protocols.py`）。

## 2. What Belongs Here（2026-09-18 剪枝后）

| 配置 | 用途 | 状态 |
|:---|:---|:---|
| `snn_{shu,wbci2c,wbci3c}_{zs,ea,eamu,eaz}_lr1e{3,4}.yaml`（24 个） | ICASSP 2027 SNN 轮网格；由 `scripts/make_snn_configs.py` 生成，**不要手改** | **主实验**用 `*_zs_lr1e4` / `*_eaz_lr1e4`（6 个）；SHU 先导用了 `snn_shu_{zs_lr1e4,zs_lr1e3,ea_lr1e3,eamu_lr1e3,eaz_lr1e3}`；其余是网格里没跑的组合 |
| `wbci_2c_cross_subject_bench.yaml`, `shu_cross_subject_bench.yaml` | 18 个 ANN 条目的基线 run（已跑完，结果在 `results_snapshot_2026-09-13/`）；SNN 配置的 `cross_subject` 块与它们逐字相同 | 保留：协议出处 + `tests/models/test_fusion_and_specs.py` 读取 |

跨 session 时代与 2026-08 的旧配置（phase0–3、`foundation_cross_subject*`、`adv_*`、`paper_baseline_3c_821`、
`shu_cross_subject_5fold`、`wbci_3c_within_subject_ceiling` 等 33 个）已移到
`../_archive/eeg-mi-online_legacy_2026-09-18/code/configs/experiments/`；它们跑出的每个 run 在
`results_snapshot_2026-09-13/experiments/<ds>/<run_id>/resolved_config*.yaml` 里都保留了当时解析后的完整配置。
对应的 runner 函数（`run_phase0_drift` … `run_cross_session_e2e`）仍留在 `code/runners.py` 里等作业结束后删除
（它们 lazy import 的模块已归档，调用会 ImportError；主实验只走 `run_foundation_cross_subject`）。

## 3. Update Rules

- 改网格 → 改 `scripts/make_snn_configs.py` 再重新生成；新设置用新 `run_id`。
- 两个数据集永远分开跑：WBCIC 58ch、SHU 32ch，各自 `outputs/experiments/<dataset>/<run_id>/` 与 `checkpoints/<dataset>/<run_id>/`。
- 单划分（`--split-seeds 41`）与多划分（配置默认 `[41,42]`）的 cell 目录命名不同（`<model>__fold<k>__seed<s>` vs
  `<model>__split41__fold<k>__seed<s>`）；往已有 run 目录续跑时两者必须一致，否则已完成 cell 会被重训（`docs/DECISIONS.md` D-030）。
- **主实验作业排队期间（2026-09-18 起）不要改 `snn_*.yaml`**：作业启动时才读配置。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `docs/ARCHITECTURE.md`: 全项目结构和文件职责说明（含哪些配置是活的、哪些是遗留）。
- `docs/PLAN_cross_subject_benchmark_and_fusion.md` §4: 基线协议（6:2:2、split seeds 41/42、配方）。
