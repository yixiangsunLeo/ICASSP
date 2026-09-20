---
title: "REPORT — 对比模型 split 43 交接（2C+SHU，Linzh）"
tags:
  - "#modality/eeg"
  - "#method/benchmark"
created: "2026-09-20"
updated: "2026-09-20 15:05"
status: "active"
---

# 对比模型 split-seed 43：2C + SHU，在 Linzh 跑

## 核心结论

对比模型的额外划分 seed **只加在 2C 和 SHU**。**3C 保持现有 LOSO，不加 seed 43。** 自己的 SNN 本轮不跑。90 个 cell 全部交给 Linzh；yuan 号上正在跑的作业不重复、不取消。登录 Linzh 后读 `/share/workspace3/icassp_linzh_share/HANDOFF_FOR_LINZH.md` 即可提交。

## 目标

学长要求论文对比模型再加一个划分 seed 43。用户指定：在 Linzh 号跑；yuan 已在跑的不要重复；SNN 还要改进，先把对比跑好；跑完后再让 yuan 搬结果。随后明确 3C 是 LOSO，不动。

## 方法

- 可跑的 9 个对比模型：`eegnet_official`、`fbcnet`、`atcnet`、`eegnex`、`labram`、`eegpt`、`eeg_deformer`、`tcformer`、`cbramod`。MDGEEG / EEGMoE 无可用代码（D-034），跳过。
- 配方与主实验 zs 相同（Adam 1e-4 / batch 128 / patience 25 / train seed 0）。
- 新配置 `bl_{wbci2c,shu}_zs_s43.yaml`：`split_seeds: [43]`，输出目录仍是 `snn_*_zs_lr1e4_v1`。**没有** 3C s43 配置，也没有改正在跑的 `snn_*.yaml` / `bl_*_zs_lr1e4.yaml`。
- cell 名：单独跑 seed 43 时写成 `model__split43__foldK__seed0`（避免 D-030 那种单 seed 丢掉 split 段）。3C seed 0 的 `model__foldK__seed0` 不变。

## 协议

| 队列 | 划分 | 本轮 |
|:---|:---|:---|
| WBCIC-SHU 2C | kfold 5 折，val 10 人 | 加 seed 43（已有 41/42） |
| SHU 2022 | kfold 5 折，val 5 人 | 加 seed 43（已有 41/42） |
| WBCIC-SHU 3C | LOSO 11 折，val 2 人，split_seed 0 | **不动** |

## 结果

尚未训练。计划 12 个 GPU 作业（2C 6 + SHU 6），每作业 2 段 48 h 接力；合计 90 cell。2C TCFormer 单独一卡（上次三进程共卡 OOM）。

## 分析

3C 每个被试已经当且仅当测试一次；再换一个 split seed 只是重抽验证被试，不是新的测试划分，所以本轮不加。2C / SHU 的 seed 43 才是新的 5 折被试划分。

## 与前序关系

D-034 外部基线是 split 41+42（3C LOSO 0）。D-036 只给 2C/SHU 对比模型补 43。yuan 收尾作业 71034 / 71213 / 71222 继续跑 41/42 与 3C LOSO。

## 下一步

1. 用 Linzh 账号打开交接文件，预检通过后提交。
2. 跑完后用户再让 yuan 执行 `merge_back_s43.sh`。
3. SNN 的 seed 43 等模型改完再议。

## 文件清单

- `/share/workspace3/icassp_linzh_share/HANDOFF_FOR_LINZH.md` — Linzh 开跑说明
- `/share/workspace3/icassp_linzh_share/merge_back_s43.sh` — 以后 yuan 合并用
- `code/configs/experiments/bl_{wbci2c,shu}_zs_s43.yaml`
- `scripts/slurm/baselines_plan_s43.txt`
- `docs/DECISIONS.md` D-036
