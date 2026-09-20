---
title: "REPORT — 外部基线 adapter 验收与训练作业提交"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/benchmark"
created: "2026-09-19"
updated: "2026-09-19"
status: "active"
---

# 外部基线（D-034）：验收通过，11 个作业已全部提交

## 核心结论

对照轮已经可以跑：TCFormer / LaBraM-base / EEGPT / CBraMod 在三个队列上都能按与主实验相同的 zs + bench 配方建模、前向、写 `describe()`；训练作业 71184–71185 与 71206–71225 已进队列（`--nice=50`，不抢主实验的 4 张卡）。LaBraM 有一处真实 bug 已修：官方 checkpoint 的 `lm_head` 必须当作合法 unexpected 键，否则三个队列都加载失败。

## 目标

把 D-034 锁定的四个外部模型接到同一套跨被试协议上，并提交 zs 臂全量训练（外加 3C 上 ATCNet / EEG-Deformer / FBCNet 的 patience-25 补跑）。

## 方法

官方代码 vendoring 到 `code/models/external/<name>/_official/`；`adapter.py` 只做格式转换与项目契约 `{logits, features, confidence}`。registry 懒加载，排队中的 SNN 作业不会在 import 时拉进这些模块。实验 YAML 由 `scripts/make_baseline_configs.py` 从 `snn_*_zs_lr1e4.yaml` 生成，cell 写入同一 `snn_<ds>_zs_lr1e4_v1` 目录。

## 协议（与主实验 / 18 ANN 对齐）

zs 臂、Adam 1e-4、batch 128、patience 25、macro-F1、seed 0、split 41+42（3C LOSO）。FM 只做 250→200/256 Hz 重采样与通道词表；不做各论文自带的幅值缩放 / 额外滤波 / EA。EEGPT 权重为 HF 镜像转换件，论文需脚注。

## 结果

| 检查 | 结果 |
|:---|:---|
| 4 模型 × {2C 58ch/2cls, 3C 58ch/3cls, SHU 32ch/2cls} 前向 | 全部 OK（修 LaBraM `lm_head` 之后） |
| 权重 sha256 | LaBraM / EEGPT / CBraMod 与 `weights/MANIFEST.yaml` 一致；EEGPT `target_encoder.*` strict |
| 通道 | LaBraM 58+32 全保留；EEGPT SHU 丢 A1/A2（30 kept），T3–T6 映射到 T7/T8/P7/P8 |
| TCFormer `@n2` 22ch/4cls T=1125 | 77,820 参数（与官方 README 一致）；比较用 yaml 默认 depth 5 |
| pytest 外部套件 | 100%，exit 0 |
| 原 `tests/` 回归（忽略新建外部测试） | 100% + 1 skip，exit 0 |
| `make_baseline_configs.py --check` | 三份 `bl_*.yaml` 与生成器一致 |
| 提交 | roster 12 条 OK；10 新作业 × 2 链 = 71206–71225；未重投 `bl3c_ann` |

作业全部 PENDING（QOS / Priority）。主实验 71018 / 71021 / 71024 / 71030 仍占用 4 GPU。

## 分析

子 Agent 写的 adapter 主体可用；唯一会让训练在第一秒炸掉的是 LaBraM 过严的 unexpected-key 检查。其余偏差（EEGPT 丢耳垂、TCFormer depth 5 vs 77.8k 的 depth 2）都按协议 / PROVENANCE 写明，不改结构。

## 与前序关系

承接 D-034 与 00:40 的脚手架（registry 懒加载、`bl_*.yaml`、`submit_baselines.sh`、已投的 71184–71185）。未改 `code/configs/experiments/snn_*.yaml`、未改现有 `scripts/slurm/bench_gpu.sbatch`、未 scancel。

## 下一步

1. 作业上卡后先看 `outputs/experiments/_smoke/<tag>` 与 `failures__*_smoke.json`。
2. 全量 cell 进 `snn_*_zs_lr1e4_v1/cells/`；`analyze_snn_round.py` 无需改路径。
3. EEGPT 脚注；表行去 MDGEEG / EEGMoE、加 CBraMod（D-034 已定，稿件 splice 时改）。

## 文件清单

- `code/models/external/{tcformer,labram,eegpt,cbramod}/`（adapter + `_official/` + `PROVENANCE.md`）
- `code/configs/models/{tcformer,labram,eegpt,cbramod}.yaml`
- `code/configs/experiments/bl_{wbci2c,shu,wbci3c}_zs_lr1e4.yaml`
- `tests/models/test_{tcformer,labram,eegpt,cbramod,external_integration}.py`
- `scripts/slurm/{baseline_gpu.sbatch,submit_baselines.sh,baselines_plan.txt}`
- `logs/slurm/bl_job_ids.txt`
