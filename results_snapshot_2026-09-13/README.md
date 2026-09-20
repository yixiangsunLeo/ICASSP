# results_snapshot_2026-09-13 — ANN 基线结果快照（只读归档）

2026-09-13 清理前把 `outputs/` 里**非二进制**的结果层原样镜像到本目录（目录结构与原 `outputs/experiments/` 一一对应：
`tables/` `figures/` `summaries/` 汇总表；`cells/**/result.json` 每折原始结果；`runs/*.csv`；`splits/`；`resolved_config.yaml`）。
`*.npz`（预测、embedding）与 `*.pt`（权重）当时已删，不可恢复。

**2026-09-18 迁移时只保留跨被试 run**（ICASSP 论文的配对基线）；跨 session 时代的 run、SHU 旧协议 `cross_subject_5fold_v1`（数字已在
`docs/PRIOR_RESULTS.md`）与 3 个 cell 的 `foundation_cross_subject_v1` 移到 `/share/home/yuan/SYX/_archive/eeg-mi-online_legacy_2026-09-18/results_snapshot_2026-09-13/`。
每个 run 目录里的 `resolved_config*.yaml` 是当时解析后的完整实验配置（对应的独立 YAML 已归档）。

| 保留的目录 | 内容 | 用途 |
|:---|:---|:---|
| `experiments/wbci_shu/wbci_2c_cross_subject_bench_v1/` | 2C，18 条目 × 5 折 × 2 划分 = 180 cell | **SNN 轮 `analyze_snn_round.py --bench` 的配对基线**；`docs/PRIOR_RESULTS.md` §3.1 |
| `experiments/shu/shu_cross_subject_bench_v1/` | SHU 2022，18 条目，180 cell | 同上；§3.2 |
| `experiments/shu/_job_ids/` | 上述 SHU run 的 Slurm 作业号 | 追溯 |
| `experiments/wbci_shu/paper_baseline_3c_821_v1/` | 3C，论文配方，7 模型 × 11 折 | `docs/RESULTS_TRUTH.md`；§3.3 |
| `experiments/wbci_shu/foundation_cross_subject_3class_v1/` | 3C，我们的配方 | 同上 |
| `experiments/wbci_shu/adv_821recipe_3c_v1/`, `atcnet_adv_3class_v1/`, `foundation_cross_subject_adv_l{01,05,10,20}_w5/` | 域对抗与 λ 扫描 | 「域对抗结案」的证据（`docs/ANALYSIS_atcnet_and_adversarial.md`） |
| `experiments/wbci_shu/wbci_3c_within_ceiling_v1/` | 3C 被试内 5 折（EEGNet / ATCNet） | 「难被试 = BCI illiteracy」的证据；注意数据饥饿，不能当上界 |

数字口径：`docs/RESULTS_TRUTH.md` 与 `docs/PRIOR_RESULTS.md` 的表全部可用 `scripts/summarize_cross_subject.py --run <本目录下的 run>` 重算复核。
本目录在 `.gitignore` 中被忽略（数千个小文件）；完整原件也在 `/share/home/yuan/SYX_backup_2026-09-13.tar.gz`。
