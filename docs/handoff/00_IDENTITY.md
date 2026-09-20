---
title: "handoff 00 — 你的身份与权限"
created: "2026-09-20"
status: "active"
audience: "3090 migration AI"
---

# 你的身份

你在 **新机器 / 新账号**（用户说的 3090 服务器）上工作。仓库是 `https://github.com/yixiangsunLeo/ICASSP` 的克隆。

4090 集群（`login01`，分区 `gpu2node`/`gpu3node`，账号 `yuan` / `Linzh`）上可能仍有作业。那些作业的 `WorkDir` 是 `/share/home/yuan/SYX/ICASSP` 或软链 `eeg-mi-online`。**你碰不到它们，也不许去碰。**

# 你有权做的

- 在本账号下安装 conda 环境 `mi_torch_cu118`。
- 写本仓库的 `code/configs/paths.local.yaml`（gitignore）。
- 对本机拷来的 manifest 跑 `scripts/repath_manifests.py`。
- 只在用户**点名模型和队列**之后，才在本集群提交 GPU 作业。没点名：环境 + 数据 + 预检做完就停。
- 不要把 4090 / Linzh 正在跑的作业自动搬过来。
- 把新的 `result.json` 和 `outputs/analysis/**` 经 `scripts/sync_results_git.sh` commit + push 到 `yixiangsunLeo/ICASSP`。
- 更新本仓库的 `docs/STATUS.md`、`progress.md`。

# 你无权做的

- 改 `code/training/`、`code/experiments/cross_subject_protocols.py`、`code/models/` 的数学或配方。
- 改 `scripts/slurm/*.sbatch` / `submit_*.sh` 的默认分区去适配本机然后 push 回去（4090 作业还在用这些文件的历史副本；本机用环境变量 `PARTITION` / `CONDA_ROOT` / `sbatch -p`）。
- 改 `code/configs/experiments/snn_*.yaml` 的 `run_id`、`output_dir`、`lr`、`batch_size`、`micro_batch_per_model`、`split_seeds`。
- 把数据、权重、checkpoint、`paths.local.yaml` 推进 GitHub。
- force-push、amend、`git clean`、`git checkout -- .`。
- 替学长写论文正文或按测试集换主模型。

# 成功标准

1. `bash scripts/handoff_preflight.sh` 打印 `PREFLIGHT OK`。
2. GPU 冒烟 2 epoch 不崩。
3. 用户点名后，结果写入 `outputs/experiments/{wbci_shu,shu}/snn_*_zs_lr1e4_v1/cells/<model>__.../result.json`（和 4090 同一相对路径）。
4. `SYNC_COMMIT=1 SYNC_PUSH=1 bash scripts/sync_results_git.sh` 之后，4090 上 `git pull` 能读到这些 `result.json`。
