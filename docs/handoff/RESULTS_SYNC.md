---
title: "handoff — 结果回传 GitHub（你来执行）"
created: "2026-09-20"
status: "active"
audience: "3090 migration AI"
---

# 目标

训练写在仓库相对路径 `outputs/experiments/.../cells/<cell>/result.json`，和 4090 **同一位置**。
你 push 这些小文件；4090 上的 Agent `git pull` 之后用同一路径读。

# 进 git / 不进 git

| 进 | 不进 |
|:---|:---|
| `outputs/experiments/**/result.json` | `*.pt` `*.pth` `*.ckpt` `best.pt` `last.pt` |
| `failures__*.json` `resolved_*.json` `resolved_*.yaml` | `predictions.npz` 任何 `*.npz` |
| `outputs/analysis/**/*.csv` `*.md` | `outputs/processed/**`（2 G 数据） |
| `docs/STATUS.md` `progress.md` | `code/configs/paths.local.yaml` |
| snapshot 的 `tables/` 与 bench `cells/*/result.json` | `logs/` `checkpoints/` `weights/` |

# 你每次收工要跑的

```bash
# 1. 更新 STATUS（cell 计数）和 progress 顶部一条
# 2. 预览将要提交的文件
bash scripts/sync_results_git.sh
# 3. 确认 --stat 里没有 .pt/.npz/paths.local.yaml
# 4. 提交并推到 yixiangsunLeo/ICASSP
SYNC_COMMIT=1 SYNC_PUSH=1 bash scripts/sync_results_git.sh
```

commit 信息用脚本默认的即可（「Sync experiment result.json ...」）。不要 `--amend`，不要 `--force`。

# remote

你 clone 的就是 `yixiangsunLeo/ICASSP`，`origin` 对。脚本会 `git push origin HEAD`。

4090 yuan 树的 `origin` 仍是旧 `EEG.git`。那边另有 remote `icassp` 指到本仓库。你不要改 4090 的 remote。

# 分析（可选，CPU）

本轮没有 eaz，分析不要传 `--run eaz=` / `--aligned-arm`：

```bash
python scripts/analyze_snn_round.py \
  --dataset wbci_shu \
  --run zs=outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1 \
  --allow-empty-family \
  --bench results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1 \
  --out outputs/analysis/snn_round/wbci2c_zs_<date>/
```

SHU 把 `--dataset shu`、run 和 bench 换成 `shu/snn_shu_zs_lr1e4_v1` 与 `shu/shu_cross_subject_bench_v1`。
表是脚本生成的，不要手改数字。分析产出一并走 sync 脚本。
