---
title: "handoff — 数据与路径（你来执行）"
created: "2026-09-20"
status: "active"
audience: "3090 migration AI"
---

# 原则

1. GitHub **没有** EEG。只有代码、配置、灵魂文件、以及小体积的 `result.json` / 分析表。
2. 训练读 csv 里的 `npz_path` **原文**，没有 `DATA_ROOT` 环境变量可救。拷完必须 `scripts/repath_manifests.py`。
3. 实验 YAML 只有逻辑键 `processed_manifest` / `shu_processed_manifest` / `wbci_3c_processed_manifest`。**不要改 YAML。**
4. 2C 有 5 个 `status=failed` 的 session，必须原样保留。`status_filter: ["ok"]` 会跳过它们。删被试会打乱折。`repath_manifests.py --verify` **只检查 ok 行** 的 npz 是否存在；failed 行仍可能留着 4090 前缀，这是正常的。

# 三棵必须到位的树

| 队列 | 拷到哪里 | 行数 / ok / 人 | 旧前缀（rewrite 的 OLD） |
|:---|:---|:---|:---|
| 2C | `$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean/` | 153 / 148 / 51 | `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean` |
| SHU | `$DATA_ROOT/SHU/processed/npz_clean/` | 125 / 125 / 25 | `/share/workspace2/moto_imagination/SHU/processed/npz_clean` |
| 3C | `$REPO/outputs/processed/wbci_shu_3c_mat_clean/` | 33 / 33 / 11 | `/share/home/yuan/SYX/ICASSP/outputs/processed/wbci_shu_3c_mat_clean` |

3C 的 `paths.local.yaml` 键保持相对路径；要改的是 **csv 里面** 的绝对 `npz_path`。

2C / SHU **不要**试图从 raw 重做（BDF + ICA，天级，raw 也在你读不到的 workspace2）。

3C 替代：拷官方 `.mat`（4090 上 `/share/workspace2/moto_imagination/WBCIC_SHU/derivatives/3C dataset_processeddata`，2.0 G）然后：

```bash
python scripts/preprocess_wbci_3c.py \
  --mat-root /path/to/3C\ dataset_processeddata \
  --out-root outputs/processed/wbci_shu_3c_mat_clean
```

这会写出本机绝对路径，3C 就不用 sed。

# 用户在 4090 上应跑的 rsync

见 `HANDOFF.md` Phase 3。你负责填 `<3090_USER@HOST>`、`<DATA_ROOT>`、`<REPO>`，把命令发给用户，等 `du -sh` 对得上再继续。

# 改写命令

见 `HANDOFF.md` Phase 7。只允许用 `scripts/repath_manifests.py`。不要手写 csv，不要 `sed` 漏掉 `meta_path` / `report_path`（脚本三个栏目一起改）。

# `paths.local.yaml`

```bash
cp docs/handoff/PATHS_LOCAL.template.yaml code/configs/paths.local.yaml
sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" code/configs/paths.local.yaml
```

`raw_data.shu_2c_root` 不能仍是 `/CHANGE/ME`，否则 `load_paths()` 直接 raise。目录本身训练时不必存在。

# 权重

分辨率（先到先得）：`$ICASSP_WEIGHTS_DIR` → `paths.local.yaml:pretrained_weights_dir` → `<repo>/weights`。
4090 的 `weights` 软链不要复现。建真目录，放三文件，sha256 见 `HANDOFF.md` Phase 8。

# 折是现算的

`splits/*.json` 不进 git。`make_subject_folds` 用 `np.random.default_rng(split_seed)` 在 **ok 被试名单** 上切。名单必须仍是 51 / 25 / 11 人，否则所有 cell 对不上。

# 不要拷的

- `checkpoints/`（7 G，只为中断续训；有 `result.json` 就会跳过整 cell）
- 4090 的 conda
- 4090 的 `paths.local.yaml`
- `outputs/processed` 以外的 raw
