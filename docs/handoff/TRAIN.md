---
title: "handoff — 训练（你来执行）"
created: "2026-09-20"
status: "active"
audience: "3090 migration AI"
---

# 输出路径锁死

配置里已经写好，不要 `--out`，不要改 `run_id`：

| 队列 | 配置 | run 目录 |
|:---|:---|:---|
| 2C zs | `code/configs/experiments/snn_wbci2c_zs_lr1e4.yaml` 以及 `bl_wbci2c_*.yaml` | `outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1` |
| SHU zs | `snn_shu_zs_lr1e4.yaml` / `bl_shu_*.yaml` | `outputs/experiments/shu/snn_shu_zs_lr1e4_v1` |
| 3C zs | `snn_wbci3c_zs_lr1e4.yaml` / `bl_wbci3c_*.yaml` | `outputs/experiments/wbci_shu/snn_wbci3c_zs_lr1e4_v1` |

cell 目录名：

- 2C / SHU、划分种子 ≠ 0：`<model>__split<NN>__fold<K>__seed0`，`@` 换成 `-`
- 3C LOSO（唯一 split_seed 0）：`<model>__fold<K>__seed0`

例：`seegnex-s_hplif__split41__fold0__seed0`，`adahgnn-mi__fold0__seed0`，`eegnex__split43__fold2__seed0`。

完成标记是 `cells/<tag>/result.json`。签名含配方和被试名单，**不含主机名、不含绝对数据路径**。所以 4090 训完的 cell 在这边会被跳过——前提是 git 里已经有那个 `result.json`，且你没有改配方。

# 先数再跑

```bash
for r in wbci_shu/snn_wbci2c_zs_lr1e4_v1 shu/snn_shu_zs_lr1e4_v1 wbci_shu/snn_wbci3c_zs_lr1e4_v1; do
  printf '%-45s %s\n' "$r" "$(find outputs/experiments/$r/cells -name result.json 2>/dev/null | wc -l)"
  ls outputs/experiments/$r/cells 2>/dev/null | sed 's/__.*//' | sort | uniq -c
done
```

**默认不提交任何作业。** 4090 / Linzh 正在跑的不要搬。等用户点名；后续主线是 `seegnex@hplif`（S-EEGNeX-D）和 `adahgnn@mi` / `adahgnn@mi_mix`。点名之后只跑他说的条目，不要重跑已经有 `result.json` 的 cell。

# 提交方式

人工入口：

```bash
python code/run.py --config code/configs/experiments/<name>.yaml \
  --models a,b@variant --split-seeds 41 --tag-suffix X --device cuda
```

Slurm（一组一卡，`N_LINKS` 条 接力）：

```bash
export PARTITION='<sinfo 里的 GPU 分区>'
export CONDA_ROOT="$(conda info --base)"
export MI_ENV=mi_torch_cu118
export N_LINKS=2
DRY_RUN=1 bash scripts/slurm/submit_snn.sh <config> <tag> "<group>"
# 分区正确后再去掉 DRY_RUN
```

外部基线 / AdaHGNN：

```bash
DRY_RUN=1 NICE=50 IDS_FILE=logs/slurm/bl_adahgnn_job_ids.txt \
  bash scripts/slurm/submit_baselines.sh scripts/slurm/baselines_plan_adahgnn.txt
```

`SMOKE_FIRST` 默认开：先 1 折 2 epoch 写 `_smoke/<tag>`，失败整作业退出。保持开着。

# 配方（不许改）

Adam 1e-4 / batch 128 / 无 scheduler / max 500 / patience 25 / monitor macro-F1 / 训练 seed 0 / 2C+SHU 划分 41+42 / 3C LOSO seed 0 / `normalization: per_sample_zscore`。本轮不跑 eaz（D-035）。SNN 不加 seed 43。对比模型 seed 43 只用 `bl_{wbci2c,shu}_zs_s43.yaml`。

3090 是 24 GB，和 4090D 同量级。先一卡一进程。OOM 时 **不要** 改 `micro_batch_per_model`（在 cell 签名里）。先减共卡进程数。

# 本轮不要跑

- 任何 `snn_*_eaz_*.yaml` / `*_ea_*` / `*_eamu_*`（H1 已取消）
- 原版 `adahgnn`（无 `@mi`）
- 3C 的 split 43
- 未在 STATUS 里列为缺口的模型

# 失败怎么看

`outputs/experiments/<ds>/<run_id>/failures__<tag>.json` 的 `n_cells_failed`。不是 slurm `.err`。
同一条 submit 命令再执行 = 续跑，永远安全。
