---
title: "ICASSP 2027 — Spiking EEGNeX for cross-subject MI-EEG"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/snn"
  - "#method/domain_generalization"
created: "2026-09-18"
updated: "2026-09-18"
status: "active"
---

# ICASSP 2027 · Spiking EEGNeX for Cross-Subject Motor Imagery

> **截稿 2026-09-24 20:00（北京）= 09-23 23:59 AoE。** 在这个目录里打开 Cursor 就能工作：
> 先读 [`AGENTS.md`](AGENTS.md)（规则与事实）→ [`docs/STATUS.md`](docs/STATUS.md)（现在做到哪、什么在跑）→
> [`docs/PLAN_spiking_eegnex_icassp2027.md`](docs/PLAN_spiking_eegnex_icassp2027.md)（方案 v2）。

## 0. 一句话

把三个队列上都最强的 ANN（EEGNeX，58 k 参数）逐层脉冲化，**让 EEG 的采样轴直接当脉冲时间轴**，配无标签的被试级对齐
（Euclidean Alignment + AdaBN），在 87 人 / 3 个跨被试队列上与已有的 **18 个 ANN 逐被试配对**比较：
主张 **非劣性**（δ = 1pp）、**对齐 × 网络类型的交互**、以及 **EEGNet 量级的能耗**。方案经 SNN / EEG-BCI / 统计 / 领域主席四位
"专家"评审修订后开跑；学长会自己改一版 EEGNeX，注册成一个 entry 即可进入同一张表。

## 1. 目录

```text
ICASSP/
├── AGENTS.md                 # 唯一灵魂记忆：身份、已验证事实、规则、契约（先读）
├── CLAUDE.md                 # Claude Code 入口 → AGENTS.md
├── progress.md               # 逐日流水（最新在上）
├── docs/
│   ├── STATUS.md             # 现在做到哪、20 个作业的状态、下一步、倒计时
│   ├── PLAN_spiking_eegnex_icassp2027.md   # 论文方案 v2（§0 评审采纳；§0.5 先导结果与主实验决定）
│   ├── PRIOR_RESULTS.md      # 之前跑过什么：3 队列 × 18 ANN（SNN 的配对基线）
│   ├── RESULTS_TRUTH.md      # WBCIC-SHU 3C 权威表
│   ├── DECISIONS.md          # D-001.. 每个影响口径的决定
│   ├── ARCHITECTURE.md       # 目录与代码分层、活/遗留标注
│   ├── ICASSP2027_SUBMISSION.md            # 截稿、格式、ORCID、检查单、稿件骨架映射
│   ├── PLAN_cross_subject_benchmark_and_fusion.md  # ANN 基线轮的协议与 27 条锁定决定
│   ├── ANALYSIS_atcnet_and_adversarial.md  # 已结案方向的分析
│   ├── CROSS_SUBJECT_PROTOCOL_MEMO.md / CROSS_SUBJECT_PROTOCOL_RESEARCH.md   # 协议问题清单 / 文献依据全文
│   ├── reports/              # 发给学长的文档（2026-09-18 简报 + 报告）
│   └── history/              # 关键会话摘录
├── paper/                    # ICASSP 2027 官方模板 + main.tex + refs.bib + figures/ tables/
├── code/                     # 训练框架；唯一入口 code/run.py；本轮新代码 code/models/spiking/
├── scripts/                  # make_snn_configs / analyze_snn_round / summarize / slurm/{submit_snn.sh, bench_gpu.sbatch}
├── tests/                    # pytest 106 项 + 1 skip（models/ + foundation/）
├── outputs/  checkpoints/  logs/          # run 产物（gitignore）
└── results_snapshot_2026-09-13/           # 只读归档：18 ANN 基线的表与 cells
```

## 2. 常用命令

```bash
cd /share/home/yuan/SYX/ICASSP
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate mi_torch_cu118

# 作业与进度
squeue -u $USER -o '%.10i %.14j %.2t %.11M %R'
find outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1/cells -name result.json | wc -l    # 目标 50
# failures__<tag>.json 每个作业都会写（哪怕 0 失败），看的是 n_cells_failed 字段：
python -c "import json,glob; [print(f, json.load(open(f)).get('n_cells_failed')) for f in sorted(glob.glob('outputs/experiments/*/snn_*_v1/failures__*.json'))]"

# 提交（每组一卡、48 h 接力；重复提交同一命令 = 断点续跑）
DRY_RUN=1 bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_wbci2c_zs_lr1e4.yaml c2z \
    "seegnex@s_hplif,seegnex@hplif" "eegnex,eegnet_official,seegnet@plif"

# 分析（CPU，登录节点可跑）：预注册检验 + H1 + 能耗 + 与 18 个 ANN 配对
python scripts/analyze_snn_round.py --dataset wbci_shu \
    --run zs=outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1 \
    --run eaz=outputs/experiments/wbci_shu/snn_wbci2c_eaz_lr1e4_v1 --aligned-arm eaz \
    --bench results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1 \
    --out outputs/analysis/snn_round/wbci_shu          # --aligned-arm eaz 必须给，否则脚本报错（H1 依赖它）

# 测试
pytest tests/models/test_spiking.py -q      # 13 项；全套 pytest tests/ -q 为 106 passed + 1 skipped（2026-09-18 剪枝后）
```

## 3. 数据边界

外部 raw / processed 路径由本机 `code/configs/paths.local.yaml` 配置（本集群只有这一份 local 文件），不进 Git；raw 只读。
3C processed 数据在 `outputs/processed/wbci_shu_3c_mat_clean/`。换机见 [`SETUP.md`](SETUP.md)，复现见 [`REPRODUCE.md`](REPRODUCE.md)。

## 4. 血缘

本仓库 = 2026-06 起的 `eeg-mi-online`（跨 session → 跨被试 → 18 模型 benchmark → SNN），2026-09-18 重命名为 `ICASSP`，
git 历史完整；剪掉的跨 session 遗留在 `../_archive/eeg-mi-online_legacy_2026-09-18/`。姊妹项目 `../Benchmark`（EEG 基础模型 benchmark）截稿后再做。
