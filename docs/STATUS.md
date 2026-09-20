---
title: "STATUS — 现在做到哪、什么在跑、下一步（每次会话结束前更新）"
tags:
  - "#modality/eeg"
  - "#method/snn"
created: "2026-09-18"
updated: "2026-09-20 20:30"
status: "active"
---

# STATUS

> 这是「当前状态」的唯一活文档。规则在 `AGENTS.md`，方案在 `docs/PLAN_spiking_eegnex_icassp2027.md`，
> 逐日流水在 `progress.md`。**每次会话结束前把本文件的时间戳和 §1–§3 更新一遍。**

## 0. 倒计时

| 里程碑 | 时间（北京） | 备注 |
|:---|:---|:---|
| **ICASSP 2027 截稿** | **2026-09-24（周四）20:00** | = 09-23 23:59 AoE。官网 Call for Papers 已显示 September 23, 2026（从 09-16 延期一次）；CMS 投稿系统的 Paper Kit 页面截至 09-18 仍写 16 September——**投稿前先登录 CMS 确认系统开放**（见 `docs/ICASSP2027_SUBMISSION.md`） |
| **主实验 zs 出齐（18:10 更新）** | **2C 主 50/50 ✅**；**3C 主 55/55 ✅**；**SHU 主 69/70**（只剩 `seegnex@s_plif` split42 fold4 在训，估 19:30） | 中期数字见 `docs/reports/REPORT_interim_results_2026-09-19.md`。eaz 已取消（D-035） |
| BCI IV 2a 锚点 loader 截止 | 09-20 早（已过） | 三位评审都提；不做则在 limitation 里说明 |
| 写作 | 2C/3C 主实验数字**现在就能分析**；2C+SHU 外部三模型已在正式目录；对照表待 yuan 剩余 3 作业 | 截稿 09-24 20:00；学长仍未定题。底稿 = `old_reference.tex` |
| **全部训练出齐（估）** | **yuan 收尾 3 卡：今晚 19:30–21:00**；**AdaHGNN-MI 上卡后约 3–6 h/队列（估 09-21 凌晨）**；**Linzh seed-43：估 09-21 下午** | 全集群 40 卡满载 |
| **外部基线轮（D-034）** | Linzh 60 cell **已合并**；yuan 还在跑：2C tcformer 7/10、3C eegpt 9/11 | 3C labram/cbramod/ANN/tcformer(SHU+3C) ✅ |
| **AdaHGNN-MI（D-040）** | **71825–71830 仍 PENDING，从未上卡**（无 Slurm 日志） | 等 yuan 收尾腾卡；冒烟后再全量 |
| **Linzh split 43（D-036）** | 4 卡在跑 2C；split43 已落 12 个 `result.json`（eegnet 5 + fbcnet 4 + atcnet 2 + labram 1）；SHU 尚未开 | 90 cell；QOS 4 卡满 |
| **3090 可迁移仓库（D-041）** | `https://github.com/yixiangsunLeo/ICASSP`（private）。交接 = 根目录 `HANDOFF.md`。npz 不进 git | 那边 AI 按 HANDOFF 落地；结果经 `result.json` 同步回本树 |

## 3090 缺口（那边 AI 只补没有 `result.json` 的格子）

以本仓库 pull 下来的 `outputs/experiments/**/result.json` 为准，不要重跑已齐的条目。期望：2C/SHU 每模型 10 cell（split 41+42），3C 每模型 11（LOSO），s43 每模型每队列 5。

| 优先 | 条目 | 配置 / plan | 期望 | 4090 侧备注（20:30） |
|:---|:---|:---|:---|:---|
| 1 | `adahgnn@mi` + `adahgnn@mi_mix` × 三队列 | `scripts/slurm/baselines_plan_adahgnn.txt` | 62 | 71825–71830 在 4090 仍可能排队；先 `find` 再决定是否在 3090 开，避免双边同时写同一 cell |
| 2 | 2C `tcformer` 未齐折、3C `eegpt` 未齐折、SHU `seegnex@s_plif` 最后 1 cell | 原 `snn_*` / `bl_*` 命令续跑 | 缺几个补几个 | 4090 收尾作业可能先完成；有 `result.json` 就跳过 |
| 3 | split 43（只 2C+SHU） | `bl_{wbci2c,shu}_zs_s43.yaml` | 90 | Linzh 在 4090 跑；3090 不要抢已经在跑的模型 |

**禁止**在 3090 重跑 2C 主 50、3C 主 55、以及已经 10/10 的外部基线。

## 1. 正在跑（lr 1e-4；**只剩 `zs` 臂**（逐 trial z-score）；eaz 臂 10 个作业 09-19 01:50 已取消，D-035）

| 组 | 数据集 · 臂 | 作业（接力链） | 条目 | run_id → 输出目录 | 预期 cell |
|:---|:---|:---|:---|:---|---:|
| c2zA/B/C / c2z_g2 | 2C · zs | **全部 COMPLETED 0:0**（71230–71234、71021 链） | eegnex / eegnet / S-EEGNet / D / S | `outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1` | **50/50 ✅** |
| c3z_g1 | 3C · zs | **71030 COMPLETED 1-11:07** → 71031 3 min 跳过 | 5 个主条目 | `outputs/experiments/wbci_shu/snn_wbci3c_zs_lr1e4_v1` | **55/55 ✅** |
| shz_g1 | SHU · zs | **71034 RUNNING（gpu03，已跑 25.5 h）** → 71035 | 只剩 `seegnex@s_plif` split42 fold4（ep15 / 估 ~70–90） | `outputs/experiments/shu/snn_shu_zs_lr1e4_v1` | 70（69 完成 + 1 在训） |
| ~~eaz 各组~~ | — | 71018–71020、71024–71029、71032–71033、71036–71037 **已取消** | — | 不用 | — |

**13:10 落盘**：2C 主 50/50；3C 主 55/55；SHU 除 s_plif 外全齐。对照表（未完成用 —）见 `docs/reports/REPORT_cohort_tables_2026-09-20.md`；分析 `outputs/analysis/snn_round/{wbci2c,shu,wbci3c}_zs_2026-09-20_1300/`。中期 `*_interim_2026-09-19_1220` 已过时。

```bash
squeue -u $USER -S -Q -o '%.10i %.14j %.2t %.11M %.12Q %R'      # -S -Q = 按优先级排，看谁下一个上卡
for r in wbci_shu/snn_wbci2c_zs_lr1e4_v1 wbci_shu/snn_wbci3c_zs_lr1e4_v1 shu/snn_shu_zs_lr1e4_v1; do
  printf '%-45s %s\n' "$r" "$(find outputs/experiments/$r/cells -name result.json 2>/dev/null | wc -l)"
  ls outputs/experiments/$r/cells | sed 's/__.*//' | sort | uniq -c; done
# failures__<tag>.json 每个作业都会写（0 失败也写）；看 n_cells_failed 字段
python -c "import json,glob; [print(f, json.load(open(f)).get('n_cells_failed')) for f in sorted(glob.glob('outputs/experiments/*/snn_*_v1/failures__*.json'))]"
```

若 gpu2node 一直排不上：对 PENDING 的作业 `scontrol update jobid=<id> partition=gpu3node`（一次一个分区）。不要 scancel。

## 1b. 外部基线轮（D-034；zs 臂；cell 写进上表同一 run 目录；清单 `logs/slurm/bl_job_ids.txt`；计划 `scripts/slurm/baselines_plan.txt`）

| 作业 | 内容 | 状态（09-20 12:55） | 预期 cell |
|:---|:---|:---|---:|
| `bl3c_ann` **71184 → 71185** | 3C · atcnet + eeg_deformer + fbcnet | ✅ COMPLETED | 33/33 |
| `bl2c_labram / eegpt / cbramod` **71206–71211** | 2C · 三模型 | **已合并进 yuan 树后 scancel**（09-20 13:00） | **30/30 ✅** |
| `bl_tcformer` **71212 ✅ → 71213 R gpu01 8.9 h** | 2C+SHU+3C · tcformer | SHU+3C ✅；**71213 单跑 2C**（7/10 完成；正在 split42 fold2 ep40 patience 8） | 31（28 ✅ + 3 未齐） |
| `blsh_labram / eegpt / cbramod` **71214–71219** | SHU · 三模型 | **已合并进 yuan 树后 scancel**（09-20 13:00） | **30/30 ✅** |
| `bl3c_labram` **71220 → 71221** | 3C · labram | ✅ COMPLETED | 11/11 |
| `bl3c_eegpt` **71222 R gpu01 9.6 h** → 71223 | 3C · eegpt | **9/11 完成**；正在 fold9 ep7（fold10 未开） | 11 |
| `bl3c_cbramod` **71224 → 71225** | 3C · cbramod | ✅ COMPLETED | 11/11 |
| **Linzh 71359–71371** | 2C+SHU labram/eegpt/cbramod | **12/12 COMPLETED 0:0**；60 cell 已 `cp -n` 进 yuan 正式 `cells/`（13:00） | 60/60 ✅ |

Adapter 已过：4 模型 × 3 队列前向、sha256、LaBraM `lm_head` 合法 unexpected 键、pytest 外部套件 + 原测试集回归。每个作业先 `SMOKE_FIRST`（1 折 2 epoch 到 `_smoke/<tag>`），失败即退出；第 2 链 `SMOKE_FIRST=0` 续跑。进度：
`for r in wbci_shu/snn_wbci2c_zs_lr1e4_v1 shu/snn_shu_zs_lr1e4_v1 wbci_shu/snn_wbci3c_zs_lr1e4_v1; do ls outputs/experiments/$r/cells | sed 's/__.*//' | sort | uniq -c; done`

## 1c. 对比模型 split 43（D-036；Linzh 跑；**只 2C + SHU**）

学长要对比模型再加划分 seed 43。**3C 保持 LOSO，不加 43。** 不跑 SNN。yuan 上已在跑的 71034 / 71213 / 71222 **不重复、不取消**。

| 项 | 内容 |
|:---|:---|
| 队列 | 2C、SHU（各 5 折 × seed 43）；3C 不在本轮 |
| 模型 | `eegnet_official, fbcnet, atcnet, eegnex, labram, eegpt, eeg_deformer, tcformer, cbramod`（MDGEEG / EEGMoE 无代码，跳过） |
| 配置 | `code/configs/experiments/bl_{wbci2c,shu}_zs_s43.yaml`（新文件，未改 live yaml） |
| 作业 | 12 个 GPU 作业 × 2 链；plan `scripts/slurm/baselines_plan_s43.txt` |
| cell | 90（9×5×2），目录 `model__split43__foldK__seed0`，写进同一 `snn_*_zs_lr1e4_v1` |
| 交接 | `/share/workspace3/icassp_linzh_share/HANDOFF_FOR_LINZH.md` |
| 作业（18:10） | **4 卡 RUNNING**：71744 2C ann（eegnet 5/5 ✅，fbcnet 正在 fold4）；71746 2C atc（atcnet 2/5 ✅，刚开 fold2）；71748 2C labram（fold1 ep2）；71750 2C eegpt（fold0 ep10）。其余 8 个首链 PD（QOS 4 卡满）。SHU 0 cell |
| 进度 | 2C split43：12 个 `result.json` / 45；SHU split43：0 / 45 |
| 合并 | 等用户要求后跑 `merge_back_s43.sh` |

## 1d. AdaHGNN-MI（D-037 → D-038 → D-039 → **D-040 已提交**；zs；cell 写进同一 `snn_*_zs_lr1e4_v1`；清单 `logs/slurm/bl_adahgnn_job_ids.txt`）

学长的实验室模型 `EEG_AdaMSHyper`（`/share/home/yuan/LY/DeepHypergraph/project_V4`）按方案 T0+T1+T2 适配为 **AdaHGNN-MI**（`code/models/external/adahgnn/mi_model.py`，≈0.1 M 参数；官方文件原样保留、不跑）。两个条目：`adahgnn@mi`（主）、`adahgnn@mi_mix`（同类别 mixup 臂）。**论文行 = 两臂中验证集 macro-F1 均值更高者**（预先声明，不看测试集）。

| 作业 | 内容 | 状态（09-20 18:10） | 预期 cell |
|:---|:---|:---|---:|
| `bl2c_adahgnn` **71825 PD gpu2node → 71826** | 2C · mi + mi_mix（一卡两进程） | **未开跑**（Priority，无日志） | 20 |
| `blsh_adahgnn` **71827 PD gpu2node → 71828** | SHU · mi + mi_mix | **未开跑** | 20 |
| `bl3c_adahgnn` **71829 PD gpu3node → 71830** | 3C · mi + mi_mix | **未开跑**（等 71034 或集群空卡） | 22 |

历史（09-20）：16:49 改写版 71803–71808 提交 → 16:55 用户否决，全部 scancel（PD、0:00，从未训练），恢复官方类包装、装 `dhg`（D-038）→ 17:09 用户改指令要适配方案（D-039）→ 17:40 方案落盘 → 17:38 用户拍板 → 18:20 实现 + 21 项单测 + npz 通道顺序核对 → 提交 71825–71830（D-040）。

冒烟检查（上卡后）：`logs/slurm/bl2c_adahgnn_1-71825.err` 应为空；`outputs/experiments/_smoke/bl2c_adahgnn_mi/failures__bl2c_adahgnn_mi_smoke.json` 的 `n_cells_failed=0`。失败即整作业退出，不烧全量；修复后重投同一命令（完成 cell 跳过）。

## 2. 已完成

| 项 | 状态 | 产物 |
|:---|:---|:---|
| ANN 基线（18 条目 × 2C/SHU，各 180 cell；3C 7 模型 × 2 配方） | ✅ 2026-08~09 | `results_snapshot_2026-09-13/experiments/...`；`docs/PRIOR_RESULTS.md`；`docs/RESULTS_TRUTH.md` |
| 方案 v1 → 四位专家评审 → v2（11 处代码修订、框架改「非劣性 + 对齐交互」） | ✅ 09-18 02:30 | `docs/PLAN_spiking_eegnex_icassp2027.md` §0 |
| S-EEGNeX / S-EEGNet 代码、EA/AdaBN、能耗核算、分析脚本、13 项单测（全套 158 通过） | ✅ 09-18 | `code/models/spiking/`, `code/experiments/cross_subject_protocols.py`, `scripts/analyze_snn_round.py`, `tests/models/test_spiking.py` |
| 3C processed 数据重生成（作业 70221，33 session 全 ok） | ✅ 09-18 01:10 | `outputs/processed/wbci_shu_3c_mat_clean/` |
| GPU 冒烟（70222 / 70305） | ✅ | 产物已归档：`../_archive/eeg-mi-online_legacy_2026-09-18/outputs/experiments/shu/snn_smoke2/`（日志仍在 `logs/slurm/snn_smoke*-*.out`） |
| **代码剪枝（D-031）**：按主实验 import 闭包把跨 session 时代的 50 个模块、33 个旧 YAML、`tests/tta/`、冒烟/被取代产物移入归档；`code/` 115 → 64 个 .py | ✅ 09-18 18:45 | 验收：compileall、106 测试、CPU 微型 run、roster 检查、剪枝后启动的 71021/71024 正常 |
| **SHU 先导（split 41，120 cell，6 组 / 12 个接力作业均 COMPLETED 0:0）** | ✅ 09-18 13:10 | `outputs/experiments/shu/snn_shu_{zs_lr1e4,zs_lr1e3,ea_lr1e3,eamu_lr1e3,eaz_lr1e3}_v1/`；分析 `outputs/analysis/snn_round/shu_pilot/` |
| 先导结论 → 主实验决定（lr 1e-4；臂 eaz；主模型不变） | ✅ 09-18 17:00 | PLAN §0.5；`docs/DECISIONS.md` D-020~D-024 |
| 主实验 8 组 20 作业提交 | ✅ 09-18 17:00 | §1 |
| 目录迁移 `eeg-mi-online` → `ICASSP`、剪遗留、灵魂记忆重写、论文模板落位 | ✅ 09-18 17:30 | 本目录；`../_archive/`；`paper/` |
| **对比模型表（学长要求）**：6 个已发表 ANN 基线（带年份 2017→2025）+ 3 个 SNN 行，数字由脚本从 CSV 生成，已放进 overleaf Table 1（格式不变）；给学长的报告含 18 条目全表、先导配对、主实验进度、2025/26 候选基线 | ✅ 09-18 19:40（D-032） | `paper/main.tex` `tab:main`；`scripts/build_comparison_table.py` → `outputs/analysis/comparison_table_2026-09-18/`；`docs/reports/REPORT_comparison_table_2026-09-18.md` |
| **稿件底稿切换 + 三张队列表**：`old_reference.tex`（团队 2026 论文）前言按 2027 kit 对齐；Sec. 4.1 的两张旧表换成 2C / SHU / 3C 三张表（EEGNet / FBCNet / ATCNet / EEG-Deformer 已填；LaBraM / EEGPT / TCFormer / MDGEEG / EEGMoE 与 Ours 行 `\tbd`；FBCNet-3C 未跑 `--`） | ✅ 09-18 23:40（D-033） | `old_reference.tex`（标记区 `% >>> AUTO-GENERATED COHORT TABLES`）；`scripts/build_dataset_tables.py` → `outputs/analysis/dataset_tables_2026-09-18/{tab_2c,tab_shu,tab_3c,tab_compact_all}.tex, dataset_tables_long.csv, dataset_tables.md` |
| **四路严格审查 + 修复**（kit 格式 / 构建 / 数字 / 领域评审）：编译通过（tectonic + T1，6 页 = 旧正文 5 页 + 参考文献页），表宽 198/224/198 pt < 244.7，字体全 Type 1，脚注 9 pt，`refs.bib` `note`→`annote`（不再漏进参考文献），47 个数字独立复算一致；生成器修 9 处正确性问题并加 `--include eegnex` / `--layout compact` / `--final` | ✅ 09-19 00:20 | `docs/reports/REPORT_manuscript_format_review_2026-09-18.md`（§6 = 待学长拍板 5 项） |
| **外部基线 adapter + 全量提交（D-034）**：TCFormer / LaBraM / EEGPT / CBraMod 三队列前向通过；LaBraM 允许预训练 `lm_head`；pytest 外部套件 + 原测试回归；作业 71184–71185、71206–71225 已投 | ✅ 09-19 01:22 | `code/models/external/`；`tests/models/test_{tcformer,labram,eegpt,cbramod,external_integration}.py`；`logs/slurm/bl_job_ids.txt`；`docs/reports/REPORT_external_baselines_submit_2026-09-19.md` |

先导头条（SHU，25 人，split 41；细节 PLAN §0.5）：lr 1e-4 优于 1e-3；EA+逐通道 z-score 最好（ANN +1.4pp、SNN +0.9pp）；
D 版 −0.48pp（n.s.）、S 版 −1.38pp（p=0.04）vs EEGNeX；S 版能耗低 17×、D 版 7×；模拟前端消融 −1.82pp（p=0.03）；H1 交互 −0.5pp（n.s.）。

## 3. 下一步（按优先级）

0. **Linzh**：登录后读 `HANDOFF_FOR_LINZH.md`，预检通过再提交 seed-43 对比作业（2C+SHU，90 cell）。
1. **盯收尾 3 作业**：71034（SHU s_plif）、71213（2C tcformer 重跑）、71222（3C eegpt）。Linzh 60 cell **已合并**，挂起作业 **已 scancel**。2C 现 83 result.json（主 50 + labram/eegpt/cbramod 30 + tcformer 3）；SHU 103（主除 s_plif + 三基线 30 + tcformer 10）。2C/3C 主条目已齐，可先跑 `analyze_snn_round.py`。
   现行提交命令（重投 / 补链时照抄；完成的 cell 自动跳过）：
   `N_LINKS=2 bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_wbci2c_zs_lr1e4.yaml c2zA "seegnex@s_hplif" -- --folds-subset 0,1,2`；`… c2zB "seegnex@s_hplif" -- --folds-subset 3,4`；`N_LINKS=1 … c2zC "seegnex@hplif"`；
   `N_LINKS=3 bash … snn_wbci2c_zs_lr1e4.yaml c2z "eegnex,eegnet_official,seegnet@plif"`（71021 链）；`PARTITION=gpu3node bash … snn_wbci3c_zs_lr1e4.yaml c3z "eegnex,eegnet_official,seegnex@hplif,seegnex@s_hplif,seegnet@plif"`；
   `PARTITION=gpu3node bash … snn_shu_zs_lr1e4.yaml shz "eegnex,eegnet_official,seegnex@hplif,seegnex@s_hplif,seegnet@plif,seegnex@s_lif,seegnex@s_plif"`。**eaz 配置（`snn_*_eaz_lr1e4.yaml`）本轮不再提交**（D-035）。
1b. **盯 AdaHGNN-MI 冒烟（71825 / 71827 / 71829 上卡后）**：`.err` 为空、`_smoke/<tag>/failures__*_smoke.json` 的 `n_cells_failed=0`；失败则看 `outputs/experiments/_smoke/<tag>/` 与 `.out` 最后 50 行，修 `mi_model.py` 后重投 `N_LINKS=2 IDS_FILE=logs/slurm/bl_adahgnn_job_ids.txt NICE=50 bash scripts/slurm/submit_baselines.sh scripts/slurm/baselines_plan_adahgnn.txt`（完成 cell 跳过）。出齐后 `analyze_snn_round.py` 自动纳入 `adahgnn@mi` / `adahgnn@mi_mix`；论文行按 D-040(4) 用验证集 macro-F1 选臂。
2. **BCI IV 2a LOSO 锚点**（截止 09-20 早）：写 `code/datasets/` loader 读 `/share/workspace2/moto_imagination/eegbench_benchmark_pkl/BNCI2014001*.pkl`（2592 × 22 × 1000 @250 Hz，4 类；LH/RH 子集 = 2 类）。pkl 顶层对象是 `utils.EEGDataLoader.EEGData`（类定义在 `../Benchmark/third_party/eeg_fm_benchmark/utils/EEGDataLoader.py`），反序列化要把该目录加进 `sys.path` 或用 `pickle.Unpickler.find_class` 映射；字段与生成方式见 `../Benchmark/scripts/convert_inhouse_to_benchmark_pkl.py`（不改 Benchmark 项目）；配方同 bench；`eegnex` + `seegnex@{hplif,s_hplif}` × {zs, eaz}；≈ 5 GPU-h。做不完则写 limitation。
3. **分析脚本补全**（CPU，可随时做）：非劣性单侧 bootstrap 下界（δ=1pp）；n≤50 精确 / n=51 置换 Wilcoxon；Hodges–Lehmann + BCa CI；Bonferroni–Dunn（k=19，对照 S-EEGNeX）；Page's L（H3）；SHU 主指标 balanced accuracy；`n_cells == n_folds × n_splits` 覆盖断言；发放率被试间分布图（对齐前后）。
4. **2C 出结果 → 跑 `analyze_snn_round.py --dataset wbci_shu --run zs=… --bench …`**（本轮没有 eaz run，**不传 `--run eaz=` / `--aligned-arm`**；若脚本坚持要对齐臂，只做 zs 的 per_model / paired_tests / energy，H1 相关表留空——需要时在 `scripts/` 下加开关，不动 `code/`）→ 填未对齐块主表（18 ANN + 4 外部 + 3 SNN，逐被试配对）→ 决定摘要措辞（非劣 / 能耗）。
   Table 1 的行用 `python scripts/build_comparison_table.py --out outputs/analysis/comparison_table_<date> --snn-2c <2C per_model.csv> --snn-shu <SHU per_model.csv> --snn-3c <3C per_model.csv>` 重生成后贴进 `paper/main.tex`（† ‡ 标记随之消失）；不要手填。
4b. **外部基线**：2C+SHU 的 labram/eegpt/cbramod **已在 yuan 正式目录**（60 cell，Linzh 跑、13:00 合并）。yuan 3C 除 eegpt 外已齐；2C tcformer 因首链三进程 OOM 在 71213 单跑补。`failures__bl2c_tcformer.json` 的 10 是首链 OOM，不是最终失败。
5. **等学长拍板（写作的前提）**：论文题目与框架、进论文的实验清单、transductive 对齐写法、S/D 哪个当主模型、seed、BCI IV 2a 做不做、能耗表述（`docs/reports/REPORT_for_senior_2026-09-18.md` §6 的 7 项）；以及学长自己的 EEGNeX 改版是否注册进同一张表。**拍板前 Agent 不写论文正文、不替学长挑实验。**
   **表相关的剩余项**（`docs/reports/REPORT_manuscript_format_review_2026-09-18.md` §6）：EEGNeX 行去留（`--include eegnex`）；Ours 行按预注册（S 版，D 兜底）还是"最好者"；
   三张双行表（~0.8 页）vs `--layout compact` 单张 `table*`（~0.3 页）。3C 补跑与 EEGMoE→CBraMod 已按 D-034 执行。
6. **写作**（学长定题后，09-21 起）：底稿 = 根目录 `old_reference.tex`（D-033）。三张队列表的刷新命令（CPU，登录节点）：
   `python scripts/build_dataset_tables.py --out outputs/analysis/dataset_tables_<date> --splice old_reference.tex [--snn-2c/--snn-shu/--snn-3c <analyze_snn_round 的 per_model.csv>] [--ours seegnex@s_hplif] [--extra 2C=<新基线 per_model.csv> --alias mdgeeg=<csv 里的键>]`；
   另一窗口的 MDGEEG / EEGMoE / EEGPT / LaBraM / TCFormer 结果只要落成 bench 或 analyze 格式的 `per_model.csv` 就能 `--extra` 接入。
   编译：Overleaf（上传 `old_reference.tex` + `paper/spconf.sty` + `paper/IEEEbib.bst` + `paper/refs.bib` + 2026 论文的 `strings.bib` 与 `main.pdf`/`visual.pdf` 图）。
   `paper/main.tex` 只剩参考价值（方案 v2 分节注释、Table 1、limitation 清单），合并后可删。
7. 有余力：2C 主 2×2 对照补 2 个训练 seed（≈ 30–60 GPU-h）；`s_hplif_causal` / LIF-before-pool 单因子消融（SHU）。

## 4. 已知风险

- 集群满载：09-18 17:30 仅 1/8 组在跑，其余 PENDING（Priority / Resources）。若 09-19 早仍未启动，考虑把 2C 两臂拆成更多小组并两分区都排。
- 2C 若 S 版低于 EEGNeX 超过 1pp：D 版作兜底主模型（H2 改写为「S 在 D 的 1pp 内、SOPs 少 ~4×」）；S 版仍以能耗 + 流式立论。
- H1 在 2C 也不成立：按观察报告（先导已如此），故事仍成立（非劣 + 能耗 + 对齐对两者增益相当）。
- 单训练 seed：写 limitation；有预算就补。
- 兼容软链接 `SYX/eeg-mi-online -> ICASSP` 与旧路径：作业跑完前不删。审计确认：20 个作业的 `WorkDir`/`Command`/`StdOut` 都经软链接解析；`bench_gpu.sbatch` 在提交时已被 Slurm 冻结拷贝（改它不影响排队作业），但 `_common.sh`、`code/**`、`snn_*.yaml` 是作业**启动时**才读——排队期间改它们会直接影响 71019–71037。
- 单划分 vs 双划分的 cell 命名（D-030）：`snn_shu_{zs_lr1e3,ea_lr1e3,eamu_lr1e3,eaz_lr1e3}_v1` 仍是 `<model>__fold<k>__seed0`；不要用双划分配置续跑进这些目录。
- **作业结束后的第二轮剪枝（闭包内文件，现在不能动）**：`code/runners.py` 删 7 个死 runner（`run_phase0_drift`, `run_phase1_baseline`, `run_phase2a_multisource`, `run_phase2b_alignment`, `run_phase2c_prototype_drift`, `run_phase3_tta`, `run_cross_session_e2e`）与 `PHASE_RUNNERS` 对应项；`code/run.py` 删 `--all`/`DEFAULT_PHASES`（配置已归档）与 `--summarize`（`summaries.summarize` 已归档）；`code/models/registry.py` 删弃用模型 `cap_eegnet / eegnet(自研) / atcnet_adv / s4erp_adv / dualcd_s4_pos` 及其 yaml、`code/models/{cap_eegnet,eegnet}.py`、`tests/foundation` 里引用 `dualcd_s4_pos` 的用例；`code/training/trainer.py` 瘦身为只留 `predict`；`code/experiments/session_protocols.py` 只留 `TrainSpec` / `evaluate_predictions`。每一步后跑 `pytest tests -q` + CPU 微型 run（命令见 D-031）。
- 作业结束后的代码小修（现在不动，`code/` 与 `scripts/slurm/` 在排队期间冻结）：
  - `scripts/slurm/bench_gpu.sbatch:29-32` 的 `BASH_SOURCE` 回退在 Slurm 下永远不生效（脚本被拷到 spool），应改为 `PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR:-.}" && pwd -P)"` 并显式检查 `_common.sh` 存在；`_common.sh:5-6` 用 `pwd -P`。
  - `cross_subject_protocols.py:975` 加 legacy 单划分目录回退；`:866-867` 的 `checkpoints`/`predictions` 改存相对路径。
  - 遗留默认值 `code/experiments/session_tta.py:354,359,703,736,742`、`prototype_drift_summarize.py:578,593`（`4_experiments/`、`1_session_drift/`）改为 `outputs/experiments/…`；5 个遗留 phase 配置的 `readable_dir`/输入表同理；`scripts/summarize_cross_subject.py:33` 的 `5_papers/tables` 示例。
  - 注释/docstring 里的旧路径：`foundation_cross_subject.yaml:16-17`、`shu_foundation_cross_subject.yaml:14`（→ `docs/CROSS_SUBJECT_PROTOCOL_{RESEARCH,MEMO}.md`）、`adv_821recipe_3c.yaml:9`、`paper_baseline_3c_821.yaml:42`、`shu_cross_subject_5fold.yaml:53`、`phase3_tta*.yaml` 头注；`code/utils/paths.py:3`、`code/datasets/{manifest,splits}.py:3`、`code/models/{eegnet,deepconvnet}.py`、`code/models/atcnet/adapter.py:7`、`code/models/eeg_foundation/adapter.py:14`、`future_online_adaptation.yaml:7` 引用的旧 `.cursor/rules` 文件名。
- **Git 护栏**：剪遗留的 commit 落地前不要 `git checkout -- .` / `git stash` / `git restore .` / `git clean`（会复活 291 个已归档文件，见 `AGENTS.md` §3.6）。
- 遗留配置 `phase2c_prototype_drift / shu_phase2c_prototype_drift / phase3_tta / shu_phase3_tta / phase3_tta_full_a0` 引用已归档的 `1_session_drift/ 4_experiments/`，跑它们会失败或在仓库根生成杂目录——不要跑（`code/configs/experiments/README.md` 已注明）。
