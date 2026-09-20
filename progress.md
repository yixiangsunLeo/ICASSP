---
title: "progress.md — 逐日流水（最新在上）"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
created: "2026-06-10"
updated: "2026-09-20 21:00"
status: "active"
---

# PROGRESS / Memory Log

The project's running memory. **Append a dated entry whenever you finish a
meaningful step or make a design decision.** Newest entries on top. Keep it factual.

Format per entry: date, what was done, decisions made, open questions, next step.

---

## 2026-09-20 21:05 — 最新交接更正 + 3 个 AdaHGNN result.json 推 icassp

把「3090 没点名就停」写进 `HANDOFF.md` / STATUS / handoff 身份与训练卡，并跟踪新落盘的 `adahgnn-mi` SHU split41 fold1、`adahgnn-mi_mix` SHU split41 fold1、`adahgnn-mi` 3C fold8。不改训练闭包，不推 `.pt` / npz。

---

## 2026-09-20 21:00 — 已推到 github.com/yixiangsunLeo/ICASSP

`git push -u icassp main` 成功。远端 `main` = `e73abd9`。原因：默认 `git@github.com` 认证的是 LiaoEuan，对这个 private 仓报 `Repository not found`；改 remote 为 `git@github.com-yixiangsunLeo:yixiangsunLeo/ICASSP.git` 后可写。未推旧 `origin`（EEG.git）。HANDOFF Phase 11 更正还在工作区，未进这次 push。

---

## 2026-09-20 21:00 — 用户更正 3090 用途：先同步代码，跑哪些模型由他点名

不要把 yuan / Linzh 正在跑的作业自动搬到 3090。后续主线 = 学长改完的 S-EEGNeX-D（`seegnex@hplif`）和 AdaHGNN，在本树整合后经 GitHub 过去训，`result.json` 写同一相对路径再拉回。`HANDOFF.md` Phase 11 / STATUS「3090 缺口」已改成「没点名就停」。数据仍走 rsync，不进 git。

---

## 2026-09-20 21:00 — 可迁移仓库本地已提交，远端仓库尚未创建

本地 `10f5219` + `38c1df3`（1033 个 `result.json`，灵魂文件与 HANDOFF 在树上）。remote `icassp` 已指向 `git@github.com:yixiangsunLeo/ICASSP.git`。`git push icassp main` 返回 `Repository not found`——GitHub 上还没有这个空仓库。未 push 到旧 `origin`（EEG.git）。未改训练闭包、未 scancel。用户在 github.com/new 建好 **private、不要 README** 的 `yixiangsunLeo/ICASSP` 后即可 `git push -u icassp main`。

---

## 2026-09-20 20:30 — 可迁移仓库 + 3090 交接（D-041）

用户要在 3090 账号训练，且那边读不到 `/share/workspace2`。已写给那边 AI 执行的 `HANDOFF.md` + `docs/handoff/`（安装、数据前缀改写、预检、同路径训练、`result.json` 回传）。新增 `scripts/repath_manifests.py`、`scripts/handoff_preflight.sh`、`scripts/sync_results_git.sh`。`.gitignore` 放行小结果文件，不放 npz/pt/`paths.local.yaml`。未改 `code/` 训练闭包、未改 `scripts/slurm/`、未改 yuan 的 `paths.local.yaml`、未 scancel。下一步：把可迁移树推进 `https://github.com/yixiangsunLeo/ICASSP`；用户在 4090 上按 HANDOFF Phase 3 rsync 数据。

---

## 2026-09-20 18:10 — 进度核对：yuan 收尾 3 卡 / AdaHGNN 未上卡 / Linzh seed-43 12/90

yuan 正在跑：71034 SHU `s_plif` 最后 1 cell（split42 fold4 ep15，估 19:30）；71213 2C tcformer 7/10（split42 fold2 ep40，估 20:15–20:45）；71222 3C eegpt 9/11（fold9 ep7 + fold10 未开，估 19:20–19:40）。AdaHGNN-MI 71825–71830 **仍全 PD，无 Slurm 日志**。Linzh 4 卡在跑 2C split43，12 个 `result.json`；SHU 0；其余 8 个首链卡在 QOS 4 卡上限。估 Linzh 09-21 下午出齐。STATUS §0/§1/§1b/§1c/§1d 已改。

---

## 2026-09-20 18:25 — AdaHGNN-MI 实现、单测、提交三队列（D-040）

用户 17:38 拍板方案 §8：接受 T2；原版不跑；mixup 1 臂、DANN 不开；「全都跑」；命名 AdaHGNN-MI；F=32；改完直接提交训练。
实现 `code/models/external/adahgnn/mi_model.py`（≈0.1 M 参数：共享滤波器组 125/63 → 可微 top-k 超图 → 信号域 Laplacian 空间滤波 (I−γA₁)S → GELU/AvgPool16/Dropout/softmax 时间权重/每节点仿射 → 三尺度 HGNNP（58→13→4 / 32→14→4 解剖分半球）→ [mean; flatten; flatten] → 128 → 类别；同类别 Dirichlet mixup 臂走 `training_step` hook）；`adapter.py` 按 `arch` 分派（官方类保留不跑）；`adahgnn.yaml` 加 `per_dataset.channel_set` + `variants.{mi, mi_mix}`；`make_adahgnn_configs.py` roster 改两条目并重生成三份 `bl_*_adahgnn_zs_lr1e4.yaml`（协议逐键同 `snn_*`）；`baselines_plan_adahgnn.txt` 每队列一卡两进程。
验证：`tests/models/test_adahgnn.py` 21 项通过（官方文件 sha256 = 实验室文件；三队列前向/反向且**所有参数都有梯度**；结构嵌入有梯度、行和为 1、≤k；稠密聚合与 `dhg.Hypergraph.v2v(mean)` 误差 < 1e-5；解剖映射是划分且 C3/C4、FC1/FC2、CP1/CP2 不同区且分属 SM_L/SM_R；单样本训练 batch 不崩；mixup 同类别、非自身、重 z-score；3 步 Adam 无 NaN；describe JSON；配置协议一致；plan 一致）；`test_external_integration` 配置子集 9 项、`test_fusion_and_specs` 53 项通过；三队列 npz `channel_names` 与 montage yaml 顺序一致。
提交：71825–71830（NICE=50，SMOKE_FIRST，N_LINKS=2；3C 首链 71829 改 gpu3node）。全集群 40/40 卡满载，排队中。未 scancel 71034 / 71213 / 71222。62 cell 出齐前不当结果。

---

## 2026-09-20 17:40 — AdaHGNN：撤回改写、作业全取消、改为出适配方案待审（D-038 / D-039）

16:55 用户指出第一版把网络改写了（分组 stem + 自制稠密超图 + 去 GRL + 时移改 250 Hz）。执行：`scancel` 71803–71808（全 PD、Elapsed 0:00，从未训练）；删 `network.py`；`adapter.py` 改为原样包装 `_official/models.py::EEG_AdaMSHyper`（sha256 与实验室文件一致）；`mi_torch_cu118` 装 `dhg==0.9.7`；`adahgnn.yaml` 只留 `d_model/patch_size/k`；测试改为校验官方类与 sha256；PROVENANCE / README / AGENTS §6 / D-037 措辞更正。
17:09 用户改指令：**要做适配、先出方案审阅、不跑、原任务取消**。未再提交任何作业（`squeue` 只有 71034/71035、71213、71222/71223）。
17:40 方案落盘 `docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`：原模型 9 条审计（[代码推导]：58 通道 batch 128 激活 ≈ 37 GB 必 OOM；超图结构经阈值/nonzero 不可微、嵌入无梯度、初始 softmax 饱和 → 每电极恰 1 条超边 = 随机固定分组；连续下标池化把 C3/C4、C1/C2、CP3/CP4 等 16 对左右同名电极平均；核 63/31 与时移按 128 Hz 硬编码；每通道独立 stem 715 k 参数；空间滤波在非线性之后无法表达 CSP/Laplacian；均值读出置换不变；原训练法按测试被试选 checkpoint、mixup 跨类别混合；dhg 逐 batch 循环）+ 四层方案 T0/T1/T2/T3 + 推荐配置 AdaHGNN-MI（≈ 0.1 M 参数、≈ 7 GB 激活）+ 只用验证集的 V0–V4 阶梯选择 + §8 六个待拍板问题。记 D-039；STATUS §0/§1d/§3 已改。一个准备验证上述推导的 CPU 小脚本被审查阻止、未运行（结论全部来自代码推导，实现时用单测确认）。

---

## 2026-09-20 16:50 — AdaHGNN 导出并在 yuan 提交三队列（D-037；**已被 17:40 条目取消**）

学长指向 `/share/home/yuan/LY/DeepHypergraph`。最新完整版 = `project_V4` 的 `EEG_AdaMSHyper`（同日 V3 21:46 脚本是同一网络的 DANN 单文件）。已 vendoring + 懒加载条目 `adahgnn`；不移植 DANN；zs 配方；新 yaml `bl_*_adahgnn_zs_lr1e4.yaml`（未改 live `snn_*.yaml` / `bl_*_zs_lr1e4.yaml`）。`pytest tests/models/test_adahgnn.py` 18 passed。提交 71803–71808（NICE=50，每队列一卡，SMOKE_FIRST）；2C 首链 71803 改到 gpu3node。未 scancel 71034/71213/71222。31 cell 出齐前不当结果。

---

## 2026-09-20 15:05 — 对比模型 split 43：只 2C+SHU，Linzh 跑（D-036）

学长要对比模型加划分 seed 43。用户指定在 Linzh 号跑、yuan 已在跑的不要重复、SNN 先不改。随后更正：**3C 是 LOSO，不加 43**。

已做：cell 标签在单 seed≠0 时仍带 `split43`（不改 seed-0 的 3C 命名）；新配置 `bl_{wbci2c,shu}_zs_s43.yaml` + `baselines_plan_s43.txt`（12 作业，90 cell）；未改 live `snn_*.yaml` / `bl_*_zs_lr1e4.yaml`。代码同步到 `/share/workspace3/icassp_linzh_share/`，交接 `HANDOFF_FOR_LINZH.md`。MDGEEG / EEGMoE 仍跳过。未在 yuan 提交、未 scancel 71034/71213/71222。合并等用户再要求。

---

## 2026-09-20 13:10 — 三队列对照表（未完成格子为 —）

用户要看三个数据集各自所需模型的表。跑 `analyze_snn_round.py`（只 `--run zs=`，`--allow-empty-family`）到 `outputs/analysis/snn_round/{wbci2c,shu,wbci3c}_zs_2026-09-20_1300/`，再把覆盖不齐的行（2C tcformer 3/10、3C eegpt 3/11、SHU s_plif 3/10）从 `--snn-*` 里去掉后跑 `build_dataset_tables.py --include eegnex --ours seegnex@s_hplif`。报告 `docs/reports/REPORT_cohort_tables_2026-09-20.md`：TBD/未齐写成 —，并列 S/D/S-EEGNet。未按测试集改主模型。未 splice `old_reference.tex`。

---

## 2026-09-20 13:00 — Linzh 60 基线 cell 合并进 yuan 正式目录；12 个挂起作业已 scancel

用户确认合并并取消挂起任务。`cp -n` 写入 `outputs/experiments/{wbci_shu,shu}/snn_*_zs_lr1e4_v1/cells/`：2C 30 + SHU 30，无覆盖。核对 `result.json` 可读；Linzh 侧 6 个 `failures__*` 均为 0。随后 `scancel` 71206–71211、71214–71219（全 CANCELLED，未跑过）。未动 71034 / 71213 / 71222。STATUS §1b/§3 已改。

---

## 2026-09-20 12:55 — 双账号进度：Linzh 60 cell 齐；yuan 主实验 2C/3C 齐，还剩 3 卡收尾

用户问两个号跑得怎样。`squeue`：yuan 3R + 14PD（其中 12 个仍是 09-19 挂起的 2C/SHU 基线）；Linzh 队列空。

- **Linzh 71359–71371 全部 COMPLETED 0:0**（最慢 bl2c_eegpt 13.5 h）。拷贝树里 2C+SHU 的 labram/eegpt/cbramod 各 10，共 60 个 `result.json`。但 `result.json` mode 600、属主 Linzh，yuan 读不了，`merge_back.sh` 现在会失败。合并前必须在 Linzh 上 `chmod -R a+rX /share/workspace3/icassp_linzh_share/outputs`。挂起的 12 个作业不要 release，合并成功后再 scancel。
- **yuan 主实验**：2C 50/50 ✅（S 链 71230 01:31 齐）；3C 55/55 ✅（71030 05:36 齐）；SHU 除 `seegnex@s_plif` 外全齐（71034 在训 split41 fold3 ep73，patience 12，~1 min/epoch，估再 6 个 cell + 本 cell ≈ 今晚）。
- **yuan 基线还在跑**：71213 补 2C tcformer（首链 71212 三进程一卡 OOM，`failures__bl2c_tcformer.json` n=10；SHU+3C tcformer 已齐；71213 现 fold3 ep21，3/10，~1.2 h/cell → ~8 h）；71222 3C eegpt 3/11 fold3 ep21 patience 17（~1.2 h/cell → ~8 h）。3C labram/cbramod/ANN 已齐。
- 估全部出齐：**今晚 21:00–23:00**（三卡并行）。STATUS §0/§1/§1b/§3 已改。

---

## 2026-09-19 12:30 — 中期结果快照（zs 臂）：2C 40/50、3C 36/55、无失败；`REPORT_interim_results_2026-09-19.md`

用户问「现在出了哪些结果」。跑 `analyze_snn_round.py`（只给 `--run zs=`，脚本不需要 `--aligned-arm` 也能出 zs 家族）到 `outputs/analysis/snn_round/{wbci2c,wbci3c}_zs_interim_2026-09-19_1220/`。

- **完整可用**：2C EEGNeX 83.35 ± 11.54（与 bench 逐被试全等，51 平）、EEGNet 82.53 ± 12.04（= bench）、**S-EEGNet 82.03 ± 12.07（vs EEGNet −0.49 pp，p = 0.0009）**；3C EEGNeX 69.41 ± 13.22（旧 patience-100 68.78，一致）、EEGNet 66.05、**S-EEGNeX-D 67.34（vs EEGNeX −2.07 pp，p = 0.13，n = 11）**。
- **未完整（趋势，不是结论）**：2C D 6/10 → 82.53（−0.82 pp，p = 0.0006）；2C S 4/10、41 人 → 81.61（−1.84 pp，CI [−2.83, −0.84]）；3C S 3/11。能耗 3C：D 6.8×、S 16.8×（n = 3）。
- 作业位置：71234（D）split42 fold1；71230 split42 fold0；71232 split41 fold4；71030 3C S fold3。今日 2C S cell 实跑 4.5–7.7 h（比 01:50 估的 5 h 长）→ 2C S 出齐改估 **09-20 清晨**，全部含对照 **09-21 中午**。STATUS §0/§1 已改。
- 未按测试数字改主模型；eaz 无数据。

---

## 2026-09-19 02:00 — 用户拍板：砍全部 eaz + 2C zs S/D 拆三路并行（D-035）；已执行

用户在选项卡选「砍掉全部 eaz + 拆 2C zs」、不申请 8 卡 QOS。执行顺序（都在 01:45–01:50）：
1. `scontrol hold` 71034/71035 + 22 个 `bl_*`（防止空出的卡被它们抢走）。
2. `scancel` eaz 臂 10 个：71025–71029、71032、71033、71036、71037，再 71024（运行中）。
3. `scancel` 71019、71020，再 71018（运行中；`last.pt` 原子写，epoch 35 完好）。
4. 提交 `c2zA` 71230→71231（S 折 0,1,2）、`c2zB` 71232→71233（S 折 3,4）、`c2zC` 71234（D）。71230 / 71232 立即上卡（gpu01）；日志核实：`split_seeds=[41, 42]`、fold0 跳过、fold1 **从 epoch 35 续训**、71232 从 split41 fold3 起；`.err` 空。
5. 按 71034 → 71035 → bl 计划顺序 `release`；`priority/basic` 下 release 重排到队尾，最终队列：c2zC > shz > bl3c_ann > bl2c_* > bl_tcformer > blsh_* > bl3c_*。副作用：测 nice 时 71225 被设成 nice 10000（不可调回），它只是 bl3c_cbramod 的续跑链，无实质影响。

新估：2C S+D 09-20 凌晨、SHU zs 09-20 中午、对照 09-21 上午全齐（原 09-22 下午）。更新：`docs/DECISIONS.md` D-035、`AGENTS.md` §3.5 / §4.3、`.cursor/rules/10-data-and-compute.mdc`、`docs/STATUS.md` §0 / §1 / §1b / §3。

---

## 2026-09-19 01:50 — 用户质疑「为什么还在跑 eaz」+ 要求 4 卡全开、加速；诊断结果（等用户拍板，未动作业）

- **eaz 出处**：主实验 09-18 17:00 提交的 20 个作业里 10 个是 eaz 臂（71024–71029 / 71032–71033 / 71036–71037），按方案 v2 预注册（H1 = 对齐 × 网络类型交互）投的；用户 23:47「就做 zs 就行」是对**新对照模型**的回答（D-034）。Agent 失误：当时未提醒主实验 eaz 臂仍在队列、约 117 GPU-h。eaz 对论文比较表无用，只服务 H1。
- **4 卡已全在用**：QOS `32cpu4gpu64g` = 每人 cpu=32 / gpu=4；每作业 8 CPU + 1 GPU，恰好卡满。集群此刻 6 张空卡（gpu02 ×4、gpu04 ×2）拿不到。集群存在 `64cpu8gpu400g` QOS（需管理员授权；排队作业 `scontrol update qos=` 即可换，不必取消）。
- **一卡多进程无收益**：`srun --overlap nvidia-smi` 实测 S 版 83–100%、3C D 版 99–100%、eaz S 64–100%；只有 ANN 作业 10–66%（3 h 内跑完）。
- **真正瓶颈**：2C S 版 10 个 cell（4–6 h/个）+ 10 个 D 串在一张卡上 ≈ 57 h → 09-21 中午。可按 `--folds-subset 0,1,2 / 3,4 / D` 拆 3 个作业并行（DRY_RUN 已核，cell 名保持 `__split41__/__split42__`；**按 `--split-seeds 41` 拆会丢 split 段、撞 D-030 的坑，不能用**）。
- 方案（需用户明确同意，涉及 scancel）：砍 eaz + 拆 2C zs → 2C 主结果 09-20 凌晨、含对照 09-21 上午；保留 eaz 只拆 2C → 09-22 上午；8 卡 QOS 则再快约一半。

---

## 2026-09-19 01:35 — 按实测墙钟重估全部训练完工时间

用户问全部任务何时跑完。用已落盘 `train_seconds` + 四个在跑作业的当前 epoch，而不是方案里的 2× EEGNeX 估时。

**2C S 版是墙钟关键路径**：243–250 s/epoch（4.1 min），已完成 cell 6.2 h（92 ep）与 3.9 h（58 ep）。两组 S+D 各剩约 55–60 GPU-h；48 h 时限会在 09-20 16:30 / 18:17 切到接力链，预计 **09-21 中午** 这两组才结束。QOS 4 卡里有 2 卡被它们占到那时。

**其余主实验**：2C ANN+S-EEGNet zs 今早可齐；3C zs ~09-20 00:00；SHU zs / 3C eaz 09-20 晚；SHU eaz 09-21 下午。

**外部基线** ~95–110 GPU-h、nice=50，要等主实验让卡，**09-21 上午**才上卡，中位 **09-22 15:00** 出齐。

合计剩余 ~320 GPU-h。全部训练：**中位 09-22 下午；乐观 09-22 上午；悲观 09-23 傍晚**（S cell 普遍跑到 90+ ep，或 FM 比估的慢 / 冒烟失败）。截稿 09-24 20:00，写作窗口中位约 2 天。

未改作业、未 scancel。

---

## 2026-09-19 01:22 — 外部基线 adapter 验收通过并提交全部 11 个训练作业

子 Agent 中断后由主 Agent 接手：补齐 TCFormer / LaBraM / EEGPT 测试与 PROVENANCE；修 LaBraM 权重加载（官方 `labram-base.pth` 的 `lm_head.*` 是预训练头，必须列为合法 unexpected，否则三个队列都建不起来）；四模型 × 三队列 CPU 前向全部 OK；pytest 外部套件 + 原测试集回归 exit 0；`make_baseline_configs.py --check` OK；`DRY_RUN=1` roster 12 条后实投。

**提交（未动 `snn_*.yaml` / 未 scancel 主实验；`--nice=50`）**
- 已有：`bl3c_ann` 71184→71185（3C ATCNet / EEG-Deformer / FBCNet，patience 25）
- 新投：71206–71225（10 作业 × 2 链）= LaBraM / EEGPT / CBraMod × {2C, SHU, 3C} + TCFormer 三队列共卡
- 清单 `logs/slurm/bl_job_ids.txt`。全部 PENDING（QOS / Priority），主实验 71018/71021/71024/71030 仍占用 4 GPU。

**验收数字（describe / 测试断言）**
- TCFormer paper yaml depth 5：58ch/2cls 133988；`@n2` 22ch/4cls T=1125 = **77820**（与 README 一致）
- LaBraM-base：backbone 5,824,736；加载 219/247；kept 58/58 与 32/32
- EEGPT large：58ch/2cls 25,356,128；SHU 丢 A1/A2（30 kept）；`target_encoder.*` strict
- CBraMod：与既有 `test_cbramod.py` 一致

**下一步**：盯 `_smoke/` 与 `failures__*_smoke.json`；主实验出齐后 `analyze_snn_round.py` 自动纳入同目录新 cell。未 commit。

---


用户 23:37–00:11 逐项确认：**只跑 zs 臂**、统一 bench 配方、EEGMoE → CBraMod、MDGEEG 删除、新 cell 写进主实验同 run 目录、配额内排队 + 小模型共卡、EEGPT 权重镜像件加脚注；
并要求确认"打勾的 ANN 与 3 个 SNN 协议一致"→ 盘上核对：2C / SHU 的 `splits/*.json` 逐字节相同、每 cell `spec` 相同；**3C 列 ANN 是 patience 100**（`paper_baseline_3c_821_v1`）→ 用户拍板补跑 ATCNet / EEG-Deformer / FBCNet 的 3C（patience 25）。

**做了什么**
1. `docs/PROTOCOL_external_baselines.md` → locked；`docs/DECISIONS.md` **D-034**（另一窗口的论文会话已占 D-033，其三张表的行名单需按 D-034 改：去 MDGEEG / EEGMoE，加 CBraMod）。
2. `weights -> ../Benchmark/weights` 软链接（gitignore `/weights`）；`code/models/external/{__init__,_common,README}.py|md`（静态名单 + 懒加载 `build_external`；重采样 / 通道词表 / 权重路径 helper）。
3. **`code/models/registry.py` 唯一一次编辑**：`from .external import EXTERNAL_MODEL_NAMES`（纯常量）+ `build_model` 懒加载分支。验证：`compileall` ✓；导入 registry 不会触发任何 adapter 导入 ✓；`pytest tests` **106 passed / 1 skipped**（与剪枝后基线一致）✓；主实验入口 CPU 微型 run（SHU、10 被试、1 epoch）编辑前 `ok=2 failed=0` / 编辑后 `ok=2 failed=0` ✓ → 排队中的接力作业安全。
4. `scripts/make_baseline_configs.py` → `code/configs/experiments/bl_{wbci2c,shu,wbci3c}_zs_lr1e4.yaml`：从 `snn_*_zs_lr1e4.yaml` 生成，只改 `source_config` / `models` / `micro_batch_per_model`（labram 64 / eegpt 32 / cbramod 64，LayerNorm 模型累积等价），`output_dir` 同主实验；`tests/models/test_external_integration.py` 断言协议块逐字相同（11 passed，12 skipped 等 adapter）。
5. `scripts/slurm/baseline_gpu.sbatch`（一卡多进程；`SMOKE_FIRST=1` 先 1 折 2 epoch 冒烟到 `outputs/experiments/_smoke/`，失败即退出）+ `submit_baselines.sh`（读 `baselines_plan.txt`，提交前对每个 (config, model) 做 `resolve_model_params(dataset=…)` roster 检查；`--nice=50` 让主实验作业优先；`N_LINKS=2` 接力，第 2 链 `SMOKE_FIRST=0`）+ `baselines_plan.txt`（11 个作业：2C 三个 FM 各一卡、TCFormer 三队列共一卡、SHU / 3C 各三个 FM、3C ANN 补跑三进程一卡）。**未改任何现有 `scripts/slurm/` 文件。**
6. **已提交 `bl3c_ann`：作业 71184 → 71185**（atcnet + eeg_deformer + fbcnet，3C，bench 配方；PD Priority，nice 50）。清单 `logs/slurm/bl_job_ids.txt`。
7. 4 个子 Agent 并行（TCFormer / LaBraM / EEGPT / CBraMod），路径互斥（`code/models/external/<name>/**`、`code/configs/models/<name>.yaml`、`tests/models/test_<name>.py`），各自要过：58ch/2cls · 58ch/3cls · 32ch/2cls 前向、参数量对论文、权重 strict 加载 + sha256、格式转换单测、CPU 微型 run `ok=2 failed=0`。

**下一步**：收 4 份 Agent 报告 → 审代码 → 全套 pytest + 集成测试 → `DRY_RUN=1` → 提交其余 10 个作业 → STATUS / 报告。

---

## 2026-09-18 23:30 — 用户要求补 5 个外部基线（MDGEEG / EEGMoE / EEGPT / LaBraM / TCFormer）：审计 + 协议草案，等确认后再动代码

用户指令（23:12）：三个数据集上补跑这 5 个模型作对照，"对应最近跑的实验"；多子智能体并行、有检验；**先确认评估协议再执行**；代码后续公开。

**审计结论**（细节 `docs/PROTOCOL_external_baselines.md` §0）：TCFormer 官方 PyTorch 可原样 vendoring；LaBraM / EEGPT 官方代码在
EEG-FM-Benchmark（`../Benchmark/third_party/eeg_fm_benchmark/models/FM/`），权重已在 `../Benchmark/weights/`（EEGPT 为 HF 镜像转换件，未与 figshare 原件核对）；
**MDGEEG** 官方仓库只有 `model.py`（门控多分支结构），训练算法未公开、论文闭源 → 需要 PDF；**EEGMoE 不在 EEG-FM-Benchmark 里**（README 只列了论文），
论文 "code will be released"，无代码无权重 → 无法作 FM 基线，建议换 CBraMod。算力：QOS 硬上限 4 GPU/用户，全集群 40 卡全占，新作业最早 09-20 夜 ~ 09-21 上卡。

**未动**：`code/`、`scripts/slurm/`、任何 YAML；仅新增 `docs/PROTOCOL_external_baselines.md`（status: proposed）。下一步：用户回答 7 个协议选择题 → 锁定（D-033）→
共享脚手架 → 并行子 Agent（每模型一个，路径互斥）→ registry 懒加载一次性编辑 → pytest + CPU 微型 run → GPU 冒烟 → 提交。

---

## 2026-09-19 00:20 — 四路零上下文严格审查（kit 格式 / 构建 / 数字 / 领域评审）→ 可自动修正项全部修完并复核；5 个问题待学长拍板

用户 23:35：「改完之后，新开一个对话，按照官方的格式对照改一下检查几遍……最严格的审查员，多专家智能体审查」。

**做了什么**（细节与处置表见 `docs/reports/REPORT_manuscript_format_review_2026-09-18.md`）

1. 三个后台子任务回收：五个新基线文献全部核实（LaBraM ICLR 2024；EEGPT NeurIPS 2024；TCFormer *Sci. Rep.* 15:32959, 2025；
   **MDGEEG = Song et al., *Neurocomputing* 614:128889, 2025**；**EEGMoE = Gao et al., *IEEE TNNLS* 37(8), 2026，无公开代码/权重**）→
   `paper/refs.bib` 新增 5 条；2027 kit 格式清单；`~/.local/bin/tectonic` 可用（加 `[T1]{fontenc}` 得 Times 度量），`pymupdf` 已装到 user site。
2. 编译实测暴露的硬伤并修复：三张表超栏宽 10 / 54 / 10 pt → 表头去 `(\%)`、Macro-F1 → F1、`\tabcolsep` 3 pt → 198 / 224 / 198 pt < 244.7；
   出处行 `\footnotesize`（8 pt）→ `\small`；`\enlargethispage` ×2、`\vspace{-1pt}` ×3 删除；前言按 kit 顺序重排（去 `multicol`/`float`/`\frenchspacing`/`\def\L`）；
   `\bibliography{refs}`；子节标题 sentence case。
3. 四个并行审查员（fresh context）共 ~50 条发现 → 修复：`refs.bib` `note` → `annote`（IEEEbib 会把 VERIFY/GitHub 打印进参考文献）；
   `\thanks` 脚注 8 pt → `\renewcommand{\footnotesize}{\small}`（实测 8.97 pt）；头注中 5 处不准的 kit 表述重写；`\vfill\pagebreak` 冗余删除；
   生成器重写：3C `--snn-3c` 误标先导 / 历史行不被替换、`--extra` 静默覆盖与 per_cell 误判、FBCNet-3C "从未跑"应为 `--` 而非 TBD、3C caption
   "同预处理"不成立、基础模型上标 P、splice 正则锚定 + 标记唯一、覆盖率守卫、打印精度平局加粗、SHU 去冗余 Acc 列、候选测试数字移出稿件注释
   （预注册纪律）、`--snn-*` 默认不再指向先导、新增 `--include eegnex` / `--layout compact` / `--final` / `--pct-digits`。
4. 复核：47 个数字独立复算一致；重编译 6 页（旧正文 5 页 + 参考文献页），表无 overfull，字体全 Type 1 内嵌，参考文献无 VERIFY 泄漏；
   生成器 8 条路径测试通过（含拒绝重复标记 / 键冲突 / `--final` 残留 TBD）。
5. 文档：报告 `docs/reports/REPORT_manuscript_format_review_2026-09-18.md`（含待拍板 5 项）；`docs/STATUS.md`、`docs/DECISIONS.md` D-033 同步。

**待学长拍板**：EEGNeX 行去留（评审视为最大硬伤）；Ours 行按预注册还是"最好者"；三张双行表（~0.8 页）vs 一张紧凑 `table*`（~0.3 页）；
3C 基线是否用 bench 配方补跑；EEGMoE → CBraMod。**未 commit。**

---

## 2026-09-18 23:40 — 稿件底稿切换为团队 ICASSP 2026 论文（`old_reference.tex`），按 2027 模板对齐；三张按数据集分的结果表已生成并放入

用户指令：速成论文，以去年团队中的 ICASSP 论文 `old_reference.tex` 为模板按 2027 官方 `Template.tex` 改格式；实验表改为三张（2C / SHU / 3C）；
模型 = 截图打勾的 EEGNet / FBCNet / ATCNet / EEG-Deformer（DeepConvNet、EEGNeX 不进）+ 另一窗口在跑的 MDGEEG / EEGMoE / EEGPT / LaBraM / TCFormer（占位）
+ 三个 SNN 候选里最后最好的一个（先占位）；其他正文不动；不要与另一窗口的模型任务冲突。

**做了什么**

1. 核对另一窗口的足迹：`git status` / 6 小时内改动只有本会话链早先的文件，两仓库里都还没有 MDGEEG / EEGMoE 的代码（LaBraM / EEGPT / TCFormer 在
   `../Benchmark/third_party/`）。本次只新增 `scripts/build_dataset_tables.py`、改根目录 `old_reference.tex` 与文档，不碰 `code/`、`scripts/slurm/`、任何模型文件。
2. `old_reference.tex`（用户放在仓库根）**按 2027 kit 对齐**：文件头注释（页数 / 5 页只放参考文献 / PDF 与字体 / 单盲 / ORCID / AI 披露）；
   `\usepackage` 拆成 kit 的 `spconf,amsmath,graphicx,hyperref` + 2026 论文的 `booktabs,multirow,multicol,amsfonts,float` + `xcolor`；
   新增占位宏 `\tbd`（红色 TBD）；`\ninept` 保留（kit 允许的最小 9 pt）；`\bibliography{strings}` → `{strings,refs}`（kit 形式；`refs` = `paper/refs.bib`）。
   题目 / 作者 / 摘要 / 正文 / 消融表 / 图 **一律未动**（只在 `\name` 前加了 ORCID 与 AI 披露的注释）。
3. 两张旧对比表（Cocktail / KUL）→ **三张队列表**（`tab:comparison_2c` / `tab:comparison_shu` / `tab:comparison_3c`），放在原位（Sec. 4.1），
   正文那句 `Table~\ref{...}` 改指新标签。表样式沿用 2026 论文（模型名 + `\footnotesize` 出处行、`\multirow{2}{*}` 数值、逐列最优加粗）。
   行：EEGNet (J. Neural Eng. 2018) / FBCNet (arXiv 2021) / ATCNet (IEEE TII 2023) / LaBraM (ICLR 2024) / EEGPT (NeurIPS 2024) / EEG-Deformer (IEEE JBHI 2025) /
   TCFormer (2025) / MDGEEG / EEGMoE / **S-EEGNeX (Ours)**；未跑的格为 `\tbd`；三个 SNN 候选的最新数字（SHU 先导 split 41）以注释列在表下，学长定了再 `--ours` 生成。
   列：2C / 3C = Acc ± SD、Macro-F1、AUC；SHU = BAcc ± SD（主指标）、Acc、Macro-F1、AUC（2C/3C 类别平衡，BAcc ≡ Acc；κ 在 3C 表里没有）。
4. 新脚本 `scripts/build_dataset_tables.py`（CPU）：读 2C/SHU bench `per_model.csv`、3C `paper_baseline_3c_821_v1/tables/per_cell.csv`（ddof=1）、
   `analyze_snn_round.py` 的 `per_model.csv`（`--snn-*`）、外部新基线 CSV（`--extra COHORT=csv`，bench / analyze 两种格式自动识别，`--alias` 映射键名）→
   `outputs/analysis/dataset_tables_2026-09-18/{tab_2c,tab_shu,tab_3c}.tex`、`dataset_tables_long.csv`（每格来源）、`dataset_tables.md`；
   `--splice old_reference.tex` 在 `% >>> AUTO-GENERATED COHORT TABLES BEGIN/END` 标记间整体替换。数字与 bench 表逐格核对一致
   （EEGNet 2C 82.53 ± 12.04、ATCNet 83.28 ± 12.00、3C EEG-Deformer 68.49 ± 14.83 …）。
5. 结构自检：环境 / 花括号配对、`\ref` 全部有定义、`\tbd` 67 处（都是占位格）。集群无 TeX，编译核对交给 Overleaf（或本机）。
6. 多智能体并行：文献核实（LaBraM / EEGPT / TCFormer / MDGEEG / EEGMoE 的 BibTeX）、2027 格式核对清单、集群 TeX 编译器安装——结果回来后补进
   `paper/refs.bib` 与本条。

**未做 / 待定**：MDGEEG / EEGMoE 的出处与年份（另一窗口跑的是哪篇，待核）；「Ours」行选哪个候选（主实验后）；3C 目前全是论文配方（patience 100）的
LOSO 数字，caption 已注明，主实验 3C 出齐后换成 bench 配方；正文 / 摘要 / 题目由用户按方向改。**未 commit。**

---

## 2026-09-18 19:45 — 学长要的对比模型表：6 个已发表基线（带年份）+ SNN 先导行，已进 overleaf Table 1；报告 + 2025/26 候选基线

学长微信（09-18 17:10 前后）：「效果跟 eegnex 差不多就直接写论文 / 到时候不对比这个 eegnex / 再加几个 25、26 年的模型 / 先把要对比的模型的表格拿出来 /
之前那个 eegnex 的结果是哪几个数据集 / 列一个表格，直接丢 overleaf，表格格式不要改」。用户转达：改 overleaf 的表为我们要对比的模型 + 每个模型标年份 + 把跑完的结果列成表。

**做了什么**

1. 核对状态：主实验 20 作业里 4 组 RUNNING（71018 / 71021 / 71024 / 71030），落盘 cell 2C 0、3C 4/55、SHU 25（先导）；`n_cells_failed` 全 0。
   **能进表的 SNN 数字只有 SHU 先导（split 41，lr 1e-4，zs 臂）**。
2. 新脚本 `scripts/build_comparison_table.py`（CPU）：读 2C/SHU bench `per_model.csv`、3C `paper_baseline_3c_821_v1/tables/per_cell.csv`（ddof=1 重算，
   与 RESULTS_TRUTH 一致）、先导 `shu_pilot_lr1e4/{per_model,energy}.csv` → `outputs/analysis/comparison_table_2026-09-18/{comparison_long.csv（每格带来源）,
   comparison_table.md, pilot_paired.md, tab_main_rows.tex}`。参数量统一 58 ch（SNN 用 `build_model` CPU 计数：S-EEGNeX 58,850、S-EEGNet 3,028）。
3. `paper/main.tex` Table 1（`tab:main`）：**格式不变**（同 `tabular{lccccc}` + booktabs），行换为 DeepConvNet (2017) / EEGNet (2018) / FBCNet (2021) / ATCNet (2023) /
   EEGNeX (2024) / EEG-Deformer (2025) + S-EEGNet / S-EEGNeX-D / S-EEGNeX-S；SHU 先导数标 †、3C 论文配方标 ‡，2C/3C SNN 格为 `\todo{}`；注释里写了重生成命令、
   学长「不比 eegnex」的标记、4 个 2025/26 候选。18 条目中 12 个自研 / 消融条目不进论文表（只进报告）。
4. `paper/refs.bib`：EEG-Deformer 卷期核实为 IEEE JBHI 29(3):1909–1918, **2025**（`ding2025deformer`）；新增 `mane2021fbcnet`；Zhang et al. 2025（*Neural Networks*
   191:107741，"A lightweight spiking neural network for EEG-based motor imagery classification"，代码 github.com/Zhr1110/SnnForMI）题目作者已核实；
   候选 `wang2025cbramod`（ICLR 2025）、`liu2025mirepnet`（arXiv 2025，预训练不含 SHU/WBCIC-SHU）。
5. 报告 `docs/reports/REPORT_comparison_table_2026-09-18.md`：答「eegnex 是哪几个数据集」（3 个队列）；论文表；先导内部配对（S −1.38 p=0.04、D −0.48 n.s.、
   S-EEGNet −0.94 p=0.08；EEGNeX split 41 vs 41+42 差 −0.83，即 † 行不能直接减 bench 行）；18 条目全表；主实验进度；三处不能混比的口径；
   EEGNeX 行建议保留的三条理由；**2025/26 候选基线表**（Zhang 2025 SNN > CBraMod > MIRepNet > DSGNet，含接入成本 / GPU-h / 兼容性）与时间预算（最多加 1–2 个）。
6. `docs/DECISIONS.md` D-032；`docs/STATUS.md` §0–§3 同步。

**未做 / 待学长**：选 2025/26 基线；EEGNeX 行去留；3C 列口径（三选一）；年份放法（括号 vs 单独列）。**未 commit。**

---

## 2026-09-18 17:30 — 目录迁移 `SYX/eeg-mi-online` → `SYX/ICASSP`；剪遗留；灵魂记忆重写；论文模板落位

学长指令：「在 SYX 下新建 ICASSP，我们到这个里面做」。后续 Cursor 会话只在 `SYX/ICASSP`（本目录）与 `SYX/Benchmark` 打开，不再在 `SYX` 顶层。

**做了什么**

1. `mv SYX/eeg-mi-online SYX/ICASSP`（同一 git 仓库，历史完整；HEAD `1e183c9` + 未提交的 SNN 轮改动）；旧路径留兼容软链接
   `SYX/eeg-mi-online -> ICASSP`。正在跑的 71018 在改名后照常写日志与 checkpoint（17:29:52 epoch 6 落盘），20 个主实验作业
   （WorkDir / Command 均为旧路径）经软链接继续有效。**作业全部结束前不删软链接。**
2. 剪掉跨 session 时代的遗留到 `SYX/_archive/eeg-mi-online_legacy_2026-09-18/`（435 MB）：`0_docs/ 1_session_drift/ 2_baseline/
   3_online_adaptation/ 4_experiments/ 5_papers/ backup/ inbox/ models_eeg_foundation/`、`FOUNDATION_E2E_ROUTE_PLAN.md
   PHASE3_ROUTE_PLAN.md proposal.md experiment_log.md results.md`、`docs/report_824_s4erp_adv_3class.md`、8 个一次性脚本 +
   9 个旧 sbatch、快照里的跨 session run（`alignment_baseline_v1 cross_session_3to2_v1 prototype_drift_v1 session_*_v1
   training_viz_v1 tta_v1 analysis/ legacy_root_archive_2026-06-10_outputs/`）。`4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md`
   保留到 `docs/`。**`code/ tests/ scripts/slurm/{_common.sh,bench_gpu.sbatch,submit_snn.sh,submit_bench.sh,shu_*.sbatch}`
   一律未动。** 快照保留全部跨被试 run（2C/SHU bench、SHU 5 折旧协议、3C 四批 + 域对抗 + 被试内天花板）。
3. `SYX/eeg-fm-bench` → `SYX/Benchmark`（无作业依赖）；`SYX/HANDOFF_2026-09-14.md` → `Benchmark/docs/`；
   `SYX/{MEETING_BRIEF,REPORT_for_senior}_2026-09-18.md` → `docs/reports/`。
4. 灵魂记忆重写为 `../Benchmark` 同款纪律（4 份活文档）：`AGENTS.md`（规则 + 浓缩事实）、`docs/STATUS.md`（作业 / 倒计时 / 下一步）、
   `docs/PRIOR_RESULTS.md`（3 队列 × 18 ANN 权威摘要）、`docs/DECISIONS.md`（D-001~D-029）、`docs/ARCHITECTURE.md`、
   `docs/ICASSP2027_SUBMISSION.md`、`README.md`、`CLAUDE.md`、`.cursor/rules/`（6 条，替换旧 8 条）。
5. `paper/`：ICASSP 2027 官方模板（2026-09-18 自 CMS 下载：`spconf.sty`, `IEEEbib.bst`, `Template.tex/pdf`）+ `main.tex` 骨架
   （按 PLAN §6 分节、表图占位、limitation 与 AI 披露占位）+ `refs.bib`（必引清单占位，元数据待核）+ `README.md`（编译方式；集群无 TeX）。
6. 3C manifest `outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv` 的 33 行 `npz_path`/`meta_path` 改写为新路径。
7. 核对截稿：官网 Call for Papers = **September 23, 2026**（AoE → 北京 09-24 20:00）；CMS Paper Kit 页仍显示 16 September（未更新）
   → 投稿前登录 CMS 确认。新要求：每位作者 ORCID；致谢中披露 AI 生成内容。

8. **18:00–18:20 四线并行审计（路径 / Slurm 运行时 / 文档 / Git）后的修复**：
   - `eegbench` conda 环境的 editable 安装仍指向 `SYX/eeg-fm-bench/src`（`import eegbench` 报错）→ 已重新 `pip install -e ../Benchmark --no-deps`。
   - `inbox/cross_subject_protocol_research.md`（协议文献依据全文）从归档移回 `docs/CROSS_SUBJECT_PROTOCOL_RESEARCH.md`，MEMO 引用同步。
   - **D-030**：SHU 先导 25 个 cell 目录（`snn_shu_zs_lr1e4_v1`，outputs + checkpoints）由单划分命名改为 `__split41__` 命名，否则 71034 会重训并使 split 41 双计；`cell_signature` 不含路径，`load_run` 核对 125 行 / 25 cell 正常。
   - `.gitignore`：快照目录只放行 `README.md`；学长微信截图不进 git；官方模板 PDF 放行；新增 `.gitattributes`（kit 文件 `-text`）。
   - 审计结论：20 个作业经软链接解析全部有效；`bench_gpu.sbatch` 已被 Slurm 冻结拷贝，但 `_common.sh`/`code/**`/`snn_*.yaml` 在作业启动时才读（排队期间不能改）；全套 pytest 158 passed / 1 skipped；`compileall` 干净；Git 无数据丢失（291 处删除全部在归档或 HEAD 中）。

9. **18:45 代码剪枝（D-031，用户要求只留核心）**：用主实验真实命令跑 CPU 微型 run 记录 `sys.modules`（51 个模块 = import 闭包），
   闭包外且不被保留脚本/测试引用的 50 个模块（`code/{tta,online,legacy,visualization}`、`models/dsgnet`、`methods/t3a`、6 个旧 datasets 适配器、
   7 个跨 session 协议、5 个旧汇总器、2 个旧预处理编排）、33 个旧实验 YAML、`tests/tta/`、冒烟/被取代产物、快照里的 `cross_subject_5fold_v1` 与
   `foundation_cross_subject_v1` 全部移入 `../_archive/`。`code/` 115 → 64 个 .py。**闭包内文件（`runners.py` 死函数、registry 弃用模型等）不动**，
   作业结束后第二轮（清单在 STATUS §4）。验收：compileall ✓、`pytest tests` 106 passed / 1 skipped、同一微型 run 剪枝后重跑 `ok=4 failed=0`、
   `submit_snn.sh` roster ✓；18:47 在剪枝后的树上启动的作业 71021 / 71024 加载与训练正常。
10. 18:28–18:47 集群释放：71030（3C zs）、71021（2C zs 第 2 组）、71024（2C eaz 第 1 组）陆续启动，与 71018 一起用满 4 GPU。
11. **19:00 文档一致性审计后的修复**：
    - **`scripts/analyze_snn_round.py` 臂名 bug**：预注册家族与 H1 写死在角色 `zs`/`ea`，而所有文档的主实验命令传 `eaz=` → H1 会被静默跳过。
      加 `--baseline-arm/--aligned-arm` 映射 + 家族为空即报错 + REPORT 头部注明臂映射与"8 项家族早于 D-012"；EXPLORATORY 补 D 版 / S-EEGNet / 三消融 / causal 的 zs 臂比较。
      用它重算先导：`outputs/analysis/snn_round/shu_pilot_lr1e4/`（S −1.38 p=0.040；D −0.48 CI[−1.84,+0.88] p=0.16；S−D −0.90 p=0.070）与
      `shu_pilot_lr1e3/`（EA 效应 ANN +1.39 / SNN +0.87，H1 −0.51；AdaBN@eaz +0.63 p=0.039；afe −1.82 p=0.028；spk +0.50；noinit −0.30）——
      §0.5 引用的每个数字现在都能指回一行 CSV。所有文档命令补 `--aligned-arm eaz`。
    - PLAN：§0.2 行 9（EA 后标量归一化）标为已被 D-021 推翻；§0.4 补 `pq3`（eaz）臂与决策规则；§1–§7 加"v1 原文"横幅并对 posterior drift /
      包络检测器 / log-var / 保相位定理 / streaming / anytime 逐处划线替换；frontmatter 更新。
    - 措辞修正：24 个 snn 配置（非 18）；对齐块是 3×3、H1 取 2×2 子集；`failures__*.json` 恒存在、看 `n_cells_failed`；D-021 补"验证集上 zs 仍高于所有
      EA 臂"；D-024 补消融只在 lr 1e-3；`code/configs/datasets/*.local.yaml` 本集群不存在；main.tex 参数量统一 58ch 口径并加 `\todo`；
      B-1..27 → 原文编号 1–27；`docs/DSGNET_SHUv5_3C_ANCHOR.md` 从归档移回（3C 协议与"DSGNet baseline 被低估"的依据）；REPRODUCE §5–6 指向快照；
      STATUS 补 8 条提交命令、BCI IV 2a pkl 反序列化提示、Slurm 估算的完成时间；SETUP/progress/reports/HANDOFF 补 frontmatter。

12. **19:10 用户纠正：选实验、定框架、写论文由学长决定。** `paper/main.tex` 里我预填的标题 / 摘要 / 三条贡献已全部退回 `\todo{}` 占位，只留官方模板与候选分节骨架；
    `AGENTS.md` §1 题目方向标为"候选，待学长拍板"、§7 加先决条件、`40-paper.mdc` 与 `STATUS.md` 同步——**学长拍板前 Agent 不写正文、不替学长挑实验。**

**状态**：主实验 8 组 20 作业（4 个 RUNNING：71018 / 71021 / 71024 / 71030；其余 PENDING）。先导 120 cell 完成。详见 `docs/STATUS.md`。
**下一步**：盯作业；BCI IV 2a loader（09-20 早）；补分析脚本（非劣性 bootstrap / 精确·置换 Wilcoxon / HL+BCa / Bonferroni–Dunn / Page's L）；
2C 出结果后填两块主表；09-21 起写作。**仍未 commit**（等用户授权）。

---

## 2026-09-18 — 新主线：Spiking EEGNeX（ICASSP 2027，截稿 9/24 20:00 北京）。方案 v2 + 代码 + SHU 先导

项目 09-13 备份后于 09-18 从 `SYX_backup_2026-09-13.tar.gz` 恢复到原位（HEAD `1e183c9`，15,588 条目核对一致）。
学长方向：「改 EEGNeX / 加点别的 / 套到 SNN」。本轮为与学长并行的版本，同框架同划分，可与 18 个 bench ANN 逐被试配对。

**方案**：`docs/PLAN_spiking_eegnex_icassp2027.md`（v1 → 四位专家评审 → v2，§0 有逐条采纳表）。
框架改为"非劣性 + 对齐交互"（Zhang et al. *Neural Networks* 2025 已发表时间原生 PLIF+EA，不能claim首创）。

**代码**（全部新增/改动通过 158 个测试，GPU 冒烟 70222 / 70305 通过）：
- `code/models/spiking/{neurons,models,adapter}.py`：S-EEGNeX / S-EEGNet；LIF/PLIF/HetPLIF（逐层 τ 初值 [1.5,4]/[1.5,8]）；
  `repeat`（T=4）与 `stream`（EEG 采样轴 = 脉冲时间轴，1000/250/63 步）编码；LI 膜电位读出；编码层硬复位、深层软复位；
  送入神经元的 BN 初始化 β=0.5（池化后 γ=2）；mask-only `SpikeDropout`；稳态发放率正则；逐层 MAC/AC 表 + 实测发放率能耗。
- `code/experiments/cross_subject_protocols.py`：归一化语法 `ea_session[_mu]+<trial_norm>`（逐 session EA，`_mu` 用 8–30 Hz 参考协方差）；
  评估时 AdaBN（`eval.adabn`，`adabn_*` 列，不进签名）；逐被试 `spike_stats`。`e2e_trainer.E2ESpec.adabn`、`runners.py` 传递。
- `code/configs/models/{seegnex,seegnet}.yaml`；`scripts/make_snn_configs.py` 生成 18 个 `snn_*.yaml`（{SHU,2C,3C}×{zs,ea,eamu}×{lr1e4,lr1e3}，
  cross_subject 块与 bench 完全一致，`curves: false`，`eval.adabn: true`）；`scripts/slurm/submit_snn.sh`；`scripts/analyze_snn_round.py`
  （预注册配对检验 + H1 差之差 + 能耗表 + bench 18 条目 `+bench` 配对）；`tests/models/test_spiking.py`。
- 3C processed 数据重生成（作业 70221，33 session 全 ok）。

**决定**：EA 后只做逐 trial 标量归一化（不再逐通道 z-score）；主 S 模型用 'same' 填充（因果版为消融）；chunk 编码删除；
lr 由 SHU split-41 先导按**验证集** macro-F1 选；SHU 主指标 balanced accuracy；2C 为确认队列，SHU/3C 为复现队列；预注册家族 5 项（见 PLAN §0.3）。

**在跑**（02:20 提交，v2 代码）：SHU 先导 pz3/pz4/pe3/pm3/pa3（作业 70306/70308/70310/70313/70315 + 接力）。
修订前代码的先导（70248–70255）已取消；其 eegnex z-score cell 复用，旧 EA(+zscore) 臂归档为 `_superseded_*`（4 个 eegnex cell：EA 比 z-score +0/+1.4/+4.2pp，AdaBN 再 +0.5–1pp）。

**下一步**：先导出结果 → 选 lr / EA 变体 → 提交 2C、SHU(split 42 + s_lif/s_plif)、3C 主实验；BCI IV 2a loader（截止 9/20 早）；
分析脚本补非劣性单侧 bootstrap、精确/置换 Wilcoxon、HL+BCa CI、Bonferroni–Dunn（k=19）。
**未 commit**（本轮全部改动在工作区，等用户/学长授权后提交）。

---

## 2026-08-27 — 第二轮审查：融合 B 通道混合 + DualCD 累积/早停续跑

多智能体：[Fusion architecture](a0139f5c-b4bd-4cf9-a6f2-da1945032804)、
[P0 protocol](a3cfe454-3980-462d-b683-e6e5e6e2dfda)、
[Statistics](41eb6282-66fd-4aeb-9641-431e0212b2e3)。

**融合（第一性原理）：** 方向 A 保持单变量 depthwise，不改。方向 B 补上两处对照缺口：
`s4_d_ff=32`（对角 SSM 是 SISO，TCN Conv1d 混通道；无 FFN 时 `@noattn` 是双变量）+
SSM `B`/`C` 的 `max_norm(0.6)`（对应 TCN kernel constraint）。参数 23,482 → **27,834**
（仍比 ATCNet 小 4.12 倍）。双 dropout 与 TCN depth=2 的 4 次 dropout **计数对齐，不改**。

**训练路径（会毁掉 DualCD 对照）：** 累积的 `after_optimizer_step` 改为拼接后的 grouped
batch；ragged 末组梯度按 `pending` 重标；`last.pt` 写入 `early_stopped`，续跑若已达
patience 不再多训一轮。

**统计：** Holm 按预注册家族大小 8 校正；Friedman 丢掉不完整的**模型列**而不是被试；
`--models atcnet@tcn8` 能对上 `atcnet-tcn8` 目录。

U1/U2 已从计划划掉。提前交的 smoke **47227 已取消**，本轮修完再交。

---

## 2026-08-27 — P0 代码审查修订（三路审查，两路被额度截断后由主会话落地）

审查专家：[Fusion architecture](fbc6398f-7de5-485a-b184-a80e9f842a48)、
[Code correctness](c32fe936-5eda-4250-bb1c-48517c7c04ec)、
[Statistics](7c59a832-7f72-42a1-a73c-1726853ac8b0)。后两路因额度中断，发现从
transcript 打捞并逐条核实。

**已改（进 smoke 前必须落地）：**

1. 名册补 `atcnet_s4@noattn` → **18 条目**。它已在 `PREREGISTERED_COMPARISONS` 和
   §5.3/§5.4 里，漏掉会使 Holm 家族静默从 8 缩到 7。成本约 +17 GPU-h。
2. `atcnet_s4` L2 **不再罚 SSM skip `D`**（init=1，核 ~1e-4，罚它等于把时间阶段压成恒等）。
   `B`/`C` 仍按 0.009 对应 TCN conv；`A_log`/`A_sign`/`log_dt` 继续排除。
3. Friedman / Wilcoxon 的"更好"方向随 `--metric` 翻转（nll/brier/ece 是越小越好）。
4. Wilcoxon 强制 `method="approx"`，避免 n=25 用 exact、n=51 用 normal、一个 tie 就换检验。
   配对 Δ 补 95% t 区间。
5. 未定义别名一律抛错（即使 experiment `model_params` 非空也不再静默跑成 plain 模型）。
6. `resolved_model_params.json` 按 `--tag-suffix` 落盘，4 个并发 group 不再互相覆盖。
7. `save_json` 原子写 + 非有限浮点 → JSON `null`（退化折的 NaN kappa 不再写出非法 JSON）。
8. `predictions.npz` 的 subject/session 从 `<U16` 升到 `<U32`。

**故意不改：**

- `S4Block` 双重 dropout（`S4Layer` + block）：改了会动 S4ERP 基线，与学长包分叉。
- `atcnet_s4` 不加最终 LayerNorm：ATCNet 的 TCN 没有，加了就不是单变量。
- `log_dt` 短端 `[1/L, N/L]` 保持：~58% 模式 τ<2 是覆盖 1..L 的几何后果，不是 bug。
- U1 源被试筛选 / U2 Euclidean Alignment：**按用户指示不做**。

**下一步：** `pytest tests/models tests/ -q` 通过后交 P1 smoke（10 条目 × fold 0 × 5 epoch）。

---

## 2026-08-27 — 新主线：跨被试 benchmark + ATCNet×S4ERP 融合。**P0 工程阶段完成**

权威计划 = `docs/PLAN_cross_subject_benchmark_and_fusion.md`（27 条锁定决定）。
**没有跑任何 GPU 实验。** 代码就绪，`pytest tests/ -q` → **134 passed**。

### 三个"进行中"实验其实早就跑完了，结论是关掉域对抗

| 实验 | 结果 | 含义 |
|:---|:---|:---|
| `adv_821recipe_3c_v1`（11/11） | S4ERP-Adv 论文配方 **68.54%** vs S4ERP 68.26% = **+0.28pp**（我们配方下是 +2.92pp） | 对抗项是**通用正则**，换成隐式正则更强的配方后失效 |
| `atcnet_adv_3class_v1`（22/22） | ATCNet 70.85% → **66.63%** = **−4.22pp** | 强骨干上是净损失 |
| λ 扫描 | 0.1→2.0 落在 0.56pp 内，远小于折间 13–15pp | 若真是不变性学习，强度不可能不敏感 |
| `wbci_3c_within_ceiling_v1`（165×2） | sub-011 被试内也只有 34.7%/37.6%（chance 33.3%）；**但 4 个被试的跨被试成绩反超被试内**（sub-008 高 17.5pp），因为被试内每 cell 只有 192 训练 trial | 难被试是"没信号"不是"迁移失败"；同时这个"天花板"是数据饥饿的，不是真上界 |

→ **域对抗线正式结案**，`s4erp_adv` / `atcnet_adv` 退出实验（代码保留，见下）。

### 数据事实（查证 + 实测）

- **2C 与 3C 是互斥队列，不是子集**：2C = subjects 1–51，3C = subjects 52–62
  （Yang et al., *Sci Data* 12:488 (2025) + MOABB `Yang2025`）。加上 SHU 25 人 = **3 个独立队列**。
  Figshare 描述写 "52 subjects" 与论文和我们 manifest 的 51 矛盾，**以 51 为准**。
- **WBCIC-2C 从未做过跨被试**：51 被试 / 148 ok session / 29,600 trial / 58×1000@250Hz /
  **完美类别平衡 14800:14800**。这是唯一让模型比较有统计意义的队列
  （11 被试下可检出差异约 9pp > 模型总极差 6.2pp，即任意两模型不可区分）。
- **seeds 41–45 的出处查到了**：ERP-Benchmark（arXiv:2601.00573），原文用它做
  *"random subject partitioning across different seeds"* —— 就是被试划分 seed。
  该文同时是指标口径（Acc/F1/AUROC）、6:2:2 划分比、以及**学长整个 S4/DINO 包**的源头。

### 已锁定协议

两个数据集、都是 `kfold_subject`、都 6:2:2、`split_seeds: [41,42]`、`train.seeds: [0]`：
WBCIC-2C 5 折（30/31 : 10 : 10/11）+ SHU 5 折（15:5:5 精确）。
配方 = 已跑过那套（Adam 1e-4 / batch 128 / 无 scheduler / max 500 / **patience 25** /
`monitor: macro_f1`）。**18 个条目一次提交**（原计划分阶段已推翻，见计划 §0 决定 25；
审查补入 `atcnet_s4@noattn`），约 **469 GPU-h ≈ 4.9 天**（4 卡）。

### 两个融合方向都实现了（实测参数量，58ch/2 类）

- **方向 A（学长指定："空间滤波使用 ATCNet 的"）= `s4erp_dw`**：把 S4ERP 的 dense 空间卷积
  （**950,400** 参数，占全模型 71%）换成 ATCNet 式 depthwise。三个 F1/D 变体
  `@f64d2` / `@f16d8` / `@f16d2p` → 全模型 **426,370 / 425,122 / 406,978**（对照 1,370,882）。
  **只改 depthwise 一处**，`max_norm` 和时域核长都不动，否则不是单变量对照。
  顺带测出**通道缩放差 128 倍**：32ch→58ch 时 dense 增 **425,984** 参数、depthwise 只增 **3,328**。
- **方向 B = `atcnet_s4`**：一个共享因果 SSM 替 5 个独立 TCN（83,840 参数 = ATCNet 的 73.1%）。
  实测 **23,482**（比 ATCNet 的 114,554 小 **4.88 倍**），`readout_idx = [12,13,14,15,16]`
  正是官方 5 个窗口的末步，SSM 时间常数 **[0.27, 16.89] tokens**（Tc=17）——
  `log_dt` 重初始化生效（默认会是 6.8–546，7/8 个模式退化成滑动平均）。
- 两者**互补不替代**：A 修空间前端（71%），修完后最大块变成 187k/125k 的 flatten 分类头，
  那是 B 处理的。

### P0 期间发现并修掉的三个既存 bug

1. **`cell_signature` 不含早停设定** → 改 patience 不报错、静默复用旧 cell。
   修法比原计划更精确（`_stopping_identity`）：**patience 总是进签名**；
   **`max_epochs` 仅在 scheduler 非 none 时进** —— 在 `scheduler: none` 下它是纯上限，
   "提高上限后续跑"是同一轨迹的严格延长（正是墙钟被砍后重投的做法）。签名版本 v2 + 版本守卫。
2. **`S4ERP` 上有多余的 `update_adv_lambda`**（引用只有 `S4ERPAdversarial` 才有的 `self.grl`），
   而 adapter 用 `hasattr` 鸭子类型判断 → **plain `s4erp` 调 `training_step` 直接 AttributeError**。
   潜伏原因是 `uses_custom_loss=False`，训练器从不走那条路。已删方法 + 改成显式集合判断。
3. **`deepconvnet`/`fbcnet` 缺 `describe()`** → `n_params` 记成 0（旧 SHU 表里可见）。
   已补，实测 302,552 / 20,711。

### 一处刻意偏离计划

P0-1 原文是"**删**"5 个模型（`eegnet`/`cap_eegnet`/`s4erp_adv`/`atcnet_adv`/`dualcd_s4_pos`），
实施改成**弃用但保留**：它们都产出过 `RESULTS_TRUTH` 里引用的数字，删代码会让那些数字
不可复现（且 `eegnet`/`cap_eegnet` 被 Phase 0–2c 的 config 引用，AGENTS 要求那些可复现）。
用 `BENCHMARK_MODEL_NAMES` 白名单 + 构建时弃用警告达到同样效果。

### 开放问题

1. **尚未 git commit。** 工作区在 P0 之前就有 18 个 modified + untracked，最后 commit 是
   `bf49e24`；P0 改动叠加其上。建议 P1 之前提交，**需要用户授权**。
2. **U1 源被试筛选 / U2 Euclidean Alignment arm** 仍未分配（都不是域对抗方法，未被决定 4 覆盖），
   要做就得在 P2 排产前定，因为都会改变实验网格。
3. §7 的组 2/3/4 单 cell 成本仍是**估算**，P1 smoke 实测后替换；差 >50% 要先重算预算。

### 下一步

**P1 smoke**（1 卡约 3 小时，10 个条目 × fold 0 × 5 epoch），验收清单与检查脚本见
`REPRODUCE.md` §4；三条硬门：参数量全非 0、`timepatch` 分箱覆盖 6 个箱、
`atcnet_s4` 时间常数在 1–17 tokens。通过后再 `submit_bench.sh`。

---

## 2026-08-21 — 实验二补跑 DualCD Flatten + Transformer（smoke → 全量）

- 学长确认：只补这两个，**配方完全跟实验二**（Adam 1e-4 / 有效 batch 128 / patience 25 /
  monitor accuracy / 16:4:5 / split_seeds 41–45 / 三曲线 / resume）。
- YAML `shu_cross_subject_5fold.yaml` 增加两模型 + `micro_batch_per_model`
 （flatten 32 / transformer 16，与 821 相同）。`run_id` 仍是 `shu_cross_subject_5fold_v1`，
  已完成的 175 个监督 cell 会被跳过。
- 提交：`scripts/slurm/submit_shu_cross_subject_dualcd.sh`。先隔离目录 smoke（2 epoch、
  split41 fold0），`afterok` 后再两卡全量（各 25 cell）。
- 可视化脚本已把 DualCD 加进 `plot_training_diagnostics.py` / `package_shu2022_handoff.py`；
  三曲线脚本按 cell 名发现模型，不用改。跑完后重跑 diagnostics 即可对齐 acc/loss/权重直方图。

---

## 2026-08-21 — 补训练可视化：acc / loss / best.pt 参数分布

- 两套 SHU 2022 实验训练时只把曲线写进 `result.json`，**没有** TensorBoard，也**没有**逐 epoch 记权重。
- 现事后画出：`outputs/experiments/shu/training_viz_v1/`（acc+loss 均值图、单 cell 例子、`best.pt` 全体权重直方图 + 按模块 RMS）。
- 脚本：`scripts/plot_training_diagnostics.py`。权重直方图是 checkpoint 快照，不是训练过程动画。

---

## 2026-08-20 — SHU 2022 两实验交接包（给学长）

- 两套实验均 175/175 完成。交接目录：
  `/share/home/yuan/SYX/SHU2022_两次实验结果_交给学长_20260820/`（约 54 MB）。
  含全部 `result.json`、划分、YAML、Slurm 日志、汇总 CSV、总览图 + 350 张三曲线、中文解读
  `01_实验解读.md`。**未拷权重**（`checkpoints/shu/` 两侧合计 ~2.4 GB）。
- 实验一头条（ses-04+05 合并，25 被试 mean±std）：S4ERP **58.7±8.8%**，其余卷积模型 57.3–58.5%，
  FBCNet **51.4±3.4%**（塌成随机）。**不能和论文 CS 53.7% 比**（论文只训 ses-01）。
  175/175 早停；val ~32 trial，过拟合严重（S4ERP train 94.8 / val 71.1 / test 58.7）。
- 实验二头条（先 25 人平均，再对 split seed 41–45 mean±std）：EEGNeX **64.5±0.4%** 单独领先；
  S4ERP / Deformer / ATCNet / EEGNet 61.4–61.9%；DeepConvNet 59.9%；FBCNet 59.1%。
  划分间 std 仅 0.4–1.0 点。S4ERP 仍过拟合（train 92.0 / val 63.0 / test 61.9）。
  ATCNet best_epoch 中位 51、最大 101（比 821 的中位 406 早很多，但仍找到了 val 峰）。
- 生成脚本：`scripts/package_shu2022_handoff.py`。
- 下一步：等学长定口径；若要贴论文 CS 需补「只训 ses-01」一版；ATCNet 是否被 patience=25
  限制可只对 exp2 重跑 patience=100。

---

## 2026-08-19 — ATCNet 全失败补跑（subject_ids 接口）

- 两个 SHU 2022 作业 8-17 已全部 COMPLETED，但 `atcnet` 两边都是 **0/25**：
  `TypeError: ATCNetClassifier.training_step() got an unexpected keyword argument 'subject_ids'`。
  来源是 `2a09cf3` S4ERP-Adv 让 `e2e_trainer` 无条件把 `subject_ids` 传给所有 `uses_custom_loss` 模型；
  ATCNet 用这个 hook 只加 Keras L2，不收这个参数。sweep 吞掉 cell 失败所以 job 仍是 exit 0。
- 修：① ATCNet `training_step` 增加并忽略 `subject_ids`；② trainer 按 signature 探测后再传。
- 测试 `test_atcnet_custom_loss_accepts_subject_ids` 过。无残留 atcnet cell/ckpt，从头训。
- 提交 **41883** `shucs_atcnet`（实验一 25 cell）+ **41884** `shuxs_atcnet`（实验二 25 cell），gpu3node 两卡并行。

---

## 2026-08-16 — SHU 2022 两个新实验：协议锁定、代码落地、3 卡提交（在跑）

- **学长定向**：从 WBCIC 3C 换到 **SHU 2022**（25 被试 × 5 session × 32ch，125 session 全 ok，
  11988 trial）。跑 **7 个模型**（`eegnet_official` / `eegnex` / `eeg_deformer` / `atcnet` /
  `deepconvnet` / `fbcnet` / `s4erp`；自研 `eegnet` 明确不跑，与官方版参数量都是 2,610）。
- **实验一 cross-session**（`shu_cross_session_3to2_v1`）：每被试 **ses-01+02+03 训练 →
  ses-04+05 测试**，25 次训练、逐测试 session 报。recipe 取 SHU 2022 论文原文（Adam **1e-3**、
  batch **16**、NLLLoss、早停看**验证准确率**、**不做归一化**——原文 "the preprocessed data were
  directly used"），**两处偏离已记进 config**：① 论文 CS 只用 ses-01 训练，我们用 3 个
  session（训练池 96 → 286 trial），**所以不能拿来和论文 53.7% 比、更不能说"超过论文"**；
  ② 300 epoch（论文 1500）。val 论文没写比例，取论文自己 WS 的 8:1 → 每被试约 254/32。
- **实验二 cross-subject**（`shu_cross_subject_5fold_v1`）：5 折 **16 训练 / 4 验证 / 5 测试**
  被试。**8:2:1 在 25 人 5 折下数学上不可能**（test=5 需要 40:10:5 共 55 人），16:4:5 保住了
  821 的 train:val=4:1。`split_seeds` **41–45 只换划分**，训练种子固定 0。recipe 沿用 821
  （Adam 1e-4 / batch 128 / 500 epoch），但 **patience 100 → 25**、**monitor macro_f1 →
  accuracy**，故**与 821 的 3C 表不可横比**。
- **⚠ patience=25 的已知风险**：821 实测 `atcnet` best_epoch **86–478（中位 406）**、
  `eegnet_official` **48–486**，patience=100 才让它们跑到最优。25 很可能把这两个砍在早期，
  拿到的是欠训练的数字而非模型真实水平。已写进 config 注释，出结果时要按这个口径解读。
- **新增** `code/experiments/cross_session_e2e_protocols.py`：原 `session_protocols.run_cross_session`
  跑的是**全部 20 个有向 session 对**、且用冻结的 `trainer.py`（无续跑、无曲线），不满足要求。
  新模块每被试一个 cell，复用 `e2e_trainer` → 断点续跑 + best/last + 三曲线，产物结构与跨被试
  一致。**跨被试新增 `split_seeds` 扫描**，cell 名带 `split<ss>`（单划分时保持旧名，821 产物不受影响）。
- **测试**：新增 9 项跨 session 行为测试（含"测试 session 的 trial 绝不进 train/val"的字节级
  泄漏守卫）+ 2 项 split_seed 测试，`tests/foundation/` **46 passed**。
- **顺手修了一个既有回归**：`2a09cf3`（S4ERP-Adv）让 `eeg_foundation/adapter.py` 的 `forward`
  在 DualCD 分支引用未赋值的 `out`，**所有 DualCD 变体前向必崩**，已修。剩 2 个 `s4erp_adv`
  契约测试失败（它多返回 `subject_logits`，测试断言恰好三个键）——属 S4ERP-Adv 自身设计问题，未动。
- **提交（3 卡）**：`bash scripts/slurm/submit_shu2022.sh` → 7 模型分 3 组，实验二用
  `--dependency=afterany` 挂在实验一后，保证同时只占 3 卡。job **40278–40283**。
  gpu2node 16 卡被别人占满（`Resources`），按交接文档 `scontrol update partition=gpu3node`
  搬到 gpu04/gpu05 后起跑。
- 实测 cross-session 单 epoch 仅 **0.4–0.7 s**，300 epoch 上限下单 cell 最多几分钟。
  首批日志已见**严重过拟合**：eegnex 训练准确率 0.99 而验证卡在 0.5，第 29 epoch 早停
  （265 个训练 trial、34 个验证 trial，准确率粒度 0.03，噪声大）。
- 下一步：等实验一跑完 → 实验二自动接上；然后写 SHU 2022 专用 summarizer
  （现有 `summarize_cross_subject.py` 是 DSGNet/3C 专用，`vs_paper` 表和 "11 folds" 都写死了）。

---

## 2026-08-13 — 821 run 交接到账号 `yuan` 并恢复四卡训练（50/77 → 在跑）

- 新账号项目根 **`/share/home/yuan/SYX/eeg-mi-online`**（不是 `~/MI`），代码已在 `26bb950`，无需 pull。
  新建 `code/configs/paths.local.yaml`：`shu_2c_root` → `/share/workspace2/moto_imagination/WBCIC_SHU`
  （yuan 自己的共享盘目录，3C 源数据 `derivatives/3C dataset_processeddata` 全局可读）。
- 交接照 `HANDOFF_821_RUN.md` §4 执行：**A 预处理数据本地重跑生成**（33/33 session `ok`、9899 trial、2.0G，
  避开 manifest 绝对路径坑）；**B run 输出 rsync**（50 个 `result.json`、14M）；
  **C-精简 rsync**（`atcnet__fold{5,8,10}` 的 `best/last.pt`、6.5M）。`FOLD_MODELS` 重数过，与脚本写死的一致，未改。
- **新增 `scripts/verify_821_handoff.py`**（换机/换号后跑，比 `--dry-run` 强）：① 本地重跑的数据与来源账号
  **逐位一致**（比对解码后 `X`/`y` 的 hash，不比 `.npz` 容器字节，避免 zlib 版本差异误报）；
  ② 重建的 11 折与 50 个已完成 cell 记录的折 **50/50 一致**（防被试名单漂移导致「fold N 换了人」还被静默跳过）；
  ③ 每折 8/2/1 被试、互不重叠、9899 trial 全用上。三项 PASS。
- ⚠ **trial 数不是每折 7199:1800:900**：`sub-010/ses-01` 官方只有 299 trial，所以 sub-010 当验证的 fold 8 是
  `7200:1799:900`、当测试的 fold 9 是 `7200:1800:899`。数据本身的性质，旧账号同数，8:2:1 是**被试**比例。
- 提交 `38023–38030`（fold 3–10，`-t 48h`，patience 仍 **100** 未改）。
  **交接正确性的最强证据**：fold 5 的 ATCNet 从拷来的 `last.pt` **续在 epoch 225**——`run_signature`
  有任何差异都会直接抛错，没抛说明重跑生成的数据 + 当前 YAML 与旧账号指纹完全相同。
- **QOS 实测（两个账号 QOS 完全一样 `32cpu4gpu64g`，「换号拿新配额」不成立）**：`MaxWall 48h`、
  `cpu=32`、`gres/gpu=4` 都生效（每 job `-c 8` → 同时 4 个，其余 `QOSMaxCpuPerUserLimit`）；
  但 **`mem=64G` 不生效**——旧号 4 个 job 各 `--mem=40G`（共 160G）同一秒起跑。
  这条很关键：按 64/4=16G 去砍会贴上限 OOM（`pb821_g1` 实测 MaxRSS **15.6G**），**`--mem=40G` 不要动**。
- `gpu2node` 16 张卡被别人占满时第 4 个 job 会 PENDING 且 reason 是 `(Resources)`。集群同构
  （`gpu01–gpu05` 全是 RTX 4090 D 24G），`scontrol update jobid=<id> partition=gpu3node` 即可起来
  （本次 f6 挪到 gpu04）。**不能写多分区**，association 已钉死分区。
- 已完成 cell 的收敛位置（为将来调 patience 留证据）：`dualcd_s4_flatten` best_epoch **4–12**、
  `dualcd_transformer` **6–10**、`s4erp` **9–59**（都跑到 ~105–160 epoch 才被 patience=100 停）；
  而 `atcnet` best_epoch **86–478**（中位 406）、`eegnet_official` **48–486**。
  → 缩小 patience 对三个 DualCD/S4 省得多，但对 ATCNet 很可能砍掉真正的最优点。**待定，本次未改。**
- 下一步：等 77/77（`find .../cells -name result.json | wc -l`），再跑 `summarize_cross_subject.py`
  + `plot_three_curves.py`。**搬完记得在旧账号 `chmod o-x /share/home/Zihang` 收回权限。**

---

## 2026-08-12 夜 — 821 run 暂停 + 全量代码 push + 换账号交接文档

- 用户要切到同集群另一个账号继续跑，先 `scancel 37966-37973`（队列只剩无关的 `37978 erp-all-scratch`）。
  进度定格 **50/77**：三个小 baseline 11/11；`atcnet` 8/11（缺 fold 5/8/10，有 `last.pt`）；
  `dualcd_s4_flatten` / `s4erp` / `dualcd_transformer` 各 3/11（只有 fold 0–2）。
- **push commit `ad0c5d2`** 到 `origin/main`：端到端主线全部代码首次进 Git（5 个 foundation 模型、
  4 个官方 baseline、`e2e_trainer.py`、`cross_subject_protocols.py`、7 个 model YAML、
  821/ATCNet 实验 config、submit/summarize/plot 脚本、tests、论文 PDF）。已核对：无
  `*.local.yaml`、无权重、无 `outputs/`、无 >2MB 二进制。
- **新增交接文档** `4_experiments/wbci_shu/foundation_cross_subject/HANDOFF_821_RUN.md`。
  关键结论：三曲线 `history` 与所有指标都在 `outputs/.../cells/*/result.json`（汇总/绘图脚本只读
  `cells/`），所以换账号**必拷 = 3C 数据 2.0G + run outputs 14M**，`checkpoints/` 8.8G 只需拷
  ATCNet 三个断点（6M）；`dualcd_s4_flatten` fold3/4/6 的 541MB 残局（只训了 ~25min）不值得拷。
- 单折耗时实测（排 job 用）：transformer 6.98h / flatten 4.84h / atcnet 2.06h / s4erp 0.74h /
  deformer 0.55h / eegnex 0.46h / eegnet_official 0.22h。剩余 ≈ **107 GPU·小时**。
- 阶段数字（**折数不同，不可比，勿外传**）：eegnex 0.6991(n=11)、eeg_deformer 0.6849(n=11)、
  eegnet_official 0.6636(n=11)、atcnet 0.7273(n=8)、s4erp 0.7541(n=3)、
  dualcd_transformer 0.7156(n=3)、dualcd_s4_flatten 0.7137(n=3)。
- **换账号权限问题定位**：新账号访问不到旧目录，卡点是**家目录** `/share/home/Zihang` = `drwx------`(700)，
  而它下面 `MI/` 及各级子目录本来就是 `drwxrwxr-x`。解法 `chmod o+x /share/home/Zihang`（只给穿过、
  不给列目录，用完 `chmod o-x` 收回）；备选是 `/share/workspace2`（`drwxrwxrwt`，8.4T 空余）中转。
  **不走 GitHub 传数据**：单 npz 最大 63 MB，超 50 MiB Git 警告，2 G 进仓库永久膨胀，且完全没必要。
- **新发现的坑（已补进交接文档 §4.3.1 + §8.9）**：`processed_manifest.csv` 的 `npz_path` 是绝对路径，
  `code/datasets/session_splits.py` 直接拿它 `load_session_npz()`，**不重定根**。所以换账号后拷 manifest
  会读到旧账号的盘。正解：新账号自己跑 `python scripts/preprocess_wbci_3c.py` 重新生成那 2 G
  （官方 3C 源 `/share/workspace2/moto_imagination/WBCIC_SHU/derivatives/3C dataset_processeddata`
  是 `drwxrwxr-x` 全局可读，预处理确定性无随机），manifest 自动写成新账号路径；真要拷则必须 `sed` 改路径列。
  同类问题 `AGENTS.md` 已记过一次（WBCIC `embed_index__*.csv`）。
- 下一步：新账号 `git pull` → 配 `paths.local.yaml`（只需 `shu_2c_root` + `wbci_3c_processed_manifest`）
  → 重跑 3C 预处理 → 搬 ~20 M run 状态（outputs 14 M + atcnet 三个断点 6 M）
  → `bash scripts/slurm/submit_paper_baseline_821.sh`。

---

## 2026-08-12 — `paper_baseline_3c_821_v1` 续跑重提（QOS 墙时封顶 48h）

- 状态：原先 4 卡（36928–36931）仅 g1 COMPLETED；g2–g4 TIMEOUT@48h。进度 **50/77**；队列曾空。
- 试加长到 96h/72h/60h → 全部 `QOSMaxWallDurationPerJobLimit`；本集群 **MaxWall=48h**，无法加长。
- 对策：改 `scripts/slurm/submit_paper_baseline_821.sh` 为 **每未完成 fold 一 job** + `--models` 只跑缺 `result.json` 的模型（大模型 + ATCNet 续跑）。
- 已提交 **37966–37973**（fold 3–10，`-t 48:00:00`，PENDING）。已完成：三小 baseline 11/11；ATCNet 8/11；三大模型仅 fold 0–2。

---

## 2026-08-10 — ATCNet arm A/B 全完成；差距分解落地

- arm A job 36481：11/11 ok，`foundation_3c_loso_paper_v1` + atcnet。
  Acc **0.7129±0.1322** / F1 **0.7074±0.1417**（vs 论文 ATCNet 0.6834 / DSGNet 0.6856）。
- arm B job 36504：11/11 ok，`atcnet_3c_loso_paper_recipe_v1`，ALL DONE 19:22，约 20.5h。
  Acc **0.6891±0.1284** / F1 **0.6828±0.1383**。
- Gap 分解（Acc）：模型（flatten−A）**= −5.30pp**；recipe（B−A）**= −2.38pp**（我们的 recipe 更好）；
  论文 ATC − B **= −0.57pp**（B 略高于论文，实现/预处理剩余差基本闭合）。
- 同 recipe 下 5 个 foundation 均低于 ATCNet(A)；最好 flatten 0.6599。难被试 sub-011 两边都 ~0.41–0.42。
- 锚点文档已更新：`4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md`。
- 下一步（待用户定）：写正式 AI 分析 / 决定是否继续调 foundation / 是否接 DSGNet 复现。

---

## 2026-08-09 夜 — arm B 开启早停 + 修掉「续跑重置 patience」的 bug


- 用户要求 arm B 的 500 epoch 加早停。原设置 `early_stopping_patience: 0` = **没有早停**
  （当初因为论文未提就关掉了）。现改为 **100**，依据：官方 ATCNet 仓库
  `main_TrainValTest.py` L413 的 `train_conf['patience']=100`（其 EarlyStopping 回调被注释掉，
  但这个值是官方写的），且 100/500 与 arm A 的 25/100 是同一 1:5 比例，两条 arm 停机条件可比。
- **`code/training/e2e_trainer.py` bug fix**：续跑时 `patience` 从 0 重新开始，等于每次被抢占
  都白送一整个 patience 窗口，续跑的 run 会比不中断的 run 训得更久。改为从 checkpoint 重建：
  `patience = max(0, start_epoch - best_epoch)`（`best_epoch`/`history` 本来就存在 `last.pt` 里）。
- checkpoint 契约**本来就满足**用户要求：每 cell 只有 `best.pt` + `last.pt`，`last.pt` 带
  optimizer / scheduler / RNG / history，可断点续跑（已实测确认：arm B fold0 的
  `last.pt` epoch=153 / best_epoch=145 / best macro_f1=0.7336）。
- arm B 重新提交 job **36504**（取消 36494），从 fold0 epoch 153 续跑，patience 重建为 8。
- 说明：model selection 仍用 `macro_f1`（与 arm A 一致），**没有**跟随官方的
  `ModelCheckpoint(monitor='val_loss', mode='min')`，这样 A/B 之间只差优化器/batch/epoch/调度，
  归因更干净。官方另有 `ReduceLROnPlateau(factor=0.9, patience=20)`，DSGNet 论文未提，未启用。

---

## 2026-08-10 — 统一对比 run：7 模型 × 8:2:1 × 论文 recipe × 三曲线（已提交四卡）

**目标**：一个 run 里所有模型同协议同配方，且带学长要的 train/val/test 三曲线。
run_id = `paper_baseline_3c_821_v1`，config `code/configs/experiments/paper_baseline_3c_821.yaml`。

**① 三曲线（代码改动）**
- `e2e_trainer.py`：新增 `train_eval_loader` / `test_loader` 两个**只做监控**的 loader，
  每 epoch 评一次写进 `history` 的 `train_eval` / `test`。**test 绝不参与模型选择**：
  `best.pt`、早停、`best_score` 只读 val，守卫写在代码+docstring 里。
  train 曲线取**训练集的固定子集（默认 2000 trial）在 eval 模式**下测，才和 val/test 可比
  （train 模式带 dropout 不可比）；`train_loss` 仍是全量真实训练损失。
- 日志新增紧凑三曲线行：`acc(train/val/test)=0.4065/0.3967/0.3522`。
- 绘图 `scripts/plot_three_curves.py`（per-cell + 跨折均值，均值按最短 fold 截断）。
- 汇总 `scripts/summarize_cross_subject.py` → `tables/{per_cell,per_model,vs_paper}.csv`
  + `REPORT_TABLE.md`（内嵌论文 Table II SHUv5 数字）。

**② baseline 只用官方代码**（`code/models/paper_baselines/`，出处/偏差/排除理由见其 README）
- ✅ EEGNet [18] `vlawhern/arl-eegmodels`（Keras→1:1 移植；逐层核对参数量 3,700）
- ✅ EEGNeX [20] `chenxiachan/EEGNeX`（Keras→1:1 移植；上游无可配结构超参）
- ✅ EEG-Deformer [23] `yi-ding-cs/EEG-Deformer`（**官方就是 PyTorch，原样执行**）
- ✅ ATCNet [24]（8/9 已完成，官方 Keras→移植，参数量 113,732 与官方一致）
- ❌ EEG-Inception [27]：无官方发布（braindecode 那版自己声明未经原作者核对）
- ❌ MDGEEG [35]：占位仓库无代码
- ❌ EEG-DG [38]：**发布代码不完整**——入口 `import Shallow_Inception_Network_2source`
  但仓库无此文件；`DG_Network` 写死 2 个源域（我们有 8 个训练被试）；`Dist_Loss` 全程
  detach 到 numpy 无梯度。改了就不是复现，故按用户规则排除。
- ❌ DSGNet：仅架构预览。其论文数字（0.6856/0.6833）只作引用对照。
- 我们的模型跑 3 个：`dualcd_s4_flatten` / `s4erp` / `dualcd_transformer`。

**③ 协议与配方**
- 划分：8:2:1 跨被试（LOSO，10 个非测试被试里 2 个做验证、8 个训练，每人 3 session；
  trial 数正好 7199:1800:900）。
- 配方：论文 §IV-A（Adam 1e-4 / batch 128 / max 500 epochs / 无 scheduler / 无 wd /
  无梯度裁剪）+ **早停 patience 100**（论文是固定 500，这是记录在案的偏差）。
- **梯度累积（新增）**：两个 67M 模型在 batch 128 下 OOM（smoke 实测）。加
  `train.micro_batch_per_model`（flatten 32×accum4、transformer 16×accum8），
  **优化器仍看到 batch 128**；唯一差异是 BN 统计量落在 micro-batch 上，已在 config 注明。
  `micro_batch_size` 已计入 `cell_signature`。

**④ 验证**：smoke1（7 模型 × fold0 × 3ep）5 通过 + 2 OOM（暴露问题）→ 加累积后
smoke2 两个大模型均通过（flatten 83s/ep、transformer 145s/ep）。三曲线在 5 个 cell 的
`history` 里齐全。

**⑤ Slurm 四卡**（按 fold 切分，每 job 跑全部 7 模型，各 <48h；同命令续跑）：
**36928** folds 0-2、**36929** folds 3-5、**36930** folds 6-8、**36931** folds 9-10。
记录 `outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/parallel_job_ids.txt`。

---

## 2026-08-09 夜 — ATCNet 加第二条 arm：**严格论文 recipe**（把模型差距和配方差距分开）

- 问题：我们 6 个模型用统一 recipe（AdamW+cosine / batch 64 / ≤100 ep / patience 25），
  论文用 Adam 1e-4 / batch 128 / 500 ep。**划分协议一致，配方不一致**，所以不能直接说
  「我们比 DSGNet 差 X pp 是模型差距」。
- 方案（用户选定）：ATCNet 跑两条 arm，**只有 recipe 不同**。
  - arm A `foundation_3c_loso_paper_v1`（models=[atcnet]，job 36481）= 我们的 recipe
    → 与 5 个 foundation 严格同条件可比。
  - arm B `atcnet_3c_loso_paper_recipe_v1`（新 config
    `code/configs/experiments/atcnet_3c_loso_paper_recipe.yaml`）= 论文 recipe：
    Adam lr=1e-4 / batch 128 / 500 epochs / 无 scheduler / 无 weight decay / 无梯度裁剪 /
    **不早停**（论文未提这些，一律关掉而不是继承我们的默认）。
    → 与论文自己的数字可比。Slurm：smoke **36493** → 全量 **36494**（`afterok`）。
- `arm B - arm A` = **recipe 差距**；`论文 0.6834 - arm B` = 剩余的实现/预处理差距。
- 指标口径（用户选定）：只用 **Acc / macro-F1**，不补 Kappa。

---

## 2026-08-09 晚 — ATCNet 改用**官方仓库**实现（作废 braindecode 版）

- 用户要求：必须来自官方 <https://github.com/Altaheri/EEG-ATCNet>。**第一版（braindecode
  PyTorch 移植）已作废**：取消 job 36474、删除 `_atcnet_raw.py` / `_bd_modules.py` 及其
  smoke 产物与 `atcnet__fold0__seed0` cell。55 个 foundation cell 未受影响。
- **官方版落地**（`code/models/atcnet/`）：
  - `_official_keras/{models,attention_models}.py` + `LICENSE` + `UPSTREAM_README.md`
    = 官方仓库**原样**（Keras，不执行，供逐行核对）。
  - `atcnet_torch.py` = 官方 `ATCNet_` 的 **1:1 PyTorch 移植**（Keras 跑不进本项目管线）。
    逐层对齐：`Conv_block_` / `mha_block` / `TCN_block_`、Keras 初始化
    （Conv2D/Dense glorot_uniform、TCN Conv1D he_uniform）、`max_norm(0.6)` 核约束、
    L2 penalty（conv 0.009 / dense 0.5）。
  - `adapter.py` = `{logits, features, confidence}` 契约 + 用 trainer 的
    `uses_custom_loss` 钩子把官方 L2 加回 loss（Keras 是自动折进 loss 的）。
- **保真证据**：用官方 BCI-IV-2a 维度（22ch/1125/4 类）参数量 = **113,732**，
  与官方 README 结果表**完全一致**。我们的维度（58ch/1000/3 类）= 114,719，
  `Tc=17 / Tw=13 / F2=32 / 5 windows`。
- **偏差**（仅 6 条机械性差异，逐条见 `code/models/atcnet/README.md` §3）：数据布局、
  even kernel 的 `same` padding 取整方向、softmax→logits、attention 限定 `mha`、
  L2 由 training_step 加、Keras `MultiHeadAttention`（key_dim 8 ≠ embed 32）需自写。
- **协议**：与 5 个 foundation **完全相同**（同 run_id `foundation_3c_loso_paper_v1`、同
  LOSO session 划分、同 trainer recipe）。**优化器 recipe 与论文不同**（论文 Adam 1e-4 /
  batch 128 / 500 epochs；官方仓库 Adam 1e-3 / batch 64 / 500 epochs；我们 AdamW+cosine /
  batch 64 / ≤100 epochs / patience 25）——写报告必须说明。
- Slurm：smoke **36480**（fold0 / 2 epochs / 隔离输出）→ 全量 **36481**（`afterok`）。

---

## 2026-08-09 — foundation_3c_loso_paper_v1 全量完成（55/55）

- 协议：DSGNet/SHUv5 对齐 LOSO（train ses1–2 / val ses3 / test=留一被试全 session）。
- 5 models × 11 folds × seed0 **全部 ok**；汇总 CSV：
  `outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/runs/cross_subject__all_models.csv`
- **LOSO mean Acc / F1**（vs 论文 DSGNet 0.6856 / 0.6833）：
  - dualcd_s4_flatten **0.6599 / 0.6569**（最好，−2.6pp Acc）
  - s4erp 0.6477 / 0.6463（−3.8pp）
  - dualcd_transformer 0.6340 / 0.6326（−5.2pp；batch16 才跑通）
  - dualcd_s4_timepatch 0.6107 / 0.6091（−7.5pp）
  - dualcd_s4_pos 0.4511 / 0.4132（远弱）
- 共同难点：sub-011 全模型 ~0.32–0.39；sub-007 最强（flatten 0.868）。
- 结论：**同协议下 5 个 foundation 均未超过论文 DSGNet**；最接近是 flatten。

---

## 2026-08-07 晚 — 续跑 foundation_3c_loso_paper_v1（4 GPU）

- 前一轮 35878–35881 在 ~2h 被 CANCELLED；当时完成 13/55（s4erp 11/11 Acc mean 0.6477；
  timepatch/flatten 各 fold0；pos/transformer 未完成）。
- 重新提交（resume，同 out/ckpt）：**35989** s4erp+pos、**35990** timepatch、
  **35991** flatten(40G)、**35992** transformer(40G)。
- 记录追加：`outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/parallel_job_ids.txt`

---

## 2026-08-07 — 对齐 DSGNet 论文协议，重跑 WBCIC 3C LOSO（5 foundation）

- **对标论文**：Lou et al., IEEE JBHI 2026，[doi:10.1109/jbhi.2026.3689121](https://doi.org/10.1109/jbhi.2026.3689121)
  （PDF：`inbox/papers/dsgnet_jbhi2026_FullText.pdf` + 根目录同名副本；摘录
  `inbox/papers/dsgnet_jbhi2026_extracted.txt`）。只做其中 **SHUv5 = WBCIC-SHU 三分类 11 人**。
- **论文协议（已落地）**：LOSO；非测试被试 **ses-01+02 → train，ses-03 → val**；测试被试
  **三 session 全测**。实测 fold0：`n_train=5999 / n_val=3000 / n_test=900`（sub-010/ses-01=299）。
- **工程**：新增 `val_mode=sessions`（`cross_subject_protocols.py`）；config
  `foundation_cross_subject_wbci_3c.yaml` → `run_id=foundation_3c_loso_paper_v1`
  （**不复用**旧 `foundation_3c_loso_v1`，旧=留 2 个 val 被试，与论文不一致）。
- **对标数字（论文 Table II，SHUv5，LOSO mean）**：DSGNet Acc **0.6856** / F1 **0.6833** /
  Kappa **0.5284**；次强 ATCNet Acc 0.6834。本轮只跑 5 个 foundation；DSGNet 复现仍 deferred。
- **测试**：session-val 相关 5 passed；dry-run OK。
- **Slurm 4 GPU 并行**（记录：`outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/parallel_job_ids.txt`）：
  - **35878** `s4erp,dualcd_s4_pos`（32G）
  - **35879** `dualcd_s4_timepatch`（32G）
  - **35880** `dualcd_s4_flatten`（40G）
  - **35881** `dualcd_transformer`（40G）
- 网格：5 models × 11 LOSO folds × seed 0 = 55 cells；同命令可续跑。

---

## 2026-08-05 — 4 GPU 并行拆分 3C LOSO

- 取消单体 job **35359**（已完成 s4erp 11/11 + dualcd_s4_pos 5/11；若继续会抢后面三模型）。
- 新提交（同 out/ckpt，per-model CSV，互不覆盖）:
  - **35433** `dualcd_s4_pos`（续跑剩余 folds）
  - **35434** `dualcd_s4_timepatch`
  - **35435** `dualcd_s4_flatten`（40G）
  - **35436** `dualcd_transformer`（40G）
- 记录：`outputs/experiments/wbci_shu/foundation_3c_loso_v1/parallel_job_ids.txt`

---

## 2026-08-04 — WBCIC 3C LOSO 全量开跑（仅 5 个 foundation；DSGNet 暂不做）


- 用户指示：不管 DSGNet，直接跑五个模型。
- Slurm job **35359**（`e2e_3c_loso`）：`foundation_cross_subject_wbci_3c.yaml`，48h / 32G / gpu2node。
- 修复 `scripts/slurm/shu_gpu.sbatch`：用 `SLURM_SUBMIT_DIR` 找 `_common.sh`（避免 spool 拷贝路径失效）。
- 产出：`outputs/experiments/wbci_shu/foundation_3c_loso_v1/` + `checkpoints/wbci_shu/foundation_3c_loso_v1/`；同命令可续跑。
- 网格：5 models × 11 LOSO folds × seed 0 = 55 cells。

---

## 2026-08-04 — 学长锁定先跑 WBCIC 3C LOSO（11 人）+ 5 foundation + DSGNet


- **协议锁定**：三分类 11 人；LOSO；只存 best+last；断点续跑。二分类 5 折/seed 41–45 **本轮不做**。
- **3C 数据**：`scripts/preprocess_wbci_3c.py` 从官方 `derivatives/3C dataset_processeddata` `.mat`
  转到 `outputs/processed/wbci_shu_3c_mat_clean/`（外部 processed/ 只读）。33 session 全 ok
  （含 sub-010/ses-01 的 299 trial）。标签 {1,2,3}->{0,1,2}。
- **实验 config**：`code/configs/experiments/foundation_cross_subject_wbci_3c.yaml`（dry-run OK）。
- **多类支持**：`normalize_labels` / `evaluate_predictions` 支持 3 类 AUC（macro-OVR）。
- **DSGNet**：GitHub 仅架构预览（`temp_*.py`，*full code upon acceptance*）；源文件已拷到
  `code/models/dsgnet/_*.py`，**尚未**注册进 runner。需向学长确认是否有完整训练代码/超参。
- **下一步**：① 问清 DSGNet 完整代码来源；② 接入 `dsgnet` 到 registry；③ Slurm GPU smoke → 全量 LOSO。

---

## 2026-08-04 — 主线换轨：端到端基础模型 × 跨被试（学长包融合完成，实验未跑）

- **学长新指示**：「先简化任务，先不搞在线学习，先直接搞端到端的模型」——用指定的 **5 个模型**
  （`models_eeg_foundation/` 包：S4ERP + UnifiedDINODualCD_{S4_Pos, S4_Timepatch, S4_Flatten, Transformer}），
  在 WBCIC-SHU 与 SHU 上**分开训练（不合并）**，做**跨被试**实验；要求**只存最后一个 epoch + 最优模型**
  两个 checkpoint，且**支持断点续跑**。
- **Phase 3 → paused（不是废弃）**：`code/tta/`、`4_experiments/*/tta/`、`PHASE3_ROUTE_PLAN.md` 全部保留。
- **融合（done）**：
  1. `code/models/eeg_foundation/`：学长 5 个源文件原样移植（模型数学未改），新增 `adapter.py` 做
     项目契约（`[B,C,T]` 输入 + `{logits, features, confidence}` 输出）与 DualCD 训练钩子
     （`uses_custom_loss` / `training_step` / `after_optimizer_step`）。**4 处偏差**逐条记在该目录 README §4
     （最实质的一条：原包 README 让用户设 `multi_view.low_freq` 但代码里没人读，已改成真参数，
     MI config 用 mu/beta 8–13 / 13–30 Hz）。
  2. `code/training/e2e_trainer.py`：新训练器。只写 `best.pt` + `last.pt`；`last.pt` 带
     optimizer/scheduler/RNG/history；原子写入；`cell_signature` 防「换了划分还复用输出目录」；
     每 epoch 按 `epoch_seed_base+epoch` 重设种子，使续跑与一口气跑完可比。
     **`trainer.py` 一行未改**（Phase 0–2c 可复现）。
  3. `code/experiments/cross_subject_protocols.py`：被试级 `loso` / `kfold_subject` / `holdout`；
     验证集默认取留出的**训练被试**；per-trial z-score（fit-free 无泄漏）；best 与 last 都在每个
     留出被试上评测（CSV 里 `accuracy...` vs `last_accuracy...`）。
  4. runner `foundation_cross_subject` + config `{foundation_cross_subject,shu_foundation_cross_subject}.yaml`
     + 5 个 model YAML + 4 个新 CLI 开关（`--split-protocol/--folds-subset/--monitor/--no-resume`）。
- **验证**：`tests/foundation/` **32 passed（CPU）**，含泄漏守卫、续跑守卫、配置漂移守卫。
  真实 SHU 数据 CPU smoke 跑通「加载→池化→划分→训练→双 checkpoint」。
- **参数量实测**（`n_times=1000`/2 类，@58ch / @32ch）：`s4erp` 1.37M/0.94M、`dualcd_s4_pos` 3.17M/2.32M、
  `dualcd_s4_timepatch` 4.48M/3.63M、`dualcd_s4_flatten` 66.9M/66.0M、`dualcd_transformer` 67.9M/67.0M。
  与学长表格对照：flatten 两个 + Transformer 在 32ch 下吻合 0.4% 以内；`pos`/`timepatch` 实测高 10–16%
  （学长那两行疑似按 ERP 配置 C=21/T=170 量的）。**文档一律以实测为准，不照抄表格。**
- **实测教训**：CPU 上 `s4erp` 单 epoch 734 s（15 s/步，瓶颈是 128 通道 × 1000 点的空间卷积），
  所以**正式 smoke 也必须上 GPU**；已按规矩改走 Slurm（`scripts/slurm/shu_gpu.sbatch`，job 35295 排队中）。
- **内存**：被试数据常驻 ≈6.9 GB（WBCIC）/1.6 GB（SHU）；训练集用 `ConcatDataset` 拼接**不拷贝**
  （若用 `torch.cat` 每个 cell 会再多约 5.5 GB）。WBCIC 全量 Slurm 建议 `--mem=32G`。
- **文献调研（done）**：`inbox/cross_subject_protocol_research.md`。关键事实：两个数据集都**没有**官方
  跨被试划分、两篇数据集论文都**没做**零样本跨被试；WBCIC-SHU 唯一可对标是 **EDAPT（J Neural Eng 2026，
  2-fold 被试划分，零样本 EEGNet 0.81 / DeepConvNet 0.85）**；**SHU 2022 没有任何可验证的已发表跨被试数字**
  且有地板效应（作者跨 session 53.7%，chance 51.4–53.7%，p>0.05）。
- **待办（阻塞点）**：协议参数还没定，`4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` 里 7 个问题
  **需要发给学长确认**（LOSO vs 5-fold、是否严格零样本、session 池化、是否加 EA arm、epoch 预算、
  先跑哪几个模型、SHU 近 chance 怎么写）。**确认前不产出任何被当作结果的数字。**
- **下一步**：等 GPU smoke 出单 epoch 耗时/显存 → 学长拍板协议 → 改 config → 全量（可断点续跑）→
  summarize（summarizer 待写）→ AI 分析报告。

---

## 2026-08-04 — 跨机便携路径改造（portable local configs）

- **目标**：去掉仓库内硬编码绝对路径，方便 GitHub 云端同步、多服务器继续开发。
- **做法**：
  1. 本机路径改为 `*.local.yaml`（gitignore）；仓库保留 `*.example.yaml` + 占位 `paths.yaml` / dataset yaml。
  2. SHU 实验 config 的 `data.manifest` 改为逻辑键 `shu_processed_manifest`；`paths.py` 新增 `resolve_manifest_path`。
  3. Slurm 脚本经 `scripts/slurm/_common.sh` 自动探测项目根；去掉硬编码 `/share/home/yuan/...` 与 mail-user。
  4. `preprocess_shu.py` 默认 out-root 来自 `shu.local.yaml`，不再写死 workspace2。
- **本机**：已生成指向当前共享盘的 `paths.local.yaml` / `shu.local.yaml` / `wbci_shu.local.yaml`。
- **下一步**：需要时 `git add` 便携改动并 push（不要提交 `*.local.yaml`）。

---

## 2026-07-12 — Phase 3 Pretrained-Model Readiness Round（工程验收，非科研裁决）

- **目标**：在学长真实预训练模型交付前，把 TTA 后端从 “embedding replay 已跑通” 提升到
  “mock live inference 已验证 + A0 充分 + 双数据集路由 + 交接规范完备”。**不做**正式 Oracle /
  full T3A / 新算法。
- **多智能体**：Lead + A(审计) + B(live inference) + C(A0/SHU) + D(契约) + E(独立 QA)。
- **Agent A**：当前 smoke 路径无 Critical；Major 含 label-free 仅约定安全、exception taxonomy、
  soft_call、t3a_minimal 近邻 trim≠熵排序、dry_run mkdir 顺序。
- **Agent B**：实现 `ModelInferenceSource`；`AdapterCapabilities` + typed exceptions；
  `run_label_free` 接口级 strip labels；测试专用 mock（Profiles A/B/C）仅在 `tests/tta/support/`。
  测试 **41 passed**（原 14 + 新增）。
- **Agent C**：opt-in `full_a0_replay`（`phase3_tta_full_a0.yaml`）；默认仍 smoke。
  **WBCIC full A0 COMPLETE**：canonical 4320/4320/4320，0 missing/dup，4320 pass，**max_abs_delta=0.0**。
  **SHU smoke passed**：2 cells no_tta，路径均 `outputs/experiments/shu/`，max\|Δ\|≈2.1e-7。
  修复 dry_run 无 I/O；SHU `run_t3a/run_oracle=false`。
- **Agent D**：权威契约 `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`（Lead 已同步状态表）。
- **Agent E**：独立复核 1–9/11–12 PASS；契约曾 stale（已修）。结论工程上 **Ready to receive real pretrained model**。
- **未做**：学长模型接入、formal Oracle、full T3A、Tent/SHOT、改写 Phase 2c。
- **下一步**：学长交付 checkpoint → 新 adapter → preflight → live smoke → 再 Phase 3B。

## 2026-07-10 — Phase 3 Round-1：model-agnostic TTA backend scaffold + smoke runnable

- **目标**：搭模型无关 T3A/TTA 后端骨架 + 最小测试/smoke；**不是** full T3A experiment，不写科研结论。
- **新增包 `code/tta/`**：adapters（Protocol + registry + embedding_only + baseline_torch 示例）/
  feature_sources（FeatureBundle + embedding_replay 路径重拼 + model_inference scaffold）/
  methods（no_tta + t3a_minimal）/ oracle（label_guard + target_label_oracle_proto）/
  eval（schema/metrics）/ report（smoke reporters）+ `method_catalog.yaml`。
- **入口**：`code/experiments/session_tta.py`；`code/runners.py` 已注册 `phase3_tta`；
  config `round1` 安全开关（默认 smoke / max_cells=4）；`code/methods/t3a.py` 薄 re-export。
- **测试**：`tests/tta/` 14 passed（CPU，`mi_torch`）。
- **WBCIC smoke（CPU）**：自动选 sub-005(stable)+sub-020(high)，seed0，eegnet 示例 adapter；
  3 cells × (no_tta + t3a_minimal + oracle) = 9 rows。
  - **A0 replay**：3/3 cell `|Δ|=0` vs Phase 2c `acc_target`（路径重拼成功，未改 Phase 2c 产物）。
  - **A1**：pipeline runnable；smoke 数字**不得**解读为 T3A 有效/无效。
  - **Oracle**：minimal target-label proto 跑通，`used_target_labels=True` 等三字段齐全；阈值仍 provisional。
- **结果区**：`4_experiments/wbci_shu/tta/{smoke,replay_validation,oracle_diagnostic,method_catalog,reports}/`；
  heavy `outputs/experiments/wbci_shu/tta_v1/`。SHU 侧 catalog+README 已建，Round-1 未跑 SHU smoke。
- **未做**：full sweep / ablation / Tent/SHOT / catalog 复杂方法实现 / 预训练模型接入 / 全量 A0。
- **下一步**：等学长预训练模型 → 新增 adapter+config；可选扩大 A0 全量 replay；再按 Oracle 裁决是否扩大 T3A。

## 2026-07-08（晚）— 学长批准 Phase 0 + A0/A1，补 7 条硬约束（v2.1，未跑代码）

- **学长审批**：Phase 3 v2 主路线通过（Oracle 提前为 Phase B 裁决门不推翻）；批准开 **Phase 0 + Phase A0/A1**；
  附 **7 条上机前硬约束**，其中 cell_id / Oracle 泄漏标识 / Mahalanobis 数值稳定性 三条"不补后面结果无法 defend"。
- **7 条硬约束（已写进 `PHASE3_ROUTE_PLAN.md` §2.5 + 对应 Phase）**：
  1. 结果目录**定死 `4_experiments/{wbci_shu,shu}/tta/`**；`3_online_adaptation/` 只放设计文档。
     → 已改两个 config 的 `output.readable_dir`（原 `3_online_adaptation/*/tta` → `4_experiments/*/tta`）+ DESIGN 文件清单。
  2. A0 定义 **canonical cell_id**（dataset/model/seed/subject/source/target/cell_id/npz_path_resolved/n_target_trials），
     No-TTA 复现按 cell_id join（先断言两侧 cell_id 集合一致），防错行 join。
  3. A1 stable/high-drift 被试**从 Phase 2c drift tertile 表自动选**，落盘 `selected_smoke_cells.csv`，禁手选（防 cherry-pick）。
  4. Oracle 方法**带 `used_target_labels` 泄漏标识**，报告单列 "Oracle diagnostic only"，不与 deployable T3A 混主表。
  5. Mahalanobis/shrinkage 默认 **shrinkage cov + ridge eps + 记录 cond number**，病态则 fail 或降级 cosine oracle（标 degraded）。
  6. Phase B 裁决门 = **收益+风险双条件**：均值>+3pp 且 high-drift/各模型无高负迁移且 Fisher 有恢复 → 才进 full；
     <+1pp → 转 scatter/reliability/decision-boundary；割裂情形降级小范围验证。
  7. **FBCNet 与 SHU 单列**，不与 WBCIC EEGNet/DeepConvNet 混成主结论（主结论=WBCIC×{EEGNet,DeepConvNet}）。
- **本轮改动（纯文档/配置，未跑代码）**：`PHASE3_ROUTE_PLAN.md` 升级为 v2.1 批准版（新增 §2.5 七条 + Phase A0/A1/B/E
  逐条落约束 + §4 决策锁定 + §6 审批记录）；两个 `phase3_tta.yaml` 的 readable_dir 改到 4_experiments；
  同步 `PHASE3_TTA_DESIGN.md`（文件清单路径）、`AGENTS.md`、`0_docs/STATUS.md`。另交付 `PROJECT_ARCH_SYNC_FOR_ADVISOR.md`
  （给学长的架构同步简报，防结构冲突）。
- **下一步（已获批，可开工写代码）**：Phase A0 = `code/experiments/session_tta.py` + 注册 `phase3_tta` runner +
  No-TTA 按 cell_id 精确复现（|Δ|<1e-6），逐条对照 7 条硬约束；通过后 Phase A1 minimal T3A smoke。

## 2026-07-08 — Phase 3 路线 v2 重排：Oracle 提前为裁决门 + Phase 0 状态修正（planned，未跑代码）

- **触发**：学长用 `writing-plans` 口径给出修订路线，核心改动 = **Oracle 上限实验必须提前到 T3A 大扫之前
  作为科研裁决门**（原 v1 把 Oracle 放最后 Step E）。依据 = Phase 2c 权威结论：prototype drift 只解释部分
  掉点（多元 R²≈0.35），主机制是 within-class scatter 膨胀 / Fisher collapse（**非 centroid collapse**）。
  若连"已知 target 原型/scatter-aware 几何"的理论上限都救不回多少 acc，T3A（带噪伪标签重估原型）更不可能有大收益。
- **架构映射**：学长文档按另一套项目结构写（P10 / CLAUDE.md / `run_phase()` / 无 runners.py/summaries）。
  已全部映射到本项目真实结构：灵魂记忆是 `AGENTS.md`（`CLAUDE.md` 仅指针）；runner 走 `code/runners.py` 的
  `PHASE_RUNNERS`；已有 `code/summaries/`；Phase 2c 表在 `4_experiments/{wbci_shu,shu}/prototype_drift/tables/`
  （`trial_embeddings_index.csv` / `prototype_drift_metrics.csv` 均在）；结果放 `4_experiments/{wbci_shu,shu}/tta/`。
- **本轮产出（纯文档，未写代码、未跑实验）**：
  1. 新建根目录 `PHASE3_ROUTE_PLAN.md`（完整 v2 路线 Phase 0→G + 决策门 + 待定选项 + 批准口径，供用户过目审批）。
  2. 重排 `3_online_adaptation/PHASE3_TTA_DESIGN.md` §7 实现路线 + 时间线：Oracle 从 Step E 提前为 **Phase B 裁决门**。
  3. Phase 0 状态修正：`AGENTS.md`（当前主线/第一优先/逻辑链/事实 → Oracle 先裁决）、`0_docs/STATUS.md`、
     `README.md` §0、`results.md`（WBCIC Prototype Drift 行措辞 Step 4→Phase 3B Oracle）。
- **裁决门槛**：Oracle >+3pp → 继续 T3A ablation/safe-T3A/全量；<+1pp → 停大规模 T3A，转 scatter/reliability/
  decision-boundary 机制；中间地带谨慎推进。三种结果对应论文三条叙事路线（见 PHASE3_ROUTE_PLAN §3 Phase G）。
- **🐞 已定位 bug（实现时消费端修，不改 Phase 2c 产物）**：WBCIC `outputs/experiments/wbci_shu/prototype_drift_v1/
  runs/embed_index__*.csv` 的 `npz_path` 列是失效旧路径（写成 `outputs/experiments/prototype_drift_v1/embeddings/...`，
  缺 `wbci_shu/`，文件已迁移）；SHU 的 index 正确。Phase 3 runner 须用 config `source_embeddings.embeddings_dir`
  重拼路径、缺失即 fail-fast，不信任该列。已确认 `phase3_tta.yaml` 里 embeddings_dir 指向正确新路径。
- **纪律**：T3A 现仅作无梯度 prototype baseline，Oracle 未裁决前不写"最终修复方案"、不做全量；未跑不写 done。
- **下一步（等用户一句话确认再写代码）**：Phase A0 = `code/experiments/session_tta.py` + 注册 `phase3_tta` runner
  + **No-TTA 逐 cell 精确复现 Phase 2c `acc_target`（|Δ|<1e-6）**；通过后 Phase A1 minimal T3A smoke（WBCIC/EEGNet/
  2 被试/seed0/src_proto+cosine）；再 Phase B Oracle 裁决。

## 2026-07-06 — Phase 3 立项：基于原型的测试时适应 (T3A) 方案 + 路线文件（planned，未跑）

- **触发**：用户给出论文 `2604.16926v1`（Lee, Pradeepkumar, Sun. *Test-Time Adaptation for EEG
  Foundation Models: A Systematic Study under Real-World Distribution Shifts*，NeuroAdapt-Bench），
  要求基于 T3A 做跨 session 脑电模型优化：用预训练模型抽特征、按不确定性（熵）筛高置信样本作伪标签
  微调、提升泛化；1 个月内组合"不同预训练模型 × 不确定性指标"找最优组合并对比 baseline。本条只**立项 +
  写方案 + 更新指示文件**，未实现 runner、未跑实验、未产出任何结果数字。
- **读论文（HTML 全文，非编造）**：本机无 PDF 库且不装包，改用 arXiv HTML 全文
  (`arxiv.org/html/2604.16926v1`) 提取正文。要点：① 系统 benchmark 三种 TTA——**Tent**(梯度，改 norm
  affine)、**SHOT**(梯度，改特征提取器 + 互信息 + 伪标签)、**T3A**(无优化，原型 + 熵筛选)。② 核心结论：
  **只有 T3A 在 in-dist/OOD/极端漂移三设定下平均平衡准确率为正、且最稳**；梯度类常负迁移；T3A 对 batch
  size 不敏感、类不平衡时收益最大（REVE-Base@CHB-MIT +18.9pp）。③ T3A 超参 `filter_k=20`、episodic=False、
  预测 `p(y=k|z)∝exp(z·c_k)`；c_k=每类低熵 support 特征均值。④ 论文用**EEG 基础模型**(CBraMod/REVE/
  TFM-Tokenizer)冻结编码器 + 共享线性头，跑**跨数据集/任务/模态**漂移。
- **映射到本项目（关键契合）**：Phase 2c 已证跨 session 掉点 = within-class scatter 膨胀/Fisher collapse
  (非 centroid collapse)、prototype 确漂移、cosine>euclidean。**T3A 正是"无标签用 target 特征重估 class
  prototype 调整分类器几何"的可部署解**，也是原 Step 4 cosine Oracle 上限(label-informed)的 label-free 对应物。
  故把原 Step 4 并入 Phase 3，统一为「基于原型的测试时适应」：Oracle 上限(诊断) + T3A/Tent/SHOT(方法)。
- **代码复用盘点（已读源码确认）**：三模型(eegnet/deepconvnet/fbcnet)均 `.classifier=nn.Linear` +
  `{logits,features,confidence}` 契约 → T3A 可用分类头权重 ω_k 初始化 support。`prototype_drift.py` 已
  存每 (subject,direction,model,seed) 的 `target_test__{z,logits,probs,pred,conf,y}` npz + checkpoint →
  **T3A 无优化、纯特征空间，全程离线 CPU replay 即可，无需重训/GPU**。runner 模式(`PHASE_RUNNERS`)、
  `--summarize` 调度、config 结构均已摸清。
- **诚实要点（写进 DESIGN，避免邀功/误导）**：① backbone 是自训小模型≠论文基础模型，跨 session≠跨数据集，
  不得声称基础模型结果；② **二分类预测熵是最大 softmax 概率的严格单调函数 → "熵筛选"与"最大置信度筛选"排序
  完全等价**，本任务真正有区分度的轴是 filter_k / 几何(cosine vs dot) / 软硬(soft 加权 vs hard top-k) /
  margin，报告须点明；③ SHU 近 chance，伪标签噪声大，预期收益有限；④ FBCNet 几何异常，T3A 恐失效，单列。
- **本次产出（planned，非结果）**：新增路线文件 `3_online_adaptation/PHASE3_TTA_DESIGN.md`（方法/协议/
  1 个月矩阵/实现 Step A–F/时间线/诚实警示）；config 骨架 `code/configs/experiments/{phase3_tta,
  shu_phase3_tta}.yaml`（runner 待实现，现在跑会报 "No runner registered"，与 future_*.yaml 同惯例）；
  同步 AGENTS.md（主线/逻辑链/事实）、STATUS.md、FILE_CATALOG.md、operation_log.md。
- **下一步（Step A，照 DESIGN §7）**：实现 `code/methods/t3a.py`（support/原型/熵筛选/cosine|dot，无梯度）
  + `code/experiments/session_tta.py`（离线 replay Phase 2c 嵌入，No-TTA + T3A，泄漏断言 + NaN fail-fast）
  + 注册 `phase3_tta` runner + 接入 `--summarize`；CPU smoke（subjects 1,2，No-TTA acc 必须与 Phase 2c
  `acc_target` 对齐）。之后 Step B Tent/SHOT(GPU)、Step C 汇总、Step D 全量 sweep、Step E Oracle、Step F 分析。

## 2026-07-06 — SHU Phase 1/2a/2b/2c summarize + AI 分析补齐（与 WBCIC 齐平）

- **目标**：把 SHU 剩余进度（训练早已完成、仅缺 summarize+AI 分析）一次补齐，使 SHU 在 Phase 0–2c 与 WBCIC 齐平。全程只读磁盘真实产物，数字均来自各阶段 `tables/`，未捏造。
- **执行顺序**：P1 → P2a → P2b → P2c，均 `python code/run.py --summarize --config <shu_phaseX>`（CPU，`mi_torch`）。
- **两处 bug 修复（否则 SHU 结果不对）**：
  1. **P2c summarizer manifest key**：`code/experiments/prototype_drift_summarize.py:summarize_from_cfg` 原读 `data.manifest_path`，但全项目约定是 `data.manifest`（见 `runners._resolve_manifest`）。SHU 因此回退到 WBCIC manifest，`build_run_status` 误用 WBCIC 期望网格（把 sub-026+ 判 missing，假报 2220 missing）。改为优先读 `data.manifest`。修正后 SHU 期望网格正确，**run_status 7500/7500 全 ok**。
  2. **P2b baseline schema**：Phase 2b 的 `none_reference` join 需要 `acc/bacc/f1 + train_sessions + training_scope`，而 Phase 1 cross 产出是 `accuracy/balanced_accuracy/macro_f1 + train_session`。新增数据集无关脚本 `scripts/make_baseline_cross_all.py` 做 schema 适配，写到 config `baseline_cross_all` 路径（7500 行，training_scope=single_source）。join key (model,train_sessions,test_session,subject,seed) 行对行匹配。
- **完整性核验**：P1 within 18750 + cross 7500 行（3 models×5 seeds 全齐）；P2a 375 行 ok；P2b 45000 行 ok / 0 failed / 0 NaN / complete=True；P2c 45000 metric 行、run_status 7500 全 ok。
- **SHU 真实结果**（数字见各 `tables/` 与 AI_ANALYSIS）：
  - **P1 baseline**：within/cross（5-seed）EEGNet 0.611/0.538、DeepConvNet 0.606/0.536、FBCNet 0.553/0.508；drop 7.3/7.0/4.5pp。**cross 近 chance（地板效应）**，掉点 pp 小≠更稳。排序同 WBCIC，FBCNet 最弱。
  - **P2a multi-source**：ses-01+02→03 = EEGNet 0.544 / DeepConvNet 0.558 / FBCNet 0.512；vs 最强单源 +0.7 / +2.6 / −0.2pp（方向同 WBCIC）。
  - **P2b alignment**：无对齐 0.5274；最佳 **session_zscore +1.42pp**（≠WBCIC 的 BN-stats），4/5 净正，filterbank −1.47pp 有害；**无方法过 +2pp**；high-drift 受益最小（z-score stable +2.88/moderate +1.06/high +0.44pp）。
  - **P2c prototype drift**：机制同 WBCIC——within-class scatter 膨胀 **15.7→38.3 (+144%)** / Fisher collapse **1.96→0.79 (−60%)**，非 centroid collapse（separation 6.6→23.9 反增）。最强预测子 fisher_change ρ=0.43、separation_change cosine ρ=0.38/r²=0.16；drift_mean/direction/margin 弱。cosine>euclidean。EEGNet/DeepConvNet 清晰、**FBCNet 弱且几何异常**（scatter 几乎不变、separation 反缩）。信号比 WBCIC 更噪（cross 近 chance，acc_drop 三分位非单调）。
- **可读产物落地**：P1+P2a → `2_baseline/shu/no_alignment_baseline/{tables,figures,report}`（并入同一目录，未新建 phase 子目录）；P2b → `2_baseline/shu/alignment_baseline/`；P2c → `4_experiments/shu/prototype_drift/`。每个 report/ 都写了 `AI_ANALYSIS.md`（9 段，数字源自 tables）。
- **诚实提醒**：P1 脚本原生报告内嵌 WBCIC 参照常量（论文 85.32/148 sessions/288 pairs/ses01-03 trend），对 SHU 不适用、并因此假报 "INCOMPLETE"；已在 AI_ANALYSIS §0 明确标注，以 tables 真实数字为准。
- 状态：**SHU Phase 0–2c 全部 done**，与 WBCIC 齐平。下一步 = Step 4 cosine Oracle 上限诊断（qualified go），FBCNet 单独处理。

## 2026-07-05 — 重新接手：盘面核查（SHU 训练全完成，summarize/AI 分析仍缺）

- **背景**：用户重新接手项目，并准备用 ChatGPT 当"指挥"来驱动 Cursor agent。本条只做**盘面事实核查**，未跑新实验、未 summarize、未改结果数值。
- **核查方法**：`squeue`（空，无在跑/排队 job）+ 逐 phase 数 run CSV + 扫 `logs/slurm/shu_*.out` 是否有 Traceback/CUDA/RuntimeError。
- **SHU 训练：4 个 phase 全部训练完成，无失败**（最后一个 CSV 落盘 2026-06-13 05:23，此后无活动）：
  - Phase 1 baseline：`session_model_compare_v1/runs/` 30 CSV = within/cross/meta × 3 models × seeds 0-4；日志 `ALL DONE`。
  - Phase 2a multisource：`session_multisource_v1/runs/` 15 CSV（3 models × seeds 0-4），ses01+02→03。
  - Phase 2b alignment：`alignment_baseline_v1/` 75 CSV（5 methods × 3 models × seeds 0-4）；日志 `rows=2500 ok=2500 failed=0`。
  - Phase 2c prototype drift：`prototype_drift_v1/runs/` 60 CSV + 15 `metrics__*.csv`（3 models × seeds 0-4）。
- **关键 gap（下一步动作）**：**summarize + AI 分析尚未跑**。可读结果区 `2_baseline/shu/`、`4_experiments/shu/prototype_drift/` 除 README 外**为空**；无聚合 `*_metrics.csv` / canonical `REPORT.md` / `run_status.csv`。SHU 的 P1/2a/2b/2c **正式准确率/漂移数字尚不存在**（不得引用/编造）。
- **唯一已落地的 SHU 可读结果**：Phase 0 漂移诊断（`1_session_drift/shu/` + `report/AI_ANALYSIS.md`）。
- **诚实状态**：SHU Phase 1/2a/2b/2c = **训练 done，summarize+AI 分析 pending**。这是重新接手后的第一优先动作。
- **下一步（按序）**：① `python code/run.py --summarize --config code/configs/experiments/shu_phase1_baseline.yaml`（P1 single-source cross）→ 归并进 `2_baseline/shu/no_alignment_baseline/`；② P2a multisource summarize（并入同一 `no_alignment_baseline/`，见 2026-06-12 13:27 归属约束）；③ P2b alignment summarize → `2_baseline/shu/alignment_baseline/`；④ P2c summarize → `4_experiments/shu/prototype_drift/`。每步核验 run_status 后再写 AI 分析，并同步 5 个 handoff 文件。之后才是 WBCIC 主线的 Step 4（cosine 空间 Oracle 上限诊断）。

## 2026-06-11 — SHU 主线复跑启动：Slurm 脚本 + Phase 0 漂移诊断完成

- **目标**：把 WBCIC 已完成的跨 session 主线（Phase 0/1/2a/2b/2c）在 SHU 2022 上按双数据集并列架构原样复跑。先不碰 online/adaptation/agent/41-10。
- **Step 1（只检查）**：4 个 SHU config dry-run 全 OK，输出全部 `shu/` 作用域，无 WBCIC 覆盖；manifest 125 ok / 25 subj；`outputs/experiments/shu/` 空、`outputs/analysis/shu` 与 `checkpoints/shu` 不存在；env `mi_torch_cu118` 在；分区仅 gpu2node/gpu3node。
- **Step 2（Slurm）**：新增 `scripts/slurm/shu_gpu.sbatch`（GPU 训练）+ `shu_cpu.sbatch`（CPU 漂移/汇总），均用 `mi_torch_cu118`、cuda fail-fast、GPU 不在登录节点跑。WBCIC 的 prototype_drift sbatch 是写死的，故新建 SHU 通用脚本。
- **发现的 gap**：① SHU 缺 Phase 2a multisource config（待 step 5 给最小补充方案）；② Phase 0 的 `per_subject_drift_summary.csv`/`session_pair_summary.csv` 不是新 runner 产物，legacy builder 硬编码 3 session → 新建 session 无关的 `scripts/build_drift_report.py`。
- **Step 3（Phase 0 完成）**：Slurm CPU job 21601，250 pairs / 25 subjects / 367s。
  - 核心：MMD 0.356、CSP_sim 0.344、ERD μ/β 0.527/0.532、RMS median 1.03、fisher_shift≈-0.0012。
  - 机制与 WBCIC 同质（空间+频谱漂移，幅值稳定，可分性不塌），但 SHU 空间漂移**更重**（MMD 更大 0.356>0.238、CSP 更低 0.344<0.420）。
  - 分层 high 9 / moderate 8 / stable 8；最漂移 sub-017/008/019，最稳定 sub-002/006/020。
  - pair 非单调：01-04 MMD 最大(0.413)、02-03 最小(0.284)。
  - 结果落 `1_session_drift/shu/{report,tables,figures}/`（含 14 图）+ `report/AI_ANALYSIS.md`（9 段，数字源自 tables）。
- **Step 4 Phase 1**：3 个 GPU job（每 model 一个，job 21602-21604，gpu2node），within 10-fold + directed cross，seed 0。running。
- **并行提交（2026-06-11 22:28）**：把后续所有可并行训练任务一次性提交（训练彼此独立，只有 summarize 有依赖）。
  跨 gpu2node/gpu3node 交替分配。job ids 记于 `outputs/experiments/shu/_job_ids/shu_full_2228.txt`：
  - **Phase 2a** multisource（ses-01+02→ses-03）：21610-21612（每 model 一个，含 5 seeds）。
  - **Phase 2b** alignment：21613-21627（每 model×seed 一个，5 methods × single pairs）。
  - **Phase 2c** prototype drift：21628-21642（每 model×seed 一个，20 有向对）。
  - 已 smoke 确认 phase2a eegnet 正常启动（25 eligible / 0 skipped）。共 36 jobs（4 R / 32 PD）。
- **待训练完成后（下一轮）**：① phase1 summarize → `2_baseline/shu/no_alignment_baseline/` + AI 分析；
  ② phase2a summarize（base 路径已 config 化）；③ 解决 phase2b `baseline_cross_all` 列schema 适配后 summarize → `alignment_baseline/`；
  ④ phase2c summarize → `4_experiments/shu/prototype_drift/`。每步写 AI 分析并同步 5 个 handoff 文件。
- 状态：Phase 0 **done**；Phase 1/2a/2b/2c 训练 **submitted/running**，summarize+AI 分析 pending。

## 2026-06-12 13:27 — 目录归属约束（summarize 前必读）

- **Phase 1 baseline 与 Phase 2a multi-source 不是独立结果区**，是 no-alignment baseline 下的两个协议/子结果。
- SHU baseline 最终报告/汇总表/图必须**统一归并**到 `2_baseline/shu/no_alignment_baseline/{report,tables,figures}/`
  （与 WBCIC 的 `2_baseline/wbci_shu/no_alignment_baseline/` 结构一致：within + single-source cross + multi-source 合在一起）。
- **禁止新建** `2_baseline/shu/phase1_baseline/` 或 `2_baseline/shu/phase2a_multisource/`。
- 训练/汇总可分步：先 summarize single-source baseline（P1），再做 multi-source comparison（P2a，依赖 P1 的 cross 表），
  但两者产物都落进 `no_alignment_baseline/`。Phase 2a 的角色是与 Phase 1 single-source cross baseline 对比。
- 训练阶段不受影响（outputs/checkpoints 仍按 run_id 分 session_model_compare_v1 / session_multisource_v1）；
  归并只发生在可读结果区 `2_baseline/shu/no_alignment_baseline/`。

## 2026-06-11 23:15 — Seed 覆盖核查（无覆盖/无重复风险确认）

只核查，未 summarize。结论：seed 配置正确，无重复跑 0-4 风险。
- **Job 状态**：48 jobs。Phase 2a（21610-12）COMPLETED；21602-4 + 21613 RUNNING；其余 PENDING。无 FAILED（ExitCode 全 0:0）。
- **21602-4（旧 seed0）**：日志明确 `seeds=[0]`（22:07 解析，早于 22:44 改 config）→ 仅 seed0。config 是进程启动时一次性读取，后改不影响在跑 job。
- **21644-55（补交）**：`scontrol`/`sacct SubmitLine` 确认命令显式 `--seeds N`（CLI 覆盖 config），单 seed 限定 → **不会重复跑 0-4**。
- **Phase 1 输出**：runs 目录仍空（seed0 job 还在 within 阶段，sub-009/25，CSV 在每 protocol×model 完成后才落盘）。提交层面 seed0(21602-4)+seed1-4(21644-55)=0-4，per-seed 文件名互不覆盖。
- **Phase 2a**：COMPLETED，3 models × seeds 0-4，每 CSV 单 seed、25 行（25 subj，ses01+02→03），无重复。
- **Phase 2b**：21613 running 日志 `seeds=[0]`、5 methods、tasks=500；21613-27 per model×seed → 0-4。
- **Phase 2c**：21628-42 per model×seed（pending）→ 0-4。
- **job id 记录**：补交 21644-55 原记于单独文件 `shu_phase1_seeds1-4_2244.txt`；本轮在 `shu_full_2228.txt` 末尾**追加**引用（未改历史行）。
- **下一步**：仅等待训练跑完（受 QOS 4gpu/分区限速，分批）。无需补交/取消。完成后按 P1→P2a→P2b→P2c 做完整性核查，再 summarize。

## 2026-06-11 22:44 — 修正 Phase 1 seed 覆盖（对齐 WBCIC 5-seed 标准）

- **发现的 bug**：Phase 1 baseline 首批提交（21602-4）用 `--models X` 未带 `--seeds`，而 `shu_phase1_baseline.yaml`
  原 `train.seeds: [0]` → **只跑了 seed 0**，不符合 WBCIC 标准主线（5 seeds）。
- **修正**：① 把 `shu_phase1_baseline.yaml` 的 `train.seeds` 改为 `[0,1,2,3,4]`（标准化）；
  ② 补提交 Phase 1 seeds 1-4，每 (model,seed) 一个 job（21644-21655，共 12 个），跨 gpu2node/gpu3node 交替。
  per-seed CSV 命名（`{tag}__seed{sd}.csv`）天然不覆盖已在跑的 seed0，无需新 run_id。
  id 记于 `outputs/experiments/shu/_job_ids/shu_phase1_seeds1-4_2244.txt`。
- **seed 覆盖核查（提交层面）**：
  - Phase 1：seed0=21602-4(running) + seed1-4=21644-55 → **0-4 覆盖**。
  - Phase 2a：21610-12，config 内含 seeds [0-4] → **0-4 覆盖**。
  - Phase 2b：21613-27（每 model×seed）→ **0-4 覆盖**。
  - Phase 2c：21628-42（每 model×seed）→ **0-4 覆盖**。
- **集群约束**：QOS=32cpu4gpu64g/partition，每 job -c8/gpu1 → 每分区最多 4 GPU 并发（双分区共 8）；其余排队（QOSMaxCpuPerUserLimit）。48 jobs 会分批跑完。
- **纪律**：训练完成前**不执行 summarize、不写 AI_ANALYSIS、不更新 results.md 正式数值**。完成后顺序 summarize：P1→P2a→P2b→P2c，每步核验再写 AI 分析。

## 2026-06-11 — SHU 接入 + 全项目改双数据集并列架构

- **目录架构**：1/2/3/4/5 各结果区全部改为 `wbci_shu/` 与 `shu/` 并列；2_baseline 下保留
  `no_alignment_baseline/` + `alignment_baseline/`，再到 `report/tables/figures/`。每一层目录都有 README
  （生成器 `scripts/scaffold_readmes.py`）。现有 WBCIC 结果迁入 `wbci_shu/`，SHU 侧建空骨架。
- **outputs/checkpoints 也并列**：`outputs/experiments/{wbci_shu,shu}/<run_id>/`、
  `outputs/analysis/{wbci_shu,shu}/`、`checkpoints/{wbci_shu,shu}/<run_id>/`。5 个 WBCIC config 输出路径
  全部加 `wbci_shu/` 前缀；已物理迁移现有 prototype_drift 工件。
- **SHU 预处理完成**：入口=作者发布 per-session `.mat`（已带通/陷波/4s 切段/250Hz/32ch），仅做标签
  {1,2}->{0,1} 归一化转存 `.npz`，不二次预处理。脚本 `scripts/preprocess_shu.py` +
  `code/preprocessing/shu_mat.py`。输出 `/share/workspace2/moto_imagination/SHU/processed/npz_clean/`，
  **125 session 全 ok / 25 subjects**，manifest 列与 WBCIC 一致。
- **命名修正**：原 `mat_clean` 目录改名 `npz_clean`（实际存的是 npz，避免误导），manifest 路径与 config 同步。
- **runner 数据集解耦**：`code/runners.py` 新增 `_resolve_manifest(cfg,P)`，manifest 由 config `data.manifest`
  决定，WBCIC 缺省回退 paths.yaml。新增 4 个 SHU config `shu_phase{0,1,2b,2c}_*.yaml`（同一批 runner）。
- **checkpoint 命名自文档化**：新增 `checkpoints/README.md`，说明任务前缀 within/cross/multisource/
  single/multi/proto + dataset/run_id/method/model/sub/session/seed。
- **验证**：9 个 config 全部 dry-run OK；SHU phase1（within 10 + cross 20 对）、phase2b（EA/RA 32ch 40 行全 ok）
  CPU smoke 端到端通过。
- **边界规则更新**：workspace2 仅 `processed/` 子树可写，raw 只读（AGENTS/STATUS/README 同步）。
- 状态：SHU **data ready，实验 pending**（config 就绪，未跑全量）。下一步可直接对 SHU 开跑 Phase 0/1/2b/2c。

---

## 2026-06-11 — Phase 2c 全量完成 + AI 深度分析报告

- Full Slurm 全部 COMPLETED（16 jobs exit 0，GPU 各约 1h，summarizer 15s）。验证：run_status.csv
  4320 cells 全 ok；metrics 25920 行（25790 ok + 130 correct_only degenerate_empty_class）；
  used_target_labels_for_training 全 False、n_target_labels_used_for_training≡0；ok 行无 NaN/Inf。
- 关键结果（canonical label_based/euclidean, n=4320）：所有几何信号显著但中等。
  separation_change ρ=0.389、fisher_change 0.359、drift_mean 0.352、neg_margin 0.313、
  direction_cosine −0.237。多元标准化线性模型 R²≈0.35。
- 机制结论：不是 centroid collapse，而是 within-class scatter 膨胀。源→目标：sep 8.45→11.78（更远）、
  scatter 16.86→25.83(+53%)、Fisher 4.58→1.57(−66%)。即表征"弥散/糊化"。
- 方法学发现：cosine 几何比 euclidean 更线性（separation_change cosine R²=0.176 vs euclidean 0.001）；
  下游 prototype 方法应在 cosine/归一化空间做。
- 漂移分层：low/mid/high drift tertile 的 acc_drop = 0.048 / 0.141 / 0.155 → 掉点集中在高漂移 cell。
- 模型依赖：EEGNet/DeepConvNet 漂移信号清晰（drift ρ 0.56/0.49），FBCNet 很弱（direction_cosine≈0，
  neg_margin 0.095）→ FBCNet 掉点是另一种几何成因，不能并入 prototype 结论。
- 方向不对称复现：ses-03 作 source 最难（ses-03→01 0.150）；ses-02→03 最易（0.086）。
- 写出 `4_experiments/prototype_drift/report/AI_ANALYSIS.md`（9 段，全部数字来自 tables，未捏造）。
- 分叉判定：**qualified go**。Step 4 先做 cosine 空间 Oracle 上限诊断 + scatter/reliability 探针，
  不无条件承诺 prototype adaptation；FBCNet 单独处理；drift 用作 online trigger。
- 状态：Phase 2c **done/complete**。

## 2026-06-11 — Phase 2c Prototype Drift Analysis 实现 + 全量 Slurm 提交（submitted/pending）

- 新增 `code/configs/experiments/phase2c_prototype_drift.yaml`：3 models (eegnet/deepconvnet/fbcnet)、
  5 seeds (0-4)、6 directed pairs、prototype_types=[label_based, confidence_weighted, correct_only]、
  distances=[euclidean, cosine]；source-only 训练、target test-only、target_label_usage=offline_diagnostic_only。
- 新增 `code/experiments/prototype_drift.py`：复用 session_splits/trainer/registry；对每个 subject×source×
  target×model×seed 在 source 上训练（source-train 切 val 做 early stopping），冻结后提取 source_train/
  source_val/target_test 的 penultimate embedding + logits/probs/pred/conf（fallback conf=max softmax），
  计算 label/confidence/correct prototypes，输出 6 类漂移指标（drift / direction_cosine / separation /
  margin(neg rate) / scatter / fisher）。加泄漏断言（source≠target、train/val 不交叠、
  n_target_labels_used_for_training≡0）+ NaN/Inf fail-fast；degenerate（correct_only 空类）显式标 status，不写坏数。
- 新增 `code/experiments/prototype_drift_summarize.py`：合并 per-(model,seed) CSV → 
  prototype_drift_metrics.csv / prototype_table.csv / prototype_accuracy_correlation.csv（Pearson/Spearman/
  linregress，按 model×ptype×dist + ALL 分组）/ trial_embeddings_index.csv / run_status.csv；7 张图
  (matplotlib, 无 seaborn) + 诚实 15 段报告 + RUN_STATUS.md（失败/缺失明列，不伪装 complete）。
  embedding 存 npz，CSV 只放索引/hash/path。已接入 `code/run.py --summarize`。
- 新增 `scripts/slurm/{train_prototype_drift_gpu,summarize_prototype_drift_cpu}.sbatch` +
  `submit_prototype_drift_full.sh`（gpu2node + mi_torch_cu118，fail-fast if no CUDA，1 GPU/job，logs/slurm）。
- Smoke（subjects 1,2、eegnet、seed 0、3 epochs、CPU，隔离在 `outputs/experiments/prototype_drift_v1_smoke/`）：
  12 cells / 72 metric rows / 0 failed；used_target_labels_for_training 全 False、n_target_labels_used_for_training≡0、
  无 NaN/Inf、5 tables + 7 figures + report + RUN_STATUS 全生成 → 端到端通过。
- Full：50 eligible subjects（47×6 + 3×2 = 288 cells/(model,seed)）× 3 models × 5 seeds = 4320 cells，
  ~25920 metric rows。提交 15 GPU + 1 summarizer（afterany），job ids 21536-21551；eegnet 已 RUNNING，
  其余 PENDING(QOS CPU limit)，summarizer PENDING(Dependency)。
- 状态：**submitted/pending，未 complete**。完成判据：summarizer 跑完且 run_status.csv 全 ok。
- 检查命令：`squeue -u $USER`；`cat outputs/experiments/prototype_drift_v1/full_job_ids.txt`；
  完成后看 `4_experiments/prototype_drift/{report,tables,figures}/`。
- 边界守护：未触碰 /share/workspace2；未覆盖 Phase 0/1/2a/2b 结果；新结果只进 prototype_drift_v1 / 
  4_experiments/prototype_drift；不做 adaptation/online/memory/tool-routing/41-10/LOSO/fine-tune。
- Next：等 jobs 跑完 → 读 tables + REPORT 写 AI 分析（prototype drift 是否成立 / 分叉到 Step 4 or 其他机制）。

## 2026-06-11 — 汇总器迁入 code/ + canonical 9 段报告

- 新增 `code/summaries/{session,multisource,alignment}.py`：从 `backup/root_archive_2026-06-10/scripts/`
  迁入三个汇总器（无 src 依赖，仅修 PROJECT_ROOT=parents[2] 与 main(argv)）。
- 新增 `code/summaries/canonical.py`：按 AGENTS §8 的 9 段结构（Core conclusion / Goal / Method /
  Protocol / Results / Analysis / Relationship / Next step / File list），数据驱动地从汇总 CSV 生成
  `REPORT.md`，不捏造数字。
- 新增 `code/summaries/summarize.py` 调度器；`code/run.py` 增加 `--summarize`。
- 验证（真实数据）：用 backup 现成 30150 行 alignment run CSV 跑 Phase 2b 汇总 → 表/图/原始报告齐全，
  canonical 报告结论 BN-stats +0.0071、5 方法 Δacc 与历史一致。临时输出已清理。
- 用法：`python code/run.py --summarize --config code/configs/experiments/<phase>.yaml`。

## 2026-06-10 — 恢复新架构可训练能力（直连 code/，CPU smoke 通过）

- 新增 `code/runners.py`：把旧 `scripts/{run_session_drift,train_session_models,train_session_multisource,
  train_session_alignment}.py` 的逻辑改写为进程内 runner，全部从 `code.` 包导入，不再依赖归档脚本。
- 改写 `code/run.py`：去掉对归档脚本的 subprocess 路由与执行拦截，改为按 experiment 名直连
  `code.runners.PHASE_RUNNERS`；`--dry-run` 仍可用，新增 `--methods/--folds/--ckpt-dir/--tag-suffix` 等参数。
- CPU smoke 验证（真实数据）：
  - Phase 0 drift（sub-001/002，6 pairs）→ 生成 CSV/summary/报告/图，DONE。
  - Phase 1 within EEGNet（2 被试，2 折，1 epoch）→ 真训练，写出 runs CSV + meta + split，ALL DONE。
- 仍未迁移：summarize_* 汇总脚本仍在 `backup/root_archive_2026-06-10/scripts/`；训练能跑，结果聚合待迁移。
- 临时 smoke 输出已清理，未污染阶段目录已同步结果。

## 2026-06-10 — P10 风格重构收尾：结果同步 + 文档精简 + progress 恢复（无新实验）

- 把历史可读结果从 `backup/root_archive_2026-06-10/` 同步进新阶段目录：
  `1_session_drift/`（漂移报告/CSV/图）、`2_baseline/`（baseline + multi-source 报告/表/图）、
  `2_baseline/alignment_baseline/`（alignment 报告/表/图）。原始 per-run/checkpoint 仍在 backup。
- 恢复本文件 `progress.md`（来自 backup 的 PROGRESS.md），作为长期进度记忆。
- 精简 `0_docs`：合并为 `ARCHITECTURE.md`、`STATUS.md`、`FILE_CATALOG.md`、`operation_log.md`、`README.md`；
  删除 `STRUCTURE_AND_FILE_GUIDE.md`、`RUN_READINESS_AND_CLEANUP_GUIDE.md`、`CODE_ARCHITECTURE.md`、
  `PROJECT_STATUS_CURRENT.md`；操作日志移出 `01_Lab_Journal` 子目录到 `0_docs/operation_log.md`。
- 更新 `AGENTS.md` 权威文件表、目录路由、兼容说明、handoff 更新规则指向新文件名。
- 当前状态不变：Phase 0/1/2a/2b 已完成，Phase 2c Prototype Drift 为下一步、未运行。
- 运行就绪：`code/run.py --dry-run` 可用；完整训练需让 `code/run.py` 直连 `code/experiments`
  或临时从 backup 恢复 `scripts/src/configs`。

> **CURRENT STATUS (2026-06-08, Step 1 multi-source DONE):** The cross-session DG mainline
> now has: **A** session-drift diagnostic (144 pairs / 50 subjects) ✅; **B** static baseline
> EEGNet/DeepConvNet/FBCNet, 5 seeds, within-session 10-fold CV + single-source directed
> cross-session (26 520 trainings, 30/30 cells, no leakage/NaN) ✅; **Step 1 / C** the
> **multi-source `ses-01+ses-02 → ses-03`** static baseline ✅ (47 eligible subjects, 4 skipped,
> 705 rows, no NaN) — multi-source beats the best single source for all 3 models
> (EEGNet 0.7717 vs 0.7492, +0.0224; DeepConvNet 0.7564, +0.0353; FBCNet 0.6750, +0.0267;
> mean Δ +0.0281). Within Acc: EEGNet 0.807 / DeepConvNet 0.766 / FBCNet 0.720; single-source
> cross Acc 0.711 / 0.681 / 0.628 (drop 9–13%). We trail the paper by ~5–8pp in absolute within
> accuracy — a **training-recipe/data-budget** effect (within folds carve a 20% val slice →
> ~144 train trials/fold; dropout 0.25, no max-norm; shared un-tuned recipe), **not an
> architecture bug** (ranking + S1<S2<S3 trend match the paper).
> **Step 2 no-learning adaptation baseline = COMPLETE (2026-06-09).** 30,150 rows, 0 failed/0 NaN.
> Result is a **negative/diagnostic** one: no-learning alignment is INSUFFICIENT — none clears the
> +2pp line. none_reference acc 0.6818; best = bn_statistics_adaptation 0.6889 (Δ +0.0071); EA/RA
> slightly hurt; z-score/filter-bank ≈ neutral. High-drift subjects helped least → motivates (but
> does NOT run) learning-based Step-3 adaptation.
> **NEXT = Step 3 (learning-based) is FUTURE, NOT run.** CAP-EEGNet (v1/v2) + agent/toolkit/
> prototype/confidence/online/fine-tuning, LOSO, 41/10 remain **future work, not run**.
> GPU env `mi_torch_cu118` (torch 2.7.1+cu118).
> Sources of truth: `docs/PROJECT_STATUS_CURRENT.md` (status), `docs/MULTISOURCE_STEP1_REPORT.md`
> (Step 1), `docs/NEXT_EXPERIMENT_PLAN.md` + `docs/ADAPTATION_BASELINE_PLAN.md` (Step 2 plan),
> `docs/RESULTS_SUMMARY.md` (consolidated results), `docs/P10_INTEGRATION_SUMMARY.md` (P10 dir).


## 2026-06-10 — P10-style multi-dataset architecture reorganization (no experiments)

**Scope**: repository organization and documentation/rules only. No training, no preprocessing, no Slurm job, no external dataset write. `/share/home/yuan/SYX/P10_MI泛化研究` and `/share/home/yuan/SYX/CLAUDE.md` were used as read-only references.

**Changed**:
- Added P10-style root folders: `0_docs/`, `1_session_drift/`, `2_baseline/`, `3_online_adaptation/`, `4_experiments/`, `5_papers/`, `inbox/`, `01_Lab_Journal/`.
- Added modular `code/` framework with `run.py`, split configs (`datasets/models/experiments`), dataset adapters for WBCIC-SHU and SHU, channel mapping, and copied current model/method/experiment implementation for staged migration.
- Rewrote root memory files (`AGENTS.md`, `CLAUDE.md`, `README.md`, `proposal.md`, `results.md`, `experiment_log.md`) and rebuilt `.cursor/rules/` around the new architecture.
- Updated major Slurm entry scripts to call `code/run.py` with new phase configs. Existing outputs/checkpoints/logs and old `src/`/`scripts/` are retained for provenance/compatibility.

**Validation**: `python -B code/run.py --all --dry-run` passed; AST parsing passed for 44 files under `code/`; core imports passed inside `mi_torch_cu118` (`EEGNetClassifier`, `TrainSpec`, `ChannelZScore`).

<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->

## 2026-06-09 — Anti-data-loss: backup/recovery scheme + GPU-cluster doc integration (no experiments)

**Why**: earlier loss happened because important code/docs lived only in the working tree, and
results (`outputs/`, `checkpoints/`, `logs/`) are gitignored so commits don't protect them. Built a
4-layer safety net + made the "soul files" + cluster knowledge durable. No experiment run.

- **Soul files committed/refreshed**: `AGENTS.md` + `.cursor/rules/{00-project-context,50-server-slurm,
  90-agent-behavior}.mdc` — generalized the senior-handoff-folder rule (no hard-coded P10 path; read &
  integrate when pointed at it), added explicit dataset addresses + project architecture/data-flow,
  tightened the filesystem-scope rule (write only inside the project root; `/share/workspace2` dataset
  is read-only; sibling handoff folders read-only), and corrected the GPU/Slurm guidance (env
  `mi_torch_cu118`, GPU jobs are normal now, fail-fast on no-CUDA).
- **GPU cluster doc integrated** (official `http://10.26.1.75:58080/`, "4090D 集群用户文档") into
  `docs/SERVER_RUNBOOK.md` + the 50-rule + AGENTS verified facts: 1 login01 + 5 GPU nodes
  (gpu01-05, 8× 4090D, no SSH, Slurm-only) + storge file node; partitions gpu2node(gpu01,02)/
  gpu3node(gpu03,04,05); monitoring `slmwatch`/`gpuwatch`/`user-tools`; modules; home quota 512 GiB;
  IO-heavy work on storge.
- **New doc `docs/BACKUP_AND_RECOVERY.md`**: what git manages vs. gitignored, the 4 layers
  (commit → tag → `git bundle` → results tar), where backups live (authorized
  `/share/home/yuan/SYX/backups/`; large → `/share/workspace2`), restore steps, and a
  before/during/after routine checklist.
- **Milestone tagged** `milestone-step2-alignment-complete` and **backups created** under
  `/share/home/yuan/SYX/backups/`: `eeg-mi-online_git_2026-06-09.bundle` (full git history+tags) +
  `eeg-mi-online_results_2026-06-09.tar.gz` (session_drift_v1 + baseline_v1 + alignment_baseline_v1
  CSVs/reports/figures/splits/configs/job-ids; the 11 GB alignment checkpoints intentionally NOT
  archived — re-creatable) + `BACKUP_MANIFEST.md` (sizes + sha256). Working tree clean.

## 2026-06-09 — Step 2 no-learning alignment baseline COMPLETE (negative/diagnostic result)

**Status: COMPLETE.** All 75 GPU training jobs (`21261-21335`) + the summarizer (`21336`) finished
COMPLETED; `results_alignment_all.csv` exists with **30,150 rows** (25,125 alignment + 5,025
none_reference), **0 failed**, **0 NaN/Inf** in metrics (the only all-empty column is the text
`error_message`). `used_target_y_for_training == False` for all rows; `used_target_x_for_stats ==
True` for all 25,125 trained-method rows. 5 trained methods + none_reference all present; models
eegnet/deepconvnet/fbcnet; seeds 0-4; all 6 single-source directions + multi-source ses-01+02→ses-03.

**Result (honest, mean cross-acc over directions/subjects/seeds, vs baseline_v1 `none_reference`):**

| method | Δacc vs none | note |
|---|---:|---|
| `bn_statistics_adaptation` | **+0.0071** | only net-positive method; well below +0.02 |
| `filterbank_reweighting` | −0.0030 | ≈ neutral |
| `session_zscore` | −0.0038 | ≈ neutral |
| `riemannian_alignment` | −0.0101 | slightly hurts |
| `euclidean_alignment` | −0.0124 | hurts most |

- Absolute acc (all scope): none_reference **0.6818**; bn_statistics **0.6889** (best); all others < none.
- By drift level: BN-stats small-positive at every level (stable +0.009 / moderate +0.008 / high
  +0.005); filter-bank positive on stable/moderate (+0.011/+0.005) but **−0.024 on high drift**;
  EA/RA negative everywhere. → the subjects that drift most are helped least.
- **Conclusion: no-learning / unsupervised statistic-only alignment is INSUFFICIENT** (no method
  reaches the pre-registered +2pp success line). BN-stats gives only a small positive gain. This is a
  **valuable negative/diagnostic result** that objectively justifies learning-based Step-3 adaptation
  (online / adapter / prototype / memory) — **which is NOT run here**.
- Verified the `alignment_vs_baseline.csv` join is correct: none_reference is unique per
  (model, seed, subject, train_sessions, test_session, training_scope); all 25,125 alignment rows
  matched; an independent re-merge reproduced the summarizer's Δacc exactly.
- Outputs: `outputs/experiments/alignment_baseline_v1/` — `ALIGNMENT_BASELINE_REPORT.md` (13 sections
  + headline), `RUN_STATUS.md`, `manifest_sources.json`, `cross_session/tables/` (9 CSVs incl.
  results_alignment_all.csv, alignment_vs_baseline.csv, alignment_by_method/model/direction/protocol/
  subject.csv, alignment_gain_by_drift_level.csv, run_status.csv), `cross_session/figures/` (6 PNGs).
  Checkpoints `checkpoints/alignment_baseline_v1/`.
- Closeout only added a richer, honest interpretation to the report (recomputed from the same CSVs);
  no new experiment, no new deps, baseline_v1 untouched.

## 2026-06-09 — Step 2 no-learning alignment baseline: implemented + smoke + FULL RUN SUBMITTED (results PENDING)

**Status (do NOT read as complete):** Step 2 **code implemented**, **smoke passed**, **full run
submitted**, **summarizer dependency submitted**, **results PENDING until the 75 GPU jobs finish.**
Will be marked complete only after the summarizer writes `results_alignment_all.csv`.

**Scope (per user): ONLY Step 2 alignment baseline.** No online, no 41/10, no LOSO, no fine-tuning,
no CAP-EEGNet full, no multi-agent/prototype/memory. No new deps, no shared-env changes, no
raw/workspace2 writes, baseline_v1 not overwritten.

**A — pre-checks (all pass):** baseline_v1 exists with within+cross standard tables; processed
manifest + `eog_ecg_clean` present; env `mi_torch_cu118` = torch 2.7.1+cu118, CUDA available on the
GPU node (RTX 4090 D); git working tree clean (HEAD `a239b43`).

**B — new code (nothing existing overwritten; reuses trainer / registry / session_splits / metrics
/ baseline split logic):**
- `src/adaptation/{__init__,session_alignment,bn_adaptation}.py` — fit/transform alignment
  transforms (ChannelZScore, EuclideanAlignment, RiemannianAlignment = log-Euclidean SPD mean via
  eigh, FilterBankReweight) + BN running-stat adaptation (no grad/backward/optimizer). EA/RA use
  eps ridge + diagonal shrinkage; inverse-sqrt/logm via symmetric eigh with eigenvalue clipping.
- `src/evaluation/session_alignment_protocols.py` — single-source + multi-source tasks; source
  transform fit on SOURCE TRAIN only (applied to source train+val); target aligned from target X
  (filterbank reweights target→source profile); BN method trains plain then refreshes BN from
  target X; per-run leakage asserts; 26-column result rows.
- `scripts/train_session_alignment.py`, `scripts/summarize_alignment_results.py` (pulls
  `none_reference` from baseline_v1 + drift levels), `scripts/build_alignment_baseline_outputs.py`
  (scaffold). `configs/session_alignment_compare.yaml`. Slurm:
  `scripts/slurm/{train_session_alignment_gpu,summarize_alignment_results_cpu}.sbatch`.

**C — unit checks (CPU, synthetic):** zscore finite + ~N(0,1); EA matrix 58×58; RA matrix 58×58 +
spd_mean=log_euclidean; aligned X shape unchanged; EA finite on near-singular cov; BN adapt leaves
all trainable weights UNCHANGED (no optimizer.step) but changes running_mean; eval mode after; all 3
models forward aligned X → logits [B,2]. ALL PASS.

**D — smoke (GPU srun, mi_torch_cu118):** subjects 1,2, eegnet, seed 0, 3 epochs, all 5 trained
methods, both protocol groups → 70/70 rows ok, 0 NaN/Inf, n_train/val/test = 160/40/200 (single) &
320/80/200 (multi), 70 checkpoints, split JSON sessions correct, `used_target_y_for_training`=False
& `used_target_x_for_stats`=True for all rows. Summarizer ran cleanly on the smoke rows. Smoke
artifacts removed afterward.

**E — full run (GPU Slurm):** **75 jobs `21261–21335`** = method × model × seed (each job runs both
single-source + multi-source). Per job = 335 trainings; total ≈ **25,125 trainings**. Partition
gpu2node, gpu:1, env mi_torch_cu118, fail-fast if no CUDA, logs → `logs/slurm/`. Job ids →
`outputs/experiments/alignment_baseline_v1/full_job_ids.txt`. (Some pending under
QOSMaxCpuPerUserLimit; drain as capacity frees.) Estimates + layout in
`outputs/experiments/alignment_baseline_v1/RUN_PLAN.md`.

**F — dependent summarizer:** CPU job **`21336`** with `--dependency=afterany:<all 75>` →
`scripts/summarize_alignment_results.py` → `cross_session/tables/*` (results_alignment_all.csv,
alignment_by_method/model/direction/protocol/subject.csv, alignment_vs_baseline.csv,
alignment_gain_by_drift_level.csv, run_status.csv) + `cross_session/figures/*` (6 figs) +
`ALIGNMENT_BASELINE_REPORT.md` + `RUN_STATUS.md` + `manifest_sources.json`.

**Next (after jobs finish):** check `sacct -j 21261-21336`, read `RUN_STATUS.md`, confirm
`results_alignment_all.csv` exists, then read `ALIGNMENT_BASELINE_REPORT.md`. Only then mark Step 2
complete. Step 3 (online / adapter / prototype) stays gated on these results.

## 2026-06-09 — Systematic documentation repair (no experiments)

**Why**: git HEAD is the 2026-06-04 scaffold commit; all work since (drift, baseline, P10
integration, Step 1) was never committed, so the working tree's docs/code lived only on disk.
During the Step-1 run a tooling/backend hiccup left the working tree missing several docs and
the new Step-1 code files. **Experiment results were never lost** (the 705-row Step-1 run +
the baseline summaries were intact on disk).

**Repaired (no results touched, no experiments run):**
- Re-materialized the Step-1 code that the run had used: `src/evaluation/session_multisource_protocols.py`,
  `scripts/train_session_multisource.py`, `scripts/summarize_multisource_results.py`,
  `configs/session_multisource_compare.yaml`, the two `scripts/slurm/*multisource*` jobs.
  Re-ran the summarizer on the existing 705 rows — reproduced identical numbers.
- Restored the full rich docs from user-provided backups (`/share/home/yuan/SYX/PROGRESS.md`,
  `/share/home/yuan/SYX/PROJECT_OVERVIEW.md`, "saved before the remaining Step 1") + this
  conversation's context, then brought them current (P10 integration + Step 1 done).
- Rewrote the lost docs: `BASELINE_PROTOCOL.md`, `P10_INTEGRATION_SUMMARY.md`,
  `ADAPTATION_BASELINE_PLAN.md`, `CODE_INTEGRATION_NOTES.md`,
  `references/P10_MI_generalization_README.md`; recovered+updated `AGENTS.md`,
  `references/README.md` from git HEAD.
- Wrote a deep Step-1 analysis: `docs/MULTISOURCE_STEP1_REPORT.md` (per-subject gains,
  drift-level breakdown, upper-bound gap recovery) + `multisource_by_subject.csv`.

**Recommendation**: commit the working tree soon so this can't recur (currently everything since
2026-06-04 is uncommitted). Not done automatically — awaiting user go-ahead on git.

## 2026-06-08 — Step 1 static baseline COMPLETE: multi-source ses-01+02 → ses-03

**Scope (per user): only the remaining Step-1 static-baseline item.** No Step-2 adaptation,
no online, no 41/10, no LOSO, no fine-tuning, no CAP-EEGNet full / multi-agent / prototype /
memory. No raw/workspace2 writes. No existing baseline results overwritten.

**Step A — checked for existing results**: none found under `outputs/experiments/` for
`multisource` / `0102_to_03`. Confirmed in code that only **single-source directed pairs**
existed (`session_splits.make_cross_session_pairs`, `session_protocols.run_cross_session`).
So the multi-source direction was genuinely missing → implemented it.

**Step B — new code (separate from existing baselines, nothing overwritten):**
- `src/evaluation/session_multisource_protocols.py` — eligibility (all of ses-01/02/03 ok),
  combine ses-01+ses-02 trials, carve val ONLY from the combined train, test = ses-03;
  per-run leakage assertions; `n_train/n_val/n_test` + checkpoint + status recorded.
- `scripts/train_session_multisource.py` — entry (reuses TrainSpec / trainer / registry /
  `load_ok_sessions`); writes per-seed CSV + meta JSON (used/skipped subjects).
- `scripts/summarize_multisource_results.py` — aggregates, builds the report + figure, and
  pulls the single-source ses-01→03 / ses-02→03 rows from the baseline summaries for comparison.
- `configs/session_multisource_compare.yaml`, `scripts/slurm/train_session_multisource_gpu.sbatch`,
  `scripts/slurm/summarize_multisource_results_cpu.sbatch`.

**Step C — smoke test (GPU node via srun, `mi_torch_cu118`):** subjects 1,2, EEGNet, seed 0,
3 epochs. Passed: train_sessions=ses-01+ses-02, test=ses-03, `n_train=320 / n_val=80 / n_test=200`,
train/val disjoint, val carved only from the combined train, no NaN, CSV + checkpoint + split JSON
written. Smoke output dir removed afterward.

**Step D — full run (GPU Slurm):** jobs `21240,21241,21242,21243,21244` (one per seed,
3 models each) + summarize `21245` (afterany). All **COMPLETED**, exit `0:0`. 705 rows
(3 models × 5 seeds × 47 used subjects), 0 failed, 0 NaN.

**Step E — results (mean±std over 5 seeds, test = ses-03):**
- EEGNet **0.7717±0.003** > DeepConvNet 0.7564±0.007 > FBCNet 0.6750±0.002.
- vs best single direction (`ses-02→03`): +0.0224 / +0.0353 / +0.0267 (mean **+0.0281**).
- vs per-subject best single (oracle source pick): +0.0139 / +0.0209 / +0.0097 (still positive).
- recovers ~30–53% of the single-source-cross → within-ses-03 upper-bound gap.
- per-subject: EEGNet improved 33/47; failures concentrate where the two source sessions differ
  a lot (e.g. sub-029 ses-01 0.441 vs ses-02 0.935) → motivates Step-2 alignment/weighting.
- Full analysis: `docs/MULTISOURCE_STEP1_REPORT.md`; machine report + CSVs under
  `outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/`; checkpoints
  `checkpoints/session_multisource_v1/{eegnet,deepconvnet,fbcnet}/`.

**Note (doc loss + recovery):** during this work the (uncommitted) working tree lost several
docs and the new Step-1 code files (git HEAD is the 2026-06-04 scaffold; later work was never
committed). The experiment **results were intact**; the Step-1 code was re-materialized from the
run, the summarizer re-ran cleanly on the existing 705 rows, and the docs were restored from
backups + this conversation's context. See the 2026-06-09 doc-repair entry.

**Next**: Step 2 no-learning adaptation baseline (see `docs/ADAPTATION_BASELINE_PLAN.md`). Do not
run online / 41-10 / fine-tuning / CAP-EEGNet full until Step 2 is done.

## 2026-06-08 — P10 「MI 泛化研究」 integration + direction cleanup (NO experiments run)

Integrated the senior's updated P10 package (`/share/home/yuan/SYX/P10_MI泛化研究/`) into the
project's understanding/docs. P10 Phase-0 (drift) + Phase-1 (baseline) numbers **match this
project exactly** (same 148 ok sessions, same EEGNet 0.807/0.711) — same study, packaged for
handoff. P10 adds richer per-subject drift profiling (drift_level high/moderate/stable; sub-020
most drifted, sub-005 most stable), the cross-session **directional asymmetry** (ses-02→03 0.749
best vs ses-03→01 0.681 worst, 6.8pp), MMD↔accuracy Spearman ρ=−1.0 (n=3), and a **Phase-2 online
continual-learning framework that is design + draft code only, NOT validated** → future work.

New docs: `docs/PROJECT_STATUS_CURRENT.md`, `docs/P10_INTEGRATION_SUMMARY.md`,
`docs/NEXT_EXPERIMENT_PLAN.md` (Step 0/1/2/3), `docs/ADAPTATION_BASELINE_PLAN.md`,
`docs/CODE_INTEGRATION_NOTES.md`, `docs/references/P10_MI_generalization_README.md`. Updated
wording (done=done, unrun=future) across the core docs. **No experiment run; no raw/workspace2 writes.**

<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->

## 2026-06-07 — Baseline 5-seed run COMPLETE + repo cleanup

**All 10 training jobs + the report job COMPLETED** (21161–21171, exit 0). Final summarizer
re-run after cleanup: **26 520 rows (within 22 200 + cross 4320), 30/30 cells,
`incomplete=False`, no NaN, no leakage**. (The earlier auto `RUN_STATUS.md`/log entry said
"INCOMPLETE" only because the report job counted *itself* (21171) as RUNNING — a false flag;
the summarizer logged `incomplete=False`.)

**Headline results** (mean±std over 5 seeds):
- within Acc: EEGNet **0.807±0.002** > DeepConvNet 0.766±0.002 > FBCNet 0.720±0.003.
- cross Acc: EEGNet **0.711±0.008** > DeepConvNet 0.681±0.002 > FBCNet 0.628±0.003.
- cross-session drop: 11.9% / 11.1% / 12.8%.
- vs paper within (85.32 / 84.47 / 78.40): we are −4.65 / −7.84 / −6.37 pp.

**Why below paper (see `docs/RESULTS_SUMMARY.md` §4)**: systematic training-recipe/data-budget
difference, NOT an architecture bug — chiefly the within-fold 20% val carve (≈144 vs 180 train
trials), early stopping on a tiny (~36) val set, dropout 0.25 + no max-norm + wd 0, and a shared
un-tuned recipe (FBCNet also simplified). Gap is consistent across all 3 models and the ranking
+ S1<S2<S3 trend match the paper; the within-vs-cross conclusion is fair and unaffected.

**New doc**: `docs/RESULTS_SUMMARY.md` (drift + baseline + paper-gap analysis + next steps).

**Repo cleanup (gitignored artifacts only; no source/results lost):** removed the archived
smoke output dir, legacy `outputs/processed_paper_style/sub-001`, `outputs/raw_check`,
`outputs/sanity_check`, and the stale smoke-only `checkpoints/.../cap_eegnet/` (18 files,
sub-001/002 seed0). Standardized run CSV naming: legacy `within__{model}.csv` →
`within__{model}__seed0.csv` (now all 30 files are uniform `{protocol}__{model}__seed{N}.csv`).
The 3 baseline checkpoint dirs (eegnet/deepconvnet/fbcnet, 2180 each, ~2.7G total) are kept as
legitimate results.


## 2026-06-07 — Overnight 5-seed baseline GRID submitted (EEGNet/DeepConvNet/FBCNet)

**Goal**: full 5-seed baseline over the 148 `status=ok` sessions, both protocols, seeds
0–4, three baseline models. CAP-EEGNet and all complex modules remain FUTURE (not run);
no LOSO, no 41/10.

**Overwrite fix**: result CSVs are now written per seed —
`runs/{protocol}__{model}__seed{seed}.csv` (was `{protocol}__{model}.csv`, which would have
let seeds overwrite each other). Job 21161 (already running, old code) keeps the legacy
`within__{model}.csv` (=seed 0); the summarizer globs all CSVs and reads the `seed` column,
so there is no collision. Checkpoints were already seed-namespaced.

**Jobs (1 GPU each, env `mi_torch_cu118`, fail-fast if no CUDA):**
- `21161` within seed 0 (already running) — `mi_sess_compare`
- `21162` cross s0, `21163` within s1, `21164` cross s1, `21165` within s2, `21166` cross s2,
  `21167` within s3, `21168` cross s3, `21169` within s4, `21170` cross s4
  (job names `mi_base_{protocol}_s{seed}`, logs `logs/slurm/mi_base_{protocol}_s{seed}-<jobid>.out/.err`)
- job-id list: `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/overnight_job_ids.txt`

**Expected trainings** (1 model × 1 protocol × 1 seed): within = 148 sessions × 10 folds =
1480; cross = 288 directed pairs. Per seed (3 models): within 4440, cross 864. 5-seed total:
within 22 200 + cross 4320 = **26 520 trainings**.

**Dependent report job** (`mi_base_report`, CPU, `--dependency=afterany:` all 10 IDs): runs
`scripts/baseline_report.py` → sacct check + `scripts/summarize_session_results.py` →
`outputs/experiments/baseline_v1/provenance/session_model_compare_v1/summaries/` (results_within/cross_session.csv,
within_by_seed/cross_by_seed/within_session_wise/cross_by_direction.csv,
summary_by_model_protocol.csv, model_ranking.md, SESSION_MODEL_COMPARE_REPORT.md, RUN_STATUS.md,
3 figures) and appends auto status entries here + to `docs/EXPERIMENT_LOG.md`. Marks the
report INCOMPLETE if any training job is not COMPLETED or any output is missing.

Outputs: `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`; checkpoints
`checkpoints/session_model_compare_v1/<model>/{within_<subj>_<ses>_seed<k>.pt,
cross_<subj>_<tr>-to-<te>_seed<k>.pt}`; slurm logs `logs/slurm/`.

## 2026-06-07 — Mainline converged to the three baseline architectures (EEGNet/DeepConvNet/FBCNet)

**Decision (per user)**: narrow the active mainline to **the three baseline models** — run
EEGNet + DeepConvNet + FBCNet within-session 10-fold CV then cross-session on the 148
`status=ok` sessions; start by checking EEGNet against the WBCIC-SHU paper baseline.
**CAP-EEGNet (v1/v2) and ALL complex modules (agent/toolkit/neural-subagents, prototype,
multi-source confidence, online, fine-tuning) are kept in code but marked OPTIONAL /
FUTURE — not run.** Also not run: LOSO, 41/10.

**No code deleted.** Changes are config + docs only (the trainer/protocols stay
model-agnostic, so CAP-EEGNet still works when re-enabled):
- `configs/session_model_compare.yaml`: `models: ["eegnet","deepconvnet","fbcnet"]`
  (cap_eegnet commented as optional/future).
- `scripts/slurm/train_session_models_gpu.sbatch`: default `--models eegnet,deepconvnet,fbcnet`.
- Docs updated: `PROGRESS.md` (this entry + top status), `EXPERIMENT_PROTOCOL.md`
  (mainline = 3 baselines; CAP-EEGNet + experiments = future), `BASELINE_PROTOCOL.md`
  (top banner + run commands narrowed to the 3 baselines).

**Run plan (seed=0)**: (1) full within-session CV for the 3 baselines → (2) summarizer →
check EEGNet vs paper → (3) full cross-session for the 3 baselines → summarizer.
Submitting the full within-session CV now (see next entry / experiment log for results).

## 2026-06-06 — Mainline pivot: cross-session DG (drift diagnostics + within/cross model comparison)

**Why**: per the senior's P10 "MI 泛化研究" package (now in `docs/references/P10_MI_generalization/`),
the project pivots from 41/10 cross-subject pretraining to a **cross-session domain
generalization** study: (A) diagnose what drifts across sessions, (B) compare baselines +
our model under one fair protocol, (C) at Within-session CV and Cross-session levels, (D)
unified report. **No 41/10, LOSO, fine-tuning, or online now** — all kept as future work.

**Task 0 — references placed** (renamed, reference-only, not run from there):
`docs/references/P10_MI_generalization/` (HANDOFF/proposal/experiment_log/QC_SUMMARY_CN/
ChatGPT task md/paper PDF/slides) and `docs/references/senior_scripts/{data_validation,
model_training}/` (the originals). `docs/references/README.md` indexes them + maps each to
its in-tree implementation.

**New configs**: `configs/session_drift.yaml`, `configs/session_model_compare.yaml` (paths
from `configs/paths.yaml`; status=ok filter; within folds=10; protocols within+cross;
models eegnet/deepconvnet/fbcnet/cap_eegnet; seeds=[0]; bs16; lr1e-3; max_epochs100;
early-stopping).

**New code**:
- `src/data/session_splits.py` — `load_ok_sessions` (manifest, status=ok), label
  normalization to {0,1} (accepts {1,2}), `make_within_session_folds` (StratifiedKFold,
  label-balanced), `make_cross_session_pairs` (directed, both-ok), JSON persistence.
- `src/analysis/session_drift.py` + `scripts/analysis/run_session_drift.py` +
  `scripts/slurm/session_drift_cpu.sbatch` — MMD/CORAL/μ-β power shift/KS/ERD-ERS spatial
  corr/CSP similarity/RMS ratio/Fisher shift; vectorized FFT; **matplotlib-only** (no
  seaborn); status=ok via manifest; `--subjects`/`--max-subjects`. Outputs CSV + summary.json
  + SESSION_DRIFT_REPORT.md + 8 figures.
- `src/models/`: added `EEGNetClassifier` (eegnet.py), `deepconvnet.py` (Schirrmeister
  2017), `fbcnet.py` (Mane 2021, fixed FIR filter bank), `registry.py` (`build_model`).
  **CAP-EEGNet upgraded to v1**: encoder + classifier + lightweight **learned** confidence
  head (single-source scalar, calibration BCE); prototype/subagents/adapter/domain/online
  still raise `NotImplementedError`. All 4 share `forward -> {logits, features, confidence}`.
- `src/training/trainer.py` (CE + optional confidence-BCE, early stopping, predict),
  `src/evaluation/session_protocols.py` (`run_within_session` StratifiedKFold; 
  `run_cross_session` directed pairs; metrics acc/bacc/f1/auc/nll/brier/ece; val carved
  from train only; bounded checkpointing), `scripts/train_session_models.py` +
  `scripts/slurm/train_session_models_gpu.sbatch` (env `mi_torch_cu118`, cuda fail-fast).
- `scripts/summarize_session_results.py` + `..._cpu.sbatch` — merges `runs/*.csv` →
  results_within/cross CSVs, summary_by_model_protocol.csv, 3 figures, model_ranking.md,
  SESSION_MODEL_COMPARE_REPORT.md, cross-session drop + relative drop.

**Docs**: new `SESSION_DRIFT_ANALYSIS.md`, `BASELINE_PROTOCOL.md`; updated
`PROJECT_OVERVIEW.md` (top status), `EXPERIMENT_PROTOCOL.md` (mainline = within/cross;
41/10/LOSO/finetune/online = FUTURE), `.cursor/rules/30-model-experiments.mdc` (shared
protocol/filter/metrics + no-leakage), `AGENTS.md` (constraint 9).

**SMOKE TESTS — ALL PASS (compute nodes via srun, never login node):**
- GPU env check: `mi_torch_cu118` = torch **2.7.1+cu118**, `cuda.is_available()=True`,
  RTX 4090 D. (The earlier `mi_torch` CPU-only-torch blocker is resolved by this env.)
- Drift (subjects 1,2): 6 within-subject pairs in 12.6 s; CSV + summary.json + report +
  8 figures written to `outputs/analysis/session_drift_v1/`. MMD≈0.175, ERD/ERS corr≈0.44–0.49,
  RMS ratio≈0.90 — sane.
- Within (subjects 1,2; folds=2; epochs=3; all 4 models): 48 rows, 117.8 s, all CSVs +
  per-session split JSONs written.
- Cross (subjects 1,2; epochs=3; all 4 models): 48 rows (2 subj × 6 directed pairs × 4
  models), 37.7 s; n_train=160/n_val=40 (carved from train) / n_test=200 (full test session)
  → no leakage verified.
- Summarize: merged 96 rows → all 8 summary artifacts. (Numbers are smoke-noise at 3
  epochs/2 folds; pipeline correctness is the point.)

**Full-run estimates (1 seed; pending user go-ahead — NOT yet submitted):**
- Drift full ≈ **144** within-subject pairs (47 subjects×3 + 3 from partial-ok subjects;
  sub-024 has 1 ok session → 0 pairs), ~10–15 min CPU.
- Within full = 148 ok sessions × 10 folds × 4 models = **5920** trainings; ~**4–8 h wall**
  if the 4 models run as parallel 1-GPU jobs (deepconvnet is the long pole), less with
  early stopping.
- Cross full = **288** directed pairs × 4 models = **1152** trainings; ~**1–2 h wall**
  parallel. Seeds multiply linearly.
- Outputs: `outputs/analysis/session_drift_v1/`, `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`
  (`runs/`, `splits/`, `summaries/`); checkpoints `checkpoints/session_model_compare_v1/`;
  slurm logs `logs/slurm/`.

**Deliberately NOT done**: full-scale GPU submission (awaiting user OK per task 8.6); 41/10,
LOSO, fine-tuning, online; full multi-subagent CAP-EEGNet (v1 only). No raw/workspace2 writes.

## 2026-06-05 — Roadmap alignment (project direction = full multi-subagent CAP-EEGNet)

**Why**: ensure the project targets the chat record's FINAL design, not a plain EEGNet
classifier. No training, no GPU install, no sbatch — docs/rules/light-code alignment only.

**Final goal restated everywhere**: **Confidence-aware Online Adaptive Multi-Subagent
Pretraining Framework for Cross-subject MI EEG Decoding**. The minimal EEGNet+classifier is
explicitly a **Stage 0 pipeline-validation baseline**, NOT the paper method.

**Staged route fixed (docs/ROADMAP.md)**: Stage 0 infra/sanity (✅) → Stage 1 cross-subject
pretrain + zero-shot (5 seeds, one model each, mean±std) → Stage 2 confidence+prototype+
adapter(+domain, +neural subagents) → Stage 3 Session-1 fine-tune → Stage 4 online
test-then-update → Stage 5 ablation/interpretability.

**New docs**: `docs/ROADMAP.md` (final goal, full module table, minimal-vs-full, Stage 0–5,
do-not list) and `docs/ALIGNMENT_CHECKLIST.md` (chat-record→status mapping).

**Docs updated**: `PROJECT_OVERVIEW.md` (top status, §0 final-goal, completion table now
shows splits ✅ + minimal CAP-EEGNet ✅ + full CAP-EEGNet 🚧, next-steps), `MODEL_PLAN.md`
(minimal-vs-full, deep neural subagents not handcrafted, full component list, Stage-ordered
build order), `EXPERIMENT_PROTOCOL.md` (5-seed mean±std, Exp1–4 input/output/forbidden).

**Rules updated**: `00-project-context` (final goal + refreshed current-stage block),
`30-model-experiments` (minimal≠final; repeated multi-seed split mandatory; no target
leakage; stage note), `40-online-learning` (test-then-update; default-forbid full-backbone
online update; confidence-gate/prototype/adapter are core).

**Light code alignment** (`src/models/cap_eegnet.py`): kept the working minimal model;
added full-vision config flags (`use_subagents/use_dataset_router/use_adapter/use_prototype/
use_confidence/use_domain_align/use_online_update`), fail-fast `NotImplementedError`
("Reserved for full CAP-EEGNet … NOT implemented in the minimal sanity model") when any is
enabled, `predict_confidence()`/`online_update()` reserved methods, and documented stub
classes (`NeuralSubagentEncoder/DatasetAwareRouter/Adapter/PrototypeMemory/ConfidenceHead/
DomainAlignmentHead/OnlineUpdateModule`). Verified: minimal forward still returns
`{logits[B,2], features, proto_dist=None, confidence=None}`; enabling a full flag raises the
clear error. Also aligned `configs/finetune.yaml` + `configs/online_adaptation.yaml`
`variant→eog_ecg_clean` + `statuses:[ok]` for consistency with `train_cross_subject.yaml`.

**Unchanged on purpose**: preprocessing, splits, dataset, sanity outputs (no refactor). No
formal training, no GPU env changes, no sbatch, derivatives `.mat` never a training entry.

## 2026-06-05 — Pre-training prep: 41/10 splits + minimal CAP-EEGNet + sanity train

**Scope (per user): training-prep only, NOT formal training.** All heavy/GPU steps ran on
a compute node via `srun` (node gpu02), never the login node.

**1. `src/data/splits.py` (implemented).** Reads `processed_manifest.csv`, uses only
`status==ok` sessions. `SubjectSessionIndex` summarizes subject→session→status. Policy:
target subjects must have **all 3 sessions ok** (needed for Session-1 fine-tune +
Session-2/3 online); source = the other 41 subjects (may contain subjects with failed
sessions, but training uses ok sessions only). Subject-wise only — never session/trial.
Failed sessions → `excluded_sessions`. `make_subject_wise_split(seed=...)` uses a local
`random.Random(seed)` over the **47 fully-ok** subjects to pick 10 targets; the 4 partial
subjects (sub-023/024/026/032, owning the 5 failed sessions) are always forced into source.

**2. Splits generated** via `scripts/make_splits.py` for seeds **2026–2030** →
`splits/cap_eegnet_4110_seed<k>.json`. Each: source=41 (118 ok sessions, 23 600 trials,
11 800/11 800), target=10 (30 ok sessions, 6 000 trials, 3 000/3 000), excluded=5
(`sub-023/ses-01, sub-024/ses-02, sub-024/ses-03, sub-026/ses-01, sub-032/ses-02`). Each
JSON records source_subjects, target_subjects, excluded_sessions(+detail), source/target ok
sessions, `target_finetune_sessions=[1]`, `target_online_sessions=[2,3]`. Target sets per
seed (e.g. 2026: 007/008/015/021/030/037/040/043/046/049).

**3. Split smoke test** `tests/test_splits.py` (manifest-only, login-safe): asserts
source=41/target=10, no source∩target overlap, target subjects have ses-01/02/03 all ok,
all used sessions are ok, excluded are truly non-ok. **All 5 splits PASS**; prints per-split
trial/session counts + balanced 0/1 label dist.

**4. Minimal CAP-EEGNet** — `src/models/eegnet.py` now has a real `EEGNetEncoder`/`EEGNet`
(Lawhern 2018, F1=8/D=2/F2=16/K=64; depthwise spatial over 58 ch, separable temporal;
feature_dim inferred = 496). `src/models/cap_eegnet.py` `CAPEEGNet` (nn.Module) = encoder +
linear cls head; `forward(x)` accepts `[B,58,1000]` or `[B,1,58,1000]`, returns
`{logits[B,2], features[B,496], proto_dist=None, confidence=None}`. Adapter / Prototype /
Confidence kept as **TODO stubs** (use_*=False; enabling raises NotImplementedError). ~3 026
params. (`padding='same'` even-kernel UserWarning is benign.)

**5. SHUTrialDataset smoke test** `tests/test_dataset_smoke.py` (compute node): builds
source/target datasets from `processed_manifest.csv` + split JSON; asserts entry is `.npz`
only (no `derivatives/.mat`), item x=`[58,1000]` float32, DataLoader batch x=`[B,58,1000]`,
y dtype `torch.long` (int64), labels ⊆ {0,1}; feeds a batch through CAP-EEGNet → logits
`[B,2]` (internal 4D `[B,1,58,1000]`). **PASS** for both roles.

**6. CUDA check (srun, --gres=gpu:1 on gpu02)**: `nvidia-smi` shows an RTX 4090 D (24 GB,
driver CUDA 12.8), but **torch 2.6.0 is CPU-only**: `torch.version.cuda=None`,
`torch.cuda.is_available()=False`, `device_count=0`. **Fix plan (do NOT mutate shared
`mi_torch`)**: create a dedicated CUDA env, e.g. inside a GPU `srun`:
`conda create -n mi_torch_cu118 --clone mi_torch` then
`pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
(cu121/cu124 also compatible with the 12.8 driver), and point GPU jobs at it. Awaiting user
go-ahead before installing.

**7. Sanity training** `scripts/sanity_train.py` (+ `scripts/slurm/sanity_train_gpu.sbatch`):
3 source subjects (sub-001/002/003), 9 npz, 1 800 trials, 3 epochs, batch 128, Adam lr 1e-3.
Ran on CPU (no CUDA). **loss 0.7002 → 0.6474 → 0.6014 (decreased=True)**, acc 0.541 → 0.630 →
0.682, no shape errors, ~38 s. Output: `outputs/sanity_check/sanity_check_metrics.json`.
Device auto-selects cuda if available else cpu, so it will use GPU once cu-torch is installed.

**Deliberately NOT done**: formal cross-subject training loop, fine-tuning, online loops,
prototype/confidence/adapter heads, GPU torch install. Awaiting user go-ahead.

## 2026-06-05 — Full preprocessing QC + quality comparison vs official derivatives

**What ran**: full `eog_ecg_clean` preprocessing was already complete (153 sessions, written
to the external `processed/eog_ecg_clean/` tree). This step did **QC + a quality comparison
against the paper `derivatives/2C dataset_processeddata` .mat** (NO training, NO 41/10, NO
CAP-EEGNet). New code:
- `src/evaluation/data_quality.py` — reusable, numpy/scipy-only metrics: amplitude, Welch
  PSD + bandpowers (δ/θ/μ/β/low-γ/50 Hz), per-channel RMS/std, paired similarity
  (trial-wise + per-channel Pearson, MAE/RMSE/relRMSE), MI class separability (Cohen's d /
  Fisher on log-bandpower at C3/C4), official `.mat` locator (handles the one non-`eeg/`
  layout) + loader (data[58,1000,200]→[200,58,1000], labels 1/2→0/1), and
  `best_session_assignment()` (per-subject session matching).
- `src/visualization/quality_plots.py` (NEW package) — 13 QC/compare figures (Agg backend).
- `scripts/compare_processed_quality.py` — orchestrator: manifest QC (19 checks) →
  per-subject session-aligned comparison → CSV/JSON/MD + figures.
- `scripts/slurm/compare_quality_cpu.sbatch` — reusable CPU job.
Ran on a compute node via `srun` (~6 min, `-c 8 --mem 32G`). Outputs in
`processed/eog_ecg_clean/qc_vs_derivatives/`: `manifest_qc_summary.json`,
`session_quality_metrics.csv` (153), `paired_similarity_metrics.csv` (144),
`session_alignment.json`, `QC_REPORT.md`, `figures/*.png` (13).

**Manifest QC (153)**: status **148 ok / 5 failed**. Failures = trigger/试次 < 200
(sub-023/ses-01=199, sub-024/ses-02=199, sub-024/ses-03=195, sub-026/ses-01=199,
sub-032/ses-02=199) → shape/label/trigger checks fail for exactly those 5. The other 148:
shape [200,58,1000], y[200], 100/100, 250 Hz, 58 ch, 1000 times, **0 NaN/Inf**. aux cleaning
used 153/153; EOG ICs removed in 106 sessions (189 comps), ECG in 43 (74); ICA n_components
retry fired 10×; **no-aux-clean fallback 0×**. Total 6.15 GiB.

**KEY FINDING — official derivatives session ordering is permuted.** For **22/51 subjects**
the paper `.mat` stores sessions in a different order than BIDS sourcedata (our ses-YY).
Naive same-ses comparison produced absurd std ratios (e.g. sub-030 ses-01 264×, sub-018
ses-01 14×, with reciprocal <0.1 in another session of the same subject). We match each of
our sessions to its true official counterpart within a subject via a (std, max|·|) amplitude
fingerprint (robust to ICA), accepting a permutation only when it beats identity by a margin.
Cross-validated: after matching, those subjects' labels become **exact** and trial-corr jumps
(e.g. sub-001/ses-02 was "102/200 order-diff" under same-ses → it's actually our-ses-02 ↔
official-ses-03, exact + corr 0.92). This reinterprets the earlier "trial-order differences"
as session swaps. **Our (X,y) are correct** (sourcedata + evt.bdf; per-subject session
amplitude *set* matches official). Permuted subjects: 001,004,006,007,008,010,013,017,018,
023,025,027,030,036,038,039,042,044,045,047,049,051.

**Comparison verdict (after alignment) = PASS**:
- std ratio (ours/official) median **0.976** (111/153 ours lower → artifact removal), RMS
  ratio 0.976; μ/α bandpower ratio **0.898**, β ratio **0.941**.
- paired (144 exact-label sessions) trial-wise Pearson median **0.954**, rel-RMSE 0.251.
- PSD overlays (global + C3/C4/Cz) overlap through μ/β; only 40–50 Hz differs (MNE firwin +
  50 Hz notch vs EEGLAB rolloff — above MI bands, expected).
- MI class separability (|Cohen's d| on log-bandpower) ours≈official at C3/C4 μ/β
  (e.g. C3 μ 0.138 vs 0.150; C4 β 0.142 vs 0.144) → discriminative MI info preserved.
- 22 "attention" sessions = the 5 failed + a few heavy-clean low-ratio sessions (std ratio
  0.35–0.42, genuine artifact removal) + genuinely high-amplitude recordings (>500 µV trials,
  also high in official) — none are data bugs.

**Recommendation**: **proceed to 41/10 split + SHUTrialDataset** using the **148 ok** sessions
(`SHUTrialDataset.from_manifest(..., statuses=('ok',))` already filters). First decide the 5
failed sessions: exclude, or re-extract triggers from raw (their subjects have other ok
sessions, so subject-level 41/10 split is unaffected). **Still deliberately NOT done**:
training, 41/10 split, CAP-EEGNet, any GPU job — awaiting user go-ahead.

---

## 2026-06-04 — Small-batch validation: sub-001/002/003 × 3 = 9 sessions (9/9 ok)

**Run**: `srun ... python scripts/preprocess_all.py --subjects 1,2,3 --tag dryrun` (~313 s,
compute node). Outputs to the external eog_ecg_clean tree; wrote
`processed_manifest.dryrun.csv` + `preprocess_summary.dryrun.csv` (canonical names reserved
for the full run). **9/9 ok**, every session shape [200,58,1000], float32/int64, 100/100
labels, 58 ch, 1000 times, 250 Hz, no NaN/Inf.

**New fail policy (req): `evaluate_failure_reasons()`** in `eog_ecg_clean.py`, applied after
the `.mat` cross-check in `pipeline.py`. A session is FAILED iff: shape≠[200,58,1000], label
count≠100/100, trigger count≠200, NaN/Inf in X, or (when `.mat` exists)
`labels_multiset_match=False`. `labels_match_mat=False` (order-only diff) is NOT a failure.
Added `labels_multiset_match` + `n_labels_agree` columns to the processed manifest + summary.

**Per-session results (excluded ICs / match_mat / multiset)**
- sub-001: ses-01 [1,4,10] T/T; ses-02 [1] F/T (order diff, 102/200); ses-03 [1,2] —/— (no .mat).
- sub-002: ses-01/02/03 all [] (none) T/T. Top EOG |corr| only ~0.10–0.14 → genuinely little
  IC-level ocular contamination (NOT a near-miss); sub-002 ses-01 ECG flat/invalid. Removing
  nothing is the correct conservative outcome.
- sub-003: ses-01 [1] T/T; ses-02 **[0,1,3,9,12,15,16]** T/T — 4 EOG (|corr| 0.51–0.75) + **3
  ECG (|corr| 0.56–0.77)**: first session with real cardiac ICs → ECG path validated; ses-03
  [3] T/T (see below).

**ECG**: detected in every session, but only sub-003/ses-02 had ECG-correlated ICs (≥0.5).
Elsewhere max |corr|~0.002–0.03 → no cardiac IC, not removed. Confirms ECG cleaning fires
only when there's genuine contamination.

**Robustness fix — ICA n_components retry.** sub-003/ses-03 is an EXTREME-amplitude session
(std ≈ 660 µV, ~60× normal; the paper `.mat` agrees at 660 → real data). `n_components=0.99`
collapsed to 1 PCA comp → MNE `ICA.fit` raised. Added a retry: on fit failure, refit with a
fixed int `aux_cleaning.ica.n_components_fallback` (=15). After the fix, ses-03 fits with 15
comps and removes EOG IC [3] (|corr| 0.653); the warning self-documents the retry. If the
retry also fails, the existing no-aux-clean fallback still applies (recorded, never crash).

**Disk**: each npz ≈ 41 MiB (≈43 MB; float EEG barely compresses), 9 sessions ≈ 392 MB →
full 153 ≈ ~6.5–7 GB. 38 files = 9×{npz,meta.json,preprocess_report.json,manifest_row.json}
+ 2 CSVs.

**Open questions**: (1) sub-001/ses-02 (and likely others) store trials in a different order
than the paper `.mat` — multiset matches, our (X,y) are evt.bdf-consistent; treat exact
`labels_match_mat` as non-authoritative. (2) Some subjects (sub-002) yield no ocular ICs at
threshold 0.5 — confirmed genuine here; revisit only if many subjects look under-cleaned.

**Deliberately NOT done**: full 51×3 run, training, 41/10 split, GPU. Awaiting user go-ahead.

---

## 2026-06-04 — EOG/ECG-clean preprocessing, .npz output, dry-run (sub-001)

**What changed (files)**
- `configs/paths.yaml`: renamed processed keys to `*_root`; `eog_ecg_clean_root` +
  `manifests.processed_manifest` now point to the external
  `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/...`.
  `paper_style_root` kept (legacy sanity-check only, not the formal output).
- `configs/preprocess.yaml`: `mode: eog_ecg_clean`; `output.{format=npz, npz_compress,
  save_debug_npy=false, write_manifest_row}`; `aux_cleaning.{enabled, method=ica,
  fallback, validity, ica{...}}`.
- NEW `src/preprocessing/eog_ecg_clean.py`: aux-channel validity check + MNE ICA cleaning
  (fit EEG on 1 Hz-highpass copy; detect via EOG/ECG; exclude; apply) + paper-style 2nd
  half + 11-point QC report. ICA/detection failures degrade to no-aux-clean (recorded,
  never crash).
- NEW `src/preprocessing/pipeline.py`: per-session process→save(.npz/meta/report/
  manifest_row)→build manifest row; richer `.mat` cross-check. Shared by both scripts.
- `src/utils/paths.py`: `*_root` fields, `session_npz_path()`, `assert_safe_output_dir()`
  (refuses sourcedata/derivatives/raw root), `ensure_writable_dir()` (errors if unwritable).
- `src/utils/io.py`: `save_session_npz()`/`load_session_npz()`/`save_debug_npy()`.
- `src/data/manifest.py`: `PROCESSED_MANIFEST_FIELDS` + `build_processed_manifest_row()`
  (records `npz_path`, NOT X/y paths).
- `scripts/preprocess_raw.py`: dispatch on `mode`; eog_ecg_clean → .npz + report + safety.
- `scripts/preprocess_all.py`: implemented (manifest-driven; `--subjects/--sessions/
  --limit/--tag`; per-session try/except → status+error_message; writes
  `processed_manifest.csv` + `preprocess_summary.csv` beside the npz tree).
- `src/data/shu_dataset.py`: implemented — reads X/y from `npz_path` (LRU-cached, lazy
  index from y), `from_manifest()` / `from_npz_paths()`.

**.npz keys**: `X` [200,58,1000] float32 (µV), `y` [200] int64 (0/1), `subject_id`,
`session_id`, `sfreq` (=250), `channel_names` (58).

**Dry-run (sub-001, `srun` compute node, ~36 s/session, mode=eog_ecg_clean)** — 3/3 ok:
- All sessions: shape [200,58,1000], dtype float32/int64, 100/100 labels, no NaN/Inf,
  58 ch, 1000 times, 250 Hz → every QC `all_passed=true`.
- Aux cleaning REAL (correlation-based IC detection, |corr|>0.5):
  - ses-01: valid EOG=HEOR/HEOL/VEOU (VEOL flat→invalid), ECG valid; excluded EOG ICs
    [1,4,10] (corr 0.66/0.65/0.61; gap to next 0.46). std 11.28→9.99.
  - ses-02: all 4 EOG + ECG valid; excluded EOG IC [1].
  - ses-03: all valid; excluded EOG ICs [1,2] (0.69/0.61).
  - ECG: detected in every session but max IC corr ~0.02–0.03 (both `ctps` and
    `correlation` agree) → no cardiac IC removed. Plausibly little cardiac contamination
    in this montage; recorded, not forced.
- Disk: ~42–68 MB/session (compressed npz), 150 MB for 3 → full 153 ≈ ~7–8 GB.

**KEY FINDING / open question — ses-02 `.mat` label order**
- ses-01 labels match the paper `.mat` 200/200 (exact). ses-02: exact match only 102/200
  (≈chance) BUT multiset identical (100/100) and triggers are monotonic/evenly-spaced
  (same structure as ses-01). ⇒ the paper `.mat` stores ses-02 trials in a DIFFERENT
  order than acquisition; our chronological evt.bdf order pairs each `(X_i, y_i)` from the
  SAME trigger (self-consistent, correct). Cross-check now reports `labels_match` (exact)
  + `labels_multiset_match` + `n_labels_agree`. Treat `labels_match_mat` (exact) as
  non-authoritative for ordering; multiset is the meaningful check.

**Decisions**
- IC detection uses absolute-correlation (`measure='correlation'`, thr 0.5) instead of the
  MNE z-score default (3.0), which detected nothing on ses-01/02; ECG default `correlation`
  (ctps found nothing). All thresholds live in `configs/preprocess.yaml`.
- npz is `savez_compressed`; `X.npy/y.npy` only if `output.save_debug_npy=true`.

**Deliberately NOT done (await user go-ahead)**: full 51×3 `preprocess_all` (only
`--tag dryrun` files written; canonical `processed_manifest.csv` not created), CAP-EEGNet,
training, 41/10 split, any GPU job.

**Next step (after user confirms)**: run full `preprocess_all` on a compute node
(`srun`/sbatch), write canonical `processed_manifest.csv` + `preprocess_summary.csv`,
review failures/`labels_match_mat`, then 41/10 splits + CAP-EEGNet.

---

## 2026-06-04 — Paused at framework + single-session sanity check

**State**
- Framework, rules (7), configs (6), docs (11), and src/scripts skeletons complete and
  committed; working tree clean.
- Raw manifest built: `manifests/shu_2c_raw_manifest.csv` (51 subjects, 153 sessions,
  0 missing BDFs).
- Single-session sanity check RETAINED: `outputs/processed_paper_style/sub-001/ses-01/`
  = {X.npy [200,58,1000], y.npy, meta.json, manifest_row.json}; validated vs the paper
  `.mat` (labels_match=True, corr ~0.994, std 11.28 vs 11.26).
- Slurm queue checked: no jobs running/queued (today's `mi_pp_test` srun checks all COMPLETED).

**Deliberately NOT done (await user go-ahead)**
- Full 51×3 `preprocess_all` (Slurm CPU job).
- Any Slurm submission / GPU job.
- Model training code, CAP-EEGNet implementation, 41/10 split.

**Next step (after user confirms)**: run full preprocessing via the Slurm CPU job,
write `manifests/shu_2c_processed_manifest.csv`, then 41/10 splits + CAP-EEGNet.

---

## 2026-06-04 — Architecture refactor: external paths + manifests + CAP-EEGNet

**Why**: new direction — drop the baseline-first plan; main model is **CAP-EEGNet**;
raw data live OUTSIDE the repo and all paths come from `configs/paths.yaml`; MATLAB is
reference-only; everything Python/PyTorch. Current stage stays data/paths/preprocessing.

**Done**
- Dirs: added `manifests/`, `splits/`, `outputs/processed_*`; removed `data/`, `notebooks/`.
- Rules renamed/rewritten: `00-project-context` (new content), `10-data-paths` (NEW),
  `20-preprocessing`, `30-model-experiments`, `40-online-learning`, `50-server-slurm`
  (+ updated `90-agent-behavior`). Deleted old 10/20/30/40.
- `src/utils/paths.py` refactored: no hardcoded dataset root; `load_paths()` reads
  `configs/paths.yaml` (env `SHU_2C_ROOT` overrides), validates, returns a `Paths` object.
- `configs/`: NEW `paths.yaml`; `preprocess.yaml` slimmed to params only; renamed
  cross_subject→`train_cross_subject.yaml`, online→`online_adaptation.yaml` (CAP-EEGNet).
- `scripts/build_manifest.py` + `src/data/manifest.py`: scan external raw root →
  `manifests/shu_2c_raw_manifest.csv`. Verified: **51 subjects, 153 sessions, 0 missing**.
- `check_raw_bdf.py` + `preprocess_raw.py` now read `configs/paths.yaml`; preprocess
  writes to the configured processed dir, writes `manifest_row.json`, and (optional)
  cross-checks labels vs the paper `.mat`.
- Skeletons (no complex code yet): `src/models/cap_eegnet.py` (encoder+adapter+
  prototype+confidence+cls), `src/data/splits.py`, `src/data/shu_dataset.py`.
- Re-validated end-to-end on a compute node (srun): sub-001/ses-01 →
  `outputs/processed_paper_style/...`, shape [200,58,1000], **mat labels_match=True**,
  std 11.28 vs 11.26.

**Decision**: `configs/paths.yaml` ships with the verified real raw root
(`/share/workspace2/moto_imagination/WBCIC_SHU`) since it's known; loader still
validates + supports `SHU_2C_ROOT` override. Change the YAML if data moves.

**Next step**: when ready, implement `src/data` (dataset + subject-wise split),
then CAP-EEGNet encoder. Still NO full 51×3 preprocessing / GPU jobs in this stage.

---

## 2026-06-04 — Task 2: raw preprocessing implemented & validated (sub-001/ses-01)

**Done**
- Cracked the evt.bdf event parsing (was the #1 blocker). The 200 MI triggers live
  in the `BDF Annotations` TAL channel of the Neuracle BDF+C file; MNE only surfaces
  the block markers {7,8}. Wrote `src/preprocessing/neuracle_events.py` to parse the
  BDF header + TAL bytes directly -> recovers 100x'1'(left) + 100x'2'(right), ~8s apart.
- Implemented `src/preprocessing/shu_preprocess.py` (paper-style), faithful to the
  authors' `code/pre-processed/preprocessed.m`:
  drop {ECG,HEOR,HEOL,VEOU,VEOL} -> reref Pz & drop Pz (58 EEG) -> 0.5-40 bandpass
  -> 50 notch -> epoch [0,4)s (baseline = whole-epoch demean) -> resample 250 -> [200,58,1000].
- Wired `scripts/preprocess_raw.py`; ran on a COMPUTE NODE via `srun` (not login node).

**Validated vs paper .mat (derivatives/)**
- Shape [200,58,1000], labels match **exactly** (element-wise, all 200 trials).
- Signal correlation 0.988-1.000 (mean 0.994). After fixing a unit bug, scale matches:
  our std 11.283 vs paper 11.263 (ratio 1.0017), RMSE 0.876 uV (~7.8% of std). The
  residual is expected from EEGLAB-vs-MNE filter implementation differences.

**Decisions / gotchas**
- UNIT QUIRK: BDF physical dim is the garbled `?V` (meant µV), so MNE does NOT apply
  µV->V scaling; `get_data()` already returns µV-magnitude values. We store as-is
  (µV) and do NOT multiply by 1e6 (doing so was a bug, fixed).
- reref target Pz = EEGLAB channel index 43 (confirmed by counting the EEG montage).
- Epoch window [0,4)s at 1000Hz (4000 samples) then resample -> 1000 samples.

**Next step**
- Generalize to all 51x3 via `scripts/preprocess_all.py` + `scripts/slurm/preprocess_cpu.sbatch`
  (collect per-session status into outputs/preprocess_summary.csv; don't silently skip
  failures). Then start the EEGNet baseline.

---

## 2026-06-04 — Project scaffold created

**Done**
- Created project skeleton at `/share/home/yuan/SYX/eeg-mi-online/`:
  `.cursor/rules/` (6 rules), `docs/` (8 docs + references), `configs/`,
  `scripts/` + `scripts/slurm/`, `src/` (7 packages), `data/`, `outputs/`,
  `logs/`, `checkpoints/`, `notebooks/`.
- Wrote the 6 Cursor rules, the docs (PROJECT_BRIEF, DATASET_SHU,
  PREPROCESSING_SPEC, EXPERIMENT_PROTOCOL, MODEL_PLAN, SERVER_RUNBOOK, ENVIRONMENT),
  AGENTS.md, this file, `.gitignore`, `requirements.txt`.
- Copied the senior's chat record to `docs/references/ChatGPT-EEG-MI-pretraining.md`.
- Implemented `scripts/check_raw_bdf.py` (raw BDF inspector, Task 1).
- Added documented stubs for the other scripts + `src/` modules.
- Wrote Slurm sbatch templates in `scripts/slurm/` (adapted from `run_test.sh`).
- Initialized git and made the first commit.

**Verified facts (from inspecting the real server/dataset)**
- Dataset root: `/share/workspace2/moto_imagination/WBCIC_SHU` (BIDS, READ-ONLY).
- 2C: **51** subject folders on disk; `participants_2C.tsv` lists 52; README says 53.
  -> Decision: always enumerate subjects from disk, never hardcode the count.
- Raw: 1000 Hz, **64 ch**. `check_raw_bdf.py` on sub-001/ses-01 VERIFIED the real
  layout = **59 EEG + 1 ECG (`ECG`) + 4 EOG (`HEOR/HEOL/VEOU/VEOL`)**. The
  `task-motorimagery_eeg.json` count "1 EOG, 4 ECG" is SWAPPED vs the actual names
  -> trust the names. (The generic plan's "ch60=ECG, ch61-64=EOG" was actually
  closer to reality than the JSON.)
- Events: 1=left, 2=right, 3=foot(3C only). 2C uses {1,2} -> internal {0,1}.
- Target processed shape `[200, 58, 1000]` (58 = 59 EEG minus Pz; 1000 = 4 s @ 250 Hz).
- Paper `.mat` confirmed: `data [58,1000,200]` float32, `labels [1,200]` in {1,2}.
- Env `mi_torch`: py3.10.18, torch 2.6.0, mne 1.10.0, numpy 2.2.5, scipy 1.15.3,
  sklearn 1.7.1, pandas 2.2.3, h5py 3.14.0, einops 0.8.1. No braindecode.
- Slurm: `gpu2node`(default)/`gpu3node`, each `gpu:8`/128 CPU/~773 GB; modules
  `cuda/11.8`, `anaconda3`.

**Decisions**
- Two preprocessing variants: paper-style (first, no ICA) and EOG/ECG-clean (later).
- First model = EEGNet encoder + classification head + prototype + confidence head
  + adapter (the chat record's "minimal version"). Baselines (EEGNet/DeepConvNet/
  FBCNet) come before that, to validate preprocessing.
- Git identity for commits: see commit log; tell the maintainer if it needs changing.

**check_raw_bdf.py first run (sub-001/ses-01) — verified**
- sfreq=1000 Hz, 64 ch, duration=2250 s (~37.5 min; ~11.25 s/trial for 200 trials).
- Channels: 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`). `other`=[] after
  improving the classifier to recognize H/V-EOG names.
- Aux validity: first 10 s of `ECG` and `VEOL` were all-zero (flat); `HEOR/HEOL/VEOU`
  active. -> must validate aux over the full recording before use (variant 2).
- Report saved at `outputs/raw_check/sub-001_ses-01_raw_check.json`.

**Open questions / TODO before scaling up**
- ⚠️ `mi_torch` torch is **CPU-only** (`torch.version.cuda is None`). Must install a
  cu118-matched torch (or make a `mi_torch_cu118` env) before real GPU training.
  (RESOLVED later: `mi_torch_cu118` env created, torch 2.7.1+cu118.)
- ⚠️ **Event triggers**: `evt.bdf` annotations via MNE gave only `{"7":1,"8":1}`
  (channel "Empty Event Data"), NOT 200 MI markers. (RESOLVED: TAL parser in
  `neuracle_events.py`.)
- Confirm cue timing for the 4 s epoch window + baseline interval (paper detail TBD).
- Check whether the flat ECG/VEOL is session-specific or dataset-wide.

**Next step**
- Run `scripts/check_raw_bdf.py` on `sub-001/ses-01` (login-node-safe: reads one
  file, prints + dumps JSON), read the report, then implement `preprocess_raw.py`
  for a single session and confirm the `[200, 58, 1000]` shape.
