---
title: "报告：稿件底稿按 ICASSP 2027 kit 对齐 + 三张队列表 + 四路严格审查结果（2026-09-18/19）"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
  - "#method/benchmark"
created: "2026-09-19"
updated: "2026-09-19 00:20"
status: "active"
---

# 稿件底稿（`old_reference.tex`）格式对齐与三张队列表：审查报告

## 0. 核心结论

1. `old_reference.tex`（团队 ICASSP 2026 论文）已按 2027 官方 kit 对齐**格式骨架**：编译通过（tectonic，Times 度量），三张队列表宽度
   198 / 224 / 198 pt < 244.7 pt 栏宽、无 overfull，字体全部 Type 1 内嵌，无页码，参考文献单独成页，脚注 9 pt。四路独立审查
   （格式 / 构建 / 数字 / 领域评审）的**可自动修正项已全部修完**并复核。
2. 三张表（2C / SHU / 3C）里已填的 **47 个数字全部由审查员用独立代码从 CSV 复算一致**（SD = 逐被试 ddof=1，三队列口径一致）。
3. **需要学长拍板的 5 个问题**（审查员意见，不是 Agent 的决定）见 §6：EEGNeX 行去留、"Ours 行按测试集最好的选"与预注册冲突、
   三张双行表占 ~0.8 页（有一键紧凑版）、3C 基线是否用 bench 配方重跑、EEGMoE 无代码无权重的替代。

## 1. 目标

用户 09-18 23:15 指令：以去年团队中的 ICASSP 论文为 LaTeX 底稿、按 2027 模板改格式；实验表改为每个数据集一张（3 张）；
模型 = 截图打勾的 EEGNet / FBCNet / ATCNet / EEG-Deformer + 另一窗口在跑的 MDGEEG / EEGMoE / EEGPT / LaBraM / TCFormer（占位）
+ 三个 SNN 候选中的一个；其他正文不动；不与另一窗口冲突。23:35 追加：按官方格式对照检查几遍，多专家严格审查。

## 2. 方法

- 表由新脚本 `scripts/build_dataset_tables.py` 从 CSV 生成并 `--splice` 进 `old_reference.tex` 的标记区（不手填）。
- 三个后台子任务：五个新基线的文献核实（含 web 检索）、2027 kit 格式清单、编译器；随后四个**零上下文**审查员并行：
  ① kit 格式合规审计、② LaTeX 构建与排版（实际编译 + 栅格化测量）、③ 数字与出处审计（独立 pandas 复算）、④ 资深 EEG-BCI 评审。
- 编译：`~/.local/bin/tectonic`（XeTeX），加 `[T1]{fontenc}` 得到 Times 度量；栅格化 `pymupdf`。集群无 pdfLaTeX，最终以 Overleaf 为准。

## 3. 协议（表的口径）

| 项 | 2C | SHU | 3C |
|:--|:--|:--|:--|
| 数据源 | bench `per_model.csv`（51 人 × 2 划分） | bench `per_model.csv`（25 人 × 2 划分） | `paper_baseline_3c_821_v1/tables/per_cell.csv`（LOSO 11 折，**论文配方 patience 100**） |
| 主指标列 | Acc ± SD | **BAcc ± SD**（D-015；与 Acc 最大差 0.03 pp，caption 自动写出） | Acc ± SD |
| 其余列 | F1、AUC | F1、AUC | F1、AUC（macro OvR） |
| SD | 逐被试（两划分先平均）ddof=1 | 同左 | 逐折 = 逐被试 ddof=1 |
| 未跑 | — | — | FBCNet 从未在 3C 跑 → `--`（"not run"），不是 TBD |

κ 在 2C/SHU 可加（`--metrics-2c acc+sd,macro_f1,auc,kappa`），3C 逐折表没有。所有表 caption 以「Unaligned block (per-trial
z-scoring only; no EA/AdaBN)」开头，防止与对齐块混比（AGENTS §4.4）。

## 4. 结果：审查发现与处置

### 4.1 已修（格式 / 构建）

| 发现（审查员） | 处置 |
|:--|:--|
| `refs.bib` 的 `note` 会被 IEEEbib.bst 打印：VERIFY、GitHub 链接全进了参考文献 [2][4]–[9]（①②，MAJOR） | 18 个 `note` → `annote`（bst 忽略）；文件头写明规则；MIRepNet 加 `howpublished = {arXiv:…}`（bst 也忽略 eprint/url/doi） |
| `\thanks` 脚注 8 pt，低于 kit 9 pt 下限（①） | 前言 `\renewcommand{\footnotesize}{\small}`；实测 8.97 pt |
| 表里出处行 `\footnotesize`（8 pt，沿用 2026 稿）（C） | 生成器改 `\small`（= 9 pt） |
| 三张表超栏宽 10 / 54 / 10 pt（编译实测） | 表头去 `(\%)`、Macro-F1 → F1、`\tabcolsep` 3 pt；实测 198 / 224 / 198 pt |
| `\enlargethispage` ×2 越出印刷区、`\vspace{-1pt}` ×3（C） | 全部删除 |
| `\vfill\pagebreak` 紧接 `\clearpage` 无意义（①） | 删 `\vfill\pagebreak`，留注释说明用法 |
| `\def\L` 覆盖 Ł、`\def\x` 无用（①） | 删 |
| 头注不准：第 5 页"只放参考文献"过严（CMS 允许资助致谢与伦理声明）、"Type 1 only"与 fonttype 42 矛盾、pdfLaTeX 非必需、漏"subset"（①） | 头注重写：长度 / 印刷区 / 字体 / 题目与 CMS 一致 / ≤ 5 index terms / 伦理声明 |
| 子节标题 Title Case（kit 要求 sentence case）（C） | 8 个 `\subsection` 改首字母大写 |
| 未用的 `multicol`、`float`；`hyperref` 应最后加载；`\frenchspacing` 非 kit（C） | 前言按 kit 顺序重排 |
| "Table 1, 2 and 3 presents"（②） | "Tables 1–3 present" |
| `\tbd` 稳健性（②） | `\DeclareRobustCommand`；注释给出最终构建时改成报错的定义 |

### 4.2 已修（生成器正确性，③②④）

| 发现 | 处置 |
|:--|:--|
| 3C 走 `--snn-3c` 时 `split_seeds="[0]"` 会被误标"先导 split 41"；`setdefault` 让 patience-100 基线压过配方一致的重跑 | 3C 不打先导标；3C 的 analyze 行**覆盖**论文配方行，剩余历史行标 ‡ 并在 caption 写明"not recipe-matched" |
| `--extra` 静默覆盖正统数值、把 2C 的 per_cell 误当 LOSO、把任意 bench CSV 硬标"splits 41+42" | 键冲突直接报错（`--allow-override` 才放行）；per_cell 先逐被试平均再算 SD（复算 = bench 82.53 ± 12.04）；非正统 CSV 不写死协议 |
| FBCNet-3C 从未跑却标 TBD | `pending` 行才 TBD，其余无数据行 `--`，caption 图例 "--: not run" |
| 3C caption "same preprocessing as 2C" 不成立（3C = 官方衍生数据，2C = 自研 ICA 管线） | 改为 "official pre-processed derivatives, same per-trial z-scoring" |
| 微调的基础模型不该被说成"同一配方" | `pretrained=True` → 名字上标 P + 图例 |
| splice 正则不锚定行首、重复标记静默 | `^` + `re.M`，标记必须各出现一次，否则报错 |
| 无覆盖率守卫 | 任一行 `n_subjects` ≠ 51/25/11 直接拒绝生成 |
| 加粗按原值比较（打印精度平局漏加粗） | 按打印字符串比较 |
| SHU 的 Acc 列冗余（≤ 0.03 pp） | 三队列统一 3 个指标列：主指标 ± SD、F1、AUC |
| 表下注释列了三个候选的**测试集数字**（与"不看测试集选主模型"冲突） | 注释只留候选名与预注册说明；数字在 `dataset_tables.md` / `dataset_tables_long.csv` |
| `--snn-shu` 默认指向先导 CSV（易误用） | 三个 `--snn-*` 默认均为空，必须显式给 |
| 新增开关 | `--include eegnex`、`--layout compact`（单张 `table*`）、`--final`（残留 `\tbd` 拒绝写入）、`--pct-digits` |

### 4.3 数字复算（③）

47 个已填格（2C 12、SHU 16、3C 9、候选 12）与独立复算全部一致，无舍入边界问题；加粗 = 列内已填最大值；
出处 CSV 每格有 `source` + `source_key`；生成片段与稿件标记区逐字节一致。

### 4.4 仍属旧正文、随重写消失（不处理）

标题作者行超宽 2.5 pt（旧作者块）、Intro 一行超宽 6.4 pt、24 个旧 `\cite` 未定义（`strings.bib` 不在仓库）、摘要 168 词 > 150、
正文 5 页（旧文长度）、`main.pdf` / `visual.pdf` 缺失（编译用占位图）。

## 5. 分析

- 数字层面无风险：全部可追溯、可复算、可一键重生成。
- 表的**科学表达**仍有评审员（④）不满意的地方，且大多不是格式问题而是框架决定：加粗 0.2–0.6 pp 的均值差在 12 pp 被试 SD 下没有统计意义
  （项目设计是逐被试配对，表里没有配对统计）；缺 EEGNeX 让"非劣性"故事无法从表上读出；三张双行表 ~0.8 页对 4 页预算偏贵。
- 生成器已为这些决定预留开关，学长拍板后一条命令切换。

## 6. 待学长拍板（按重要性）

1. **EEGNeX 行**：截图未勾、学长说"不比 eegnex"；评审 ④ 视其缺席为 BLOCKER（S-EEGNeX 的母网络、三队列第一、预注册 H0-NI 的对照）。
   → `--include eegnex` 即加回。
2. **"Ours" 行怎么选**：用户口径"最后哪个效果最好用哪个"；项目规则（AGENTS §3.3 / §4.3）= 预注册主模型 S-EEGNeX-S，D 版为书面兜底
   （2C 上 S 低于 EEGNeX > 1 pp 时），**不按测试集挑**。建议：按预注册；若要展示两版，表里放 S 与 D 两行（H2-NI 本来就要比）。
3. **版面**：三张双行表 ≈ 370 mm ≈ 0.8 页；`--layout compact` 给出单张 `table*`（三队列并排、单行、≈ 0.3 页，宽约 140 mm < 178 mm）。
4. **3C 基线配方**：现在是 patience 100 的历史 LOSO 数字；主实验只重跑 EEGNet / EEGNeX（patience 25）。要么接受 ‡ 标注，
   要么在同 run 目录补跑 ATCNet / EEG-Deformer / FBCNet（≈ 10–15 GPU-h，09-21 前无卡）。
5. **EEGMoE**：无代码无权重，无法作为该论文的方法运行；另一窗口建议 CBraMod 替代（脚本已支持：有结果自动出现）。MDGEEG 训练算法未公开，
   若只跑官方结构需改名 "MDGEEG (arch.)"。

## 7. 与前序的关系

- D-032 的 Table 1（`paper/main.tex`，6 ANN + 3 SNN 单表）仍在；本报告的三张表按学长"每数据集一张"与用户勾选的名单生成，是 D-033 的产物。
- 另一窗口的 `docs/PROTOCOL_external_baselines.md`（23:30，待确认）：新基线 cell 写进主实验同 run 目录 → `analyze_snn_round.py` 汇总 →
  本脚本经 `--snn-*` 自动纳入，键名 `tcformer / labram / eegpt / mdgeeg / cbramod` 已对齐。

## 8. 下一步

1. 学长回复 §6 的 1–5 → 用对应开关重生成（一条命令）。
2. 主实验出齐后：`--snn-2c/-shu/-3c` + `--ours <key>` 重生成；提交前 `--final`。
3. Overleaf 编译（上传 `old_reference.tex` + `paper/{spconf.sty,IEEEbib.bst,refs.bib}` + 旧稿的 `strings.bib` 与图），核对 pdfLaTeX 下无 overfull。
4. 正文重写时逐项清掉头注里"content-side"事项（摘要 ≤ 150 词、prior-work 段、ORCID、AI 披露句、伦理声明）。

## 9. 文件清单

- 稿件：`old_reference.tex`（根目录）；样式与文献：`paper/spconf.sty`、`paper/IEEEbib.bst`、`paper/refs.bib`（新增 5 条：`jiang2024labram`、
  `wang2024eegpt`、`altaheri2025tcformer`、`song2025mdgeeg`、`gao2026eegmoe`）
- 生成器：`scripts/build_dataset_tables.py`；产物：`outputs/analysis/dataset_tables_2026-09-18/{tab_2c,tab_shu,tab_3c,tab_compact_all}.tex`、
  `dataset_tables_long.csv`、`dataset_tables.md`
- 决定：`docs/DECISIONS.md` D-033；状态：`docs/STATUS.md`；流水：`progress.md`
