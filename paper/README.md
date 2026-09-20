# paper/ — ICASSP 2027 稿件（样式文件 + 早期骨架）

> **2026-09-18 23:40 起，稿件底稿是仓库根目录的 `old_reference.tex`**（团队去年中的 ICASSP 2026 论文，已按 2027 kit 对齐前言、放入三张队列表；D-033）。
> 它编译时需要本目录的 `spconf.sty`、`IEEEbib.bst`、`refs.bib`（表格引用的基线文献）以及 2026 论文自己的 `strings.bib` 与图。
> 三张表由 `scripts/build_dataset_tables.py --splice old_reference.tex` 生成，不手填。
> 下面的 `main.tex` 是更早的格式骨架（方案 v2 分节注释、Table 1 基线行、limitation 清单），供合并参考；题目 / 框架 / 正文仍由学长决定。

| 文件 | 说明 |
|:---|:---|
| `main.tex` | 格式骨架（官方模板 + 候选分节 + 空表格；`\todo{}` 为占位，投稿前全部清空） |
| `refs.bib` | 必引文献（`AGENTS.md` §7.3 清单）；标 `VERIFY` 的条目是凭记忆写的，**投稿前逐条核对** |
| `spconf.sty`, `IEEEbib.bst` | ICASSP 2027 官方样式（2026-09-18 自 https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php 下载） |
| `official_kit/` | 官方样例 `Template.tex/.pdf`、`format.pdf`、`Author-PDF-Guide-V32.pdf`（IEEE PDF 规范） |
| `figures/` | 图（建议 PDF 矢量；由 `outputs/analysis/**` 的 CSV 画出，脚本放 `scripts/`） |
| `tables/` | 由 CSV 生成的 LaTeX 表片段（`\input{tables/xxx.tex}`），不要手填数字 |

## 编译

登录节点现在有 pdfLaTeX（2026-09-19 装的 TinyTeX / TeX Live 2026：`~/.TinyTeX`，`~/.local/bin/{pdflatex,bibtex,latexmk}`；
备用 `~/.local/bin/tectonic` 是 XeTeX 引擎，需在文件前加 `\RequirePackage[OT1]{fontenc}` 才会用 Times）。在 `/tmp` 里拷一份编译，不要在仓库里留 `.aux/.log`。任选：

```bash
# 1) Overleaf：上传本目录（main.tex refs.bib spconf.sty IEEEbib.bst figures/ tables/），编译器 pdfLaTeX
# 2) 本机 TeX Live
cd paper && latexmk -pdf main.tex        # 或 pdflatex main && bibtex main && pdflatex main && pdflatex main
pdffonts main.pdf | grep -i type3        # 必须为空（IEEE 禁 Type 3 字体）
```

## 硬要求（详见 `docs/ICASSP2027_SUBMISSION.md`）

4 页技术内容 + 可选第 5 页只放参考文献 / 资助 / 合规声明；PDF ≤ 5 MB；文件名 = 第一作者姓氏.pdf；单盲带署名；每位作者 ORCID；
致谢中披露 AI 辅助（已在 `\thanks{}` 里放了一句）。**截稿 2026-09-23 23:59 AoE（北京 09-24 20:00）。**

## 数字来源

- 主表 / CD 图 / 对齐块（3 模型 × 3 条件；H1 取 2×2 子集）/ 能耗：`outputs/analysis/snn_round/{wbci_shu,shu,wbci_3c}/{per_model,paired_tests,interaction_H1,energy}.csv`
  —— 生成命令见 `README.md` §2（**必须带 `--aligned-arm eaz`**）
- 18 个 ANN 基线：`results_snapshot_2026-09-13/experiments/{wbci_shu/wbci_2c_cross_subject_bench_v1,shu/shu_cross_subject_bench_v1}/tables/`
- 消融（SHU 先导，**lr 1e-3**，配对对象为 lr 1e-3 的 S-EEGNeX-S）：`outputs/analysis/snn_round/shu_pilot_lr1e3/paired_tests.csv`；lr 1e-4 的 S/D vs EEGNeX：`shu_pilot_lr1e4/paired_tests.csv`
- 表格生成脚本：待写 `scripts/make_paper_tables.py`（主实验期间允许在 `scripts/` 根下新增，不在作业闭包内）
