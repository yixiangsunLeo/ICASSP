---
title: "ICASSP 2027 投稿说明 — 截稿、格式、检查单、稿件骨架"
tags:
  - "#modality/eeg"
  - "#method/snn"
created: "2026-09-18"
updated: "2026-09-18"
status: "active"
---

# ICASSP 2027 投稿说明

## 1. 时间（核对于 2026-09-18）

| 来源 | 写的是 | 备注 |
|:---|:---|:---|
| 官网 Call for Papers（https://2027.ieeeicassp.org/call-for-papers/） | **Full Paper Submission Deadline: September 23, 2026** | 从最初的 09-16 延期一次；先前记录「官网明示不再延期」 |
| CMS Paper Kit（https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php） | Paper Submission Deadline 16 September 2026 | 09-18 仍未更新，**以官网为准，但投稿前登录 CMS 确认系统开放** |
| 换算 | 09-23 23:59 **AoE** = **北京时间 09-24（周四）20:00** | 不要拖到最后一小时；CMS 高峰会慢 |

会议：2027-05-16 ~ 21，Toronto。录用通知 2027-01-13；终稿 2027-01-27；作者注册截止 2027-02-10。

## 2. 格式硬要求（Paper Kit 原文要点）

- **4 页技术内容**（含图、可含参考文献）+ **可选第 5 页**，只能放参考文献、资助致谢、Compliance with Ethical Standards 声明。总长 ≤ 5 页。
- 模板：`paper/spconf.sty`（ICASSP 2027 官方，2026-09-18 从 CMS 下载）+ `paper/IEEEbib.bst`；`paper/official_kit/Template.tex` / `Template.pdf` / `format.pdf` / `Author-PDF-Guide-V32.pdf` 为官方样例与 PDF 规范。**不要用 ICASSP 2025 的模板**。
- 单栏摘要、双栏正文、US Letter；Type 1 字体（禁 Type 3 位图字体）；PDF ≤ **5 MB**；文件名 = **第一作者姓氏.pdf**（如 `sun.pdf`）。
- **单盲**：publish-ready，带作者署名与单位。LaTeX 模板默认支持 2 个作者块，多作者按 FAQ 3 处理（`\name{...}` 里用 `\sthanks` / 逗号分隔）。
- **新增 2027：每位作者需 ORCID iD**（提前让所有作者注册 https://orcid.org）。
- **AI 使用披露**：IEEE 要求在致谢中披露 AI 生成的内容（文字/图/代码）及所用系统与范围；纯编辑/语法润色可不披露但建议披露。本项目代码与文本大量由 Cursor Agent 辅助生成 → **致谢必写一句披露**。
- 预印本允许（arXiv / TechRxiv / 个人主页），不算 prior publication。
- 撤稿要全体作者同意并邮件 papers@2027.ieeeicassp.org。

## 3. 编译

集群登录节点**没有** TeX（`pdflatex` / `latexmk` 都不存在）。三种办法：

1. **Overleaf**（推荐，与学长协作）：上传 `paper/` 整个目录（`main.tex`, `refs.bib`, `spconf.sty`, `IEEEbib.bst`, `figures/`），编译器 pdfLaTeX。
2. 本机 TeX Live：`cd paper && latexmk -pdf main.tex`（或 `pdflatex main; bibtex main; pdflatex main; pdflatex main`）。
3. 集群上临时装：`conda create -n tex -c conda-forge texlive-core` 体量大，不建议在截稿周做。

投稿前用 Acrobat / `pdffonts main.pdf` 确认无 Type 3 字体；CMS 会做 IEEE PDF eXpress 风格的检查。

## 4. 稿件骨架 ↔ 方案 §6 的映射（`paper/main.tex` 已按此摆好）

| 节 | 内容 | 数据来源 |
|:---|:---|:---|
| Abstract | 87 人 3 队列、18 ANN 配对；非劣性（δ = 1pp）；对齐 × 网络类型交互；能耗（以 EEGNet 为参照，不写乘数） | `outputs/analysis/snn_round/wbci_shu/*.csv` |
| 1 Introduction | 跨被试 MI 的瓶颈是被试偏移（自证：前 8 名差 1pp、被试 SD 12pp、67 M 垫底）；SNN 的能耗/流式优势与阈值尺度脆弱性；三条贡献；与 Zhang 2025 / NiSNN-A 的关系 | `docs/PRIOR_RESULTS.md` |
| 2 Method | 2.1 EEGNeX → S-EEGNeX 层映射（图 1）；2.2 时间原生编码（LIF₁ 在 250 Hz 采样率上 = 1-bit 电平交叉编码，'same' 填充 look-ahead ≤ 128 ms）；2.3 HetPLIF；2.4 EA + AdaBN（transductive-unsupervised，写明）；2.5 能耗核算 | PLAN §2 |
| 3 Experiments | 3.1 数据与协议（3 队列、6:2:2 k-fold / LOSO、配方、配对设计、预注册家族）；3.2 表 1 未对齐主表（Acc / BA / F1 / 参数 / SynOps / µJ；CD 图）；3.3 表 2 对齐块（3 模型 × {无, EA, EA+AdaBN}；H1 取 {EEGNeX, S-EEGNeX} × {无, EA+AdaBN} 的 2×2 子集）+ 图 2 交互；3.4 图 3 发放率被试间分布（对齐前后）；3.5 消融小表（编码位置、读出、初始化、神经元 LIF/PLIF/HetPLIF、D vs S） | `analyze_snn_round.py` 输出 |
| 4 Conclusion | 一段；limitation：transductive、单 seed、配方依赖、SHU 地板、S 版慢 3× | — |
| Acknowledgments（第 5 页） | 资助；**AI 使用披露** | — |
| References（第 5 页） | `refs.bib`（必引清单见 §5） | — |

## 5. 必须引用并明确对比（refs.bib 已放占位条目，**投稿前逐条核对元数据**）

Zhang, Wang, An, Zheng, Wu, *Neural Networks* 191:107741 (2025)（PLIF 直接跑 1000 采样点 + EA）· NiSNN-A (TNNLS 2025) · HR-SNN (TCDS 2024) · LENet / SRIF（ANN→SNN 转换）· Yin, Corradi, Bohté (Nat. Mach. Intell. 2021) · Perez-Nieves et al. (Nat. Commun. 2021，异质时间常数) · Fang et al. PLIF (ICCV 2021) · He & Wu (TBME 2020，Euclidean Alignment) · Junqueira et al. 2024 · Wimpff et al. 2024 · Horowitz (ISSCC 2014) + Dampfhoffer et al. 2023（能耗口径）· Chen et al. EEGNeX (2024) · Lawhern et al. EEGNet (JNE 2018) · Altaheri et al. ATCNet (TII 2023) · Ding et al. EEG-Deformer (2024) · Yang et al. WBCIC-SHU (*Sci Data* 2025) · Ma et al. SHU 2022 · DSGNet (JBHI 2026，3C 对标) · Li et al. AdaBN (2016) · Zenke & Vogels 2021（活动正则）· Neftci et al. 2019（代理梯度）。

## 6. 提交检查单

- [ ] 所有数字能指回 `outputs/analysis/**/*.csv` 或 `results_snapshot_2026-09-13/**/tables/*.csv` 的一行
- [ ] 两块主表没有混比（对齐 SNN 不与未对齐 ANN 直接比）
- [ ] 预注册 5 项检验的 p 与 Holm 校正一致；非劣性给单侧 CI 下界；报 n_eff、胜/平/负
- [ ] 能耗表述以 EEGNet / S-EEGNet 为参照；乘数不在标题/摘要
- [ ] transductive 对齐、单 seed、配方依赖、SHU 地板效应写进 limitation
- [ ] 被删除的表述没有回流（posterior drift、anytime/streaming、LIF = 包络检测器、1-bit 保相位定理、20×）
- [ ] 必引清单全部出现且元数据核对
- [ ] 4 页正文 + 第 5 页只放参考文献/致谢/合规声明；≤ 5 MB；无 Type 3 字体；文件名 = 第一作者姓
- [ ] 每位作者 ORCID；作者顺序与单位学长确认
- [ ] 致谢里的 AI 使用披露
- [ ] 登录 CMS 确认投稿系统开放、选对 track（Regular）与 EDICS
- [ ] 学长拍板 7 项（`docs/reports/REPORT_for_senior_2026-09-18.md` §6）已有答复
