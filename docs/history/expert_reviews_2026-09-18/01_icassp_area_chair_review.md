---
title: "Expert review of PLAN v1 (2026-09-18 00:45, independent subagent)"
created: "2026-09-18"
updated: "2026-09-18"
status: "archived"
---

> Exported verbatim from the review subagent's final message. The brief it received follows; the review itself starts at the first `#` heading.
> How each point was adopted: `docs/PLAN_spiking_eegnex_icassp2027.md` §0 and `docs/DECISIONS.md` D-001–D-019.

<details><summary>Review brief (prompt given to the reviewer)</summary>

```text
You are a senior ICASSP area chair and frequent reviewer for the BISP (biomedical signal processing) and MLSP tracks, with deep knowledge of the EEG/BCI deep-learning literature and of neuromorphic/spiking papers submitted to signal-processing venues. You are pre-reviewing a research plan for an ICASSP 2027 submission (deadline in 6 days; 4 pages + 1 reference page) so the authors can fix positioning before running experiments.

Read the plan: /share/home/yuan/SYX/eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md
Read the completed benchmark the plan builds on: /share/home/yuan/SYX/MEETING_BRIEF_2026-09-18.md (sections 1-3: 3 cohorts, 87 subjects, 18 ANN models, paired stats; EEGNeX is the strongest/most efficient ANN).

Use web search to check the 2024-2026 literature on spiking neural networks for motor-imagery / EEG decoding (e.g., spiking EEGNet variants, "A lightweight spiking neural network for EEG-based motor imagery classification" Neural Networks 2025, SpiTranNet-LIF 2026, SRIF neuron 2026, ISAM-MTL 2025, spiking transformers for EEG, temporal/streaming SNNs for biosignals, energy-accounting conventions) and on cross-subject MI SOTA with alignment (Euclidean Alignment, EEG-Deformer, ATCNet, MIRepNet, EEG foundation models). Identify the closest prior work to each of the plan's claimed contributions: (1) temporal-native spiking of a compact MI CNN with the EEG time axis as the spiking time axis, (2) heterogeneous learnable time constants for EEG, (3) the interaction "unsupervised alignment helps SNNs more than ANNs", (4) large-scale (87 subjects, 3 cohorts) paired evaluation of an SNN against 18 ANN baselines, (5) energy accounting from measured firing rates.

Then answer, as the meta-reviewer would:
1. Novelty verdict per contribution: new / incremental / already done (cite the closest work with venue and year).
2. What is the single strongest framing for this paper at ICASSP (title + 3 bullet contributions) that the planned evidence can actually support? Give 2 alternative framings ranked.
3. The most likely reasons for rejection at ICASSP and how to pre-empt each within 6 days (e.g., missing BCI IV 2a, transductive alignment, single seed, no hardware measurement, unclear novelty vs spiking-EEGNet papers).
4. Which experiments in section 3/4 of the plan are essential for acceptance vs nice-to-have; what minimal table/figure set fits 4 pages.
5. Reference list: the 12-15 must-cite papers (with venue/year) for this exact story.
6. A 1-10 predicted acceptance probability score for (a) the plan as written, (b) the plan with your recommended framing, with one-sentence justifications.

Be specific and critical; the authors need to know what will get attacked. Do not modify any files.
```

</details>

# Meta-review: "Spiking EEGNeX for cross-subject MI" (ICASSP 2027 plan v1)

**Bottom line first.** The plan is well-engineered and its evaluation infrastructure (87 subjects, paired per-subject statistics against 18 ANN entries) is genuinely better than anything in the SNN-for-MI literature. But three of the five claimed contributions are already published, one of them by Dongrui Wu's group — the inventors of Euclidean Alignment and a very likely reviewer pool. If the paper is written as "we propose temporal-native spiking + heterogeneous τ + energy accounting", it will be scored as an incremental spiking-EEGNet variant on unfamiliar datasets and rejected. If it is written as a rigorous non-inferiority + alignment-interaction study, it has a real chance.

I inspected the released code of the closest prior work ([Zhr1110/SnnForMI](https://github.com/Zhr1110/SnnForMI), Neural Networks 2025): `--T` defaults to `250*4 = 1000`, the model is `1×1 spatial conv → depthwise temporal conv → BN → permute(time→step) → PLIF → mean over time → FC`, and `main.py` has `--loo` (cross-subject) and `--EA` flags. That is "EEG sample axis = spike axis, learnable τ, rate readout, EA for cross-subject", published with code. Your Section 2.1 right-hand column is a deeper instance of it, not a new idea.

---

## 1. Novelty verdict per contribution

| # | Claimed contribution | Verdict | Closest prior work (venue, year) | What is actually still new |
|---|---|---|---|---|
| 1 | Temporal-native spiking of a compact MI CNN, EEG time axis = spiking axis | **Already done (in principle); incremental as an instance** | Zhang, Wang, An, Zheng, **Wu**, "A lightweight SNN for EEG-based MI classification", *Neural Networks* 191:107741, 2025 (PLIF over T=1000 raw samples @250 Hz, cross-subject on 3 datasets). Also C. Zhang, Pan, Della Santina, **NiSNN-A**, *IEEE TNNLS* 2025 (raw real-valued EEG, 400 steps/trial into LIF, LOSO on OpenBMI 54 subj + 2a). Biosignal precedent: Yin, Corradi, Bohté, *Nat. Mach. Intell.* 2021 (streaming ECG at sample rate with adaptive LIF). Hardware: Kumar et al., *TMLR* 2022 (EEG SNN on Loihi); Sharifshazileh et al., *Nat. Commun.* 2021 (ADM level-crossing EEG front-end). | Spiking a *multi-scale dilated* backbone with LIF at three temporal resolutions (1000/250/63 steps) after linear pooling; a causal/streaming variant; the explicit S-vs-D (native vs. T-repeat) comparison at identical weights. Do **not** write "first". |
| 2 | Heterogeneous learnable τ for EEG | **Incremental** | Perez-Nieves et al., *Nat. Commun.* 2021 (per-neuron heterogeneous learnable τ); Fang et al., PLIF, *ICCV* 2021; Yin et al., *NMI* 2021 (per-neuron τ co-trained, ECG); Zheng et al., DH-SNN, *Nat. Commun.* 2024 (heterogeneous dendritic timescales, EEG/DEAP); Zhang et al. 2025 already use PLIF on MI. | Only the physiologically-scaled log-uniform init (24–260 ms) and the "continuous analogue of EEGNeX's dilations" reading. This is an ablation, not a headline. |
| 3 | Interaction: unsupervised alignment helps SNNs more than ANNs | **New as a quantified effect; ingredients published** | EA/AdaBN on ANNs: He & Wu, *IEEE TBME* 2020; Junqueira et al., *J. Neural Eng.* 2024; Wimpff, Döbler, Yang, *Int. Winter Conf. BCI* 2024 (AdaBN alone ≈ +7 pp cross-subject on 2a). EA on an SNN: Zhang et al. 2025 (`--EA` in code). SNN shift-sensitivity + firing-rate re-normalisation: Zhao et al., "Threshold Modulation for OTTA of SNNs", *IJCNN* 2025 (shows firing-rate distribution shift under domain shift, on CV data). | Nobody has reported the *differential* gain (SNN vs. ANN, paired, with CI) on EEG, nor the inter-subject firing-rate-variance mechanism. This is the paper's only scientifically new claim — and it is contingent on H1 holding. |
| 4 | 87 subjects / 3 cohorts / 18 ANN baselines, paired | **Incremental in kind, strong in rigor** | NiSNN-A (OpenBMI, 54 subj, LOSO, 7 CNN + 10 SNN baselines); HR-SNN, *IEEE TCDS* 2024 (PhysioNet, 109 subjects, global training + 2a); MOABB, Chevallier et al., arXiv 2404.15319, 2024 (stat-meta-analysis framework, no SNN). | First SNN-vs-ANN comparison in MI-BCI with paired Wilcoxon–Holm / Friedman–Nemenyi and *strong* ANN baselines (ATCNet, EEG-Deformer, EEGNeX; SNN papers compare only to EEGNet/Deep/ShallowConvNet). Do not claim "largest" (109 > 87). Weakness: all three cohorts are from one lab (Shanghai University). |
| 5 | Energy from measured firing rates | **Already done (standard practice)** | NiSNN-A, HR-SNN, Zhang 2025, Yin 2021 all do this with Horowitz *ISSCC* 2014 constants (4.6/0.9 pJ); Lemaire et al., *ICONIP* 2022 formalise it; Dampfhoffer et al., *IEEE TETCI* 2023 show it ignores memory traffic and is only valid at very high sparsity. | The S-vs-D "T-multiplier" point (native time removes the ×T on SynOps) is a modest, real observation. Keep energy as a reported metric with caveats, never in the title. |

---

## 2. Strongest framing the planned evidence can support

**Recommended (rank 1): non-inferiority + alignment interaction.**

Title: *"Aligned Spiking EEGNeX: Subject Alignment Closes the Spiking Gap in Cross-Subject Motor-Imagery Decoding"* (or, if H1 is weak: *"Spiking EEGNeX Is Non-Inferior to 18 ANNs Across 87 Subjects When Subject-Aligned"*)

- **S-EEGNeX**: a temporal-native, causal spiking mapping of the strongest compact MI CNN in a 3-cohort benchmark; one weight set, three temporal resolutions, no T-repeat; the LIF-as-envelope-detector reading is *validated*, not asserted (one panel: LIF₁ spike count vs. per-trial log band-power, r reported).
- **Evidence**: pre-registered non-inferiority (margin 1 pp) against EEGNeX and paired comparison against 17 further ANNs on 87 subjects / 3 cohorts, plus a BCI IV 2a anchor; the first SNN–ANN comparison in MI-BCI with per-subject paired statistics and a CD diagram.
- **Finding**: label-free subject alignment (EA + AdaBN) reduces inter-subject firing-rate variance and yields a larger gain for the SNN than for its ANN twin (interaction Δ with 95 % CI); SynOps/estimated energy reported with the D-version (T=4) as the literature-convention baseline.

**Alternative A (rank 2): mechanistic paper.** *"Firing-Rate Homeostasis via Test-Time Alignment Makes Spiking MI Decoders Subject-Robust."* Contributions: threshold scale-sensitivity → per-subject firing-rate drift (measured); EA+AdaBN as zero-label homeostasis; interaction effect. Higher ceiling, but dies if the interaction is < ~1.5 pp or not significant. Choose this only after seeing the SHU pilot.

**Alternative B (rank 3): benchmark paper.** *"Spiking vs. Artificial Networks for Cross-Subject Motor Imagery: A Paired 87-Subject Benchmark."* ICASSP BISP does accept benchmark papers (ICASSP 2026 had "Temporal context and architecture: a benchmark for naturalistic EEG decoding"), but novelty scores will be low. Fallback if S-EEGNeX loses > 2 pp everywhere.

---

## 3. Most likely rejection reasons and 6-day pre-emption

1. **"Spiking EEGNet already exists" (Zhang 2025, NiSNN-A, HR-SNN, LENet, SRIF, SpiTranNet).** Pre-empt with a 5-row related-work table: dataset size / #ANN baselines / native-time? / alignment? / paired stats? — and word the contributions exactly as in §2. Cite all of them; Wu's group will recognise omission instantly.
2. **No standard dataset.** WBCIC-SHU and SHU 2022 are public and in MOABB (`Yang2025`), but reviewers cannot anchor numbers. Add **BCI IV 2a LOSO** (2-class LH/RH as in NiSNN-A ≈ 66 %, and 4-class): 9 folds × {eegnex, S-EEGNeX-S, S-EEGNet} × {z-score, EA} ≈ 54 tiny cells, ~5 GPU-h; loader from the EEG-FM-Bench pkl is half a day. Also state plainly that all three cohorts come from one site.
3. **Transductive alignment.** Standard in BCI, but only if: (a) applied identically to every ANN entry (fairness), (b) explicitly label-free, (c) you show a first-k-trial calibration curve (k = 8/16/32, evaluation-only) or online running EA (Wimpff 2024). Cite Wu's "Revisiting Euclidean Alignment" (arXiv 2502.09203, 2025) for correct usage.
4. **Single training seed.** SNN surrogate-gradient training is noisier than ANN training. Run 3 seeds for the two headline entries on SHU (~10 GPU-h) and report seed-SD vs. subject-SD; argue the paired 87-subject design controls the dominant variance. Put it in limitations.
5. **Energy is estimated, not measured.** Nobody will demand hardware at ICASSP, but they will attack "15–20×": conv1 (~30 M MACs) stays analog and dominates SNN energy; membrane updates and memory traffic are ignored. Report SynOps and per-layer firing rates as primary, energy as "estimated compute energy, 45 nm, excl. memory", cite Dampfhoffer 2023, and never put a multiplier in title/abstract.
6. **Training the S-version over 1000 iterative steps.** NiSNN-A reports iterative LIF failing at long T on EEG (that is their motivation for NiLIF). Zhang 2025 trained one PLIF layer over 1000 steps fine, so it is feasible, but your three stacked LIF layers with BPTT are a real risk. Pilot this first; if it collapses, the D-version becomes the main model and the story changes — decide by 9/19 evening.
7. **"Differences are within noise, so what?"** If S ≈ EEGNeX within 1 pp, that is a positive non-inferiority result only if the margin is pre-registered and a CI is shown. Say so explicitly.
8. **"Streaming" without evidence.** Either add an anytime accuracy-vs-time curve (evaluation-only, cheap) or delete the word.
9. **SHU 2022 floor effect and 3C n = 11.** Demote both to secondary; do not build claims on them.
10. **Reproducibility.** Save checkpoints and per-trial predictions this time (needed for AdaBN and for paired stats); promise code release.

---

## 4. Essential vs. nice-to-have; minimal table/figure set

| Priority | Experiment | Why | Cost (4×4090D) |
|---|---|---|---|
| **Essential** | `eegnex` vs `seegnex_stream@hplif` × {z-score, EA} × AdaBN {off, on}, 2C + SHU, splits 41/42 | H0 + H1: the whole paper | 2C ≈ 25, SHU ≈ 10 GPU-h |
| **Essential** | `seegnet@plif` × {z-score, EA} | the literature's SNN baseline | ≈ 12 GPU-h |
| **Essential** | `seegnex@hplif` (D, T=4) × {z-score, EA} | literature convention + T-multiplier (H2/H5) | ≈ 15 GPU-h |
| **Essential** | BCI IV 2a LOSO anchor (3 entries × 2 alignments) | reason 2 above | ≈ 5 GPU-h + loader |
| **Essential** | SynOps / firing-rate logging, LIF₁-count vs log-var correlation, first-k calibration curve | free at evaluation time | CPU |
| Nice-to-have | HetPLIF vs PLIF vs LIF (H3) | ablation; run on SHU only | ≈ 8 GPU-h |
| Nice-to-have | 3 training seeds for headline pair on SHU | reason 4 | ≈ 10 GPU-h |
| Nice-to-have | 3C cohort (LOSO 11) for 2 entries + `eegnex` re-run | third cohort headline | ≈ 20 GPU-h |
| Nice-to-have | anytime curve | supports "streaming" | CPU |
| Drop | `@same`, `spike_all`, T=8, soft reset, H4 as a claim | no reviewer will reward these in 4 pages | — |

Total essential ≈ 70 GPU-h (< 1 day on 4 cards); everything through the nice-to-haves ≈ 120 GPU-h — comfortably inside your 2.5-day window, with a full day of slack for the queue.

Four-page layout: **Fig. 1** EEGNeX → S-EEGNeX mapping (S vs D, LIF placement, 1000/250/63 steps; small). **Table 1** (the only table): rows EEGNeX, S-EEGNeX-S, S-EEGNeX-D, S-EEGNet, best-other-ANN (`atcnet@noattn`), EEGNet; columns 2C / SHU / 2a accuracy & macro-F1, each under {z-score, EA+AdaBN}; params; SynOps; est. energy; paired Δ vs EEGNeX with Holm-adjusted p. **Fig. 2** interaction: per-subject alignment gain for ANN vs SNN (paired dots + CI) beside per-layer inter-subject firing-rate variance before/after alignment. **Fig. 3** Friedman–Nemenyi CD diagram on 2C with all 18 ANN entries + SNN entries — this is how the "18 baselines" fits on one page. Neuron ablation and first-k curve go in two sentences of text.

---

## 5. Must-cite references (15)

1. Lawhern et al., EEGNet, *J. Neural Eng.* 2018.
2. Chen et al., EEGNeX, *Biomed. Signal Process. Control* 87:105475, 2024.
3. Altaheri et al., ATCNet, *IEEE Trans. Ind. Inform.* 19(2), 2023; Ding et al., EEG-Deformer, *IEEE JBHI* 2024/25 (one entry each or grouped).
4. He & Wu, Euclidean Alignment, *IEEE TBME* 67(2), 2020.
5. Junqueira, Aristimunha, Chevallier, de Camargo, systematic evaluation of EA with DL, *J. Neural Eng.* 21(3):036038, 2024.
6. Wimpff, Döbler, Yang, calibration-free OTTA (EA + AdaBN + entropy), *12th Int. Winter Conf. on BCI*, 2024; plus Li et al., AdaBN, *Pattern Recognition* 2018.
7. H. Zhang, H. Wang, J. An, S. Zheng, D. Wu, lightweight SNN for MI, *Neural Networks* 191:107741, 2025 — **closest prior; position against it explicitly**.
8. C. Zhang, W. Pan, C. Della Santina, NiSNN-A, *IEEE TNNLS* 2025 (doi 10.1109/TNNLS.2025.3538335).
9. HR-SNN, end-to-end SNN for 4-class MI, *IEEE Trans. Cogn. Dev. Syst.* 16(6), 2024.
10. Liao et al., LENet ANN→SNN for MI, *Biomed. Signal Process. Control* 100:107000, 2025; SRIF neuron, *Brain Sciences* 16(6):592, 2026 (newest conversion-based SNN-MI).
11. Kumar, Tang, Yoo, Michmizos, decoding EEG with SNNs on neuromorphic hardware (Loihi), *TMLR* 2022.
12. Neftci, Mostafa, Zenke, surrogate gradient learning, *IEEE Signal Processing Magazine* 36(6), 2019 (the ICASSP-native SNN reference).
13. Fang et al., PLIF, *ICCV* 2021; Perez-Nieves et al., neural heterogeneity, *Nat. Commun.* 12:5791, 2021; Yin, Corradi, Bohté, adaptive SRNNs, *Nat. Mach. Intell.* 3:905, 2021.
14. Horowitz, *ISSCC* 2014 (4.6/0.9 pJ) together with Dampfhoffer et al., "Are SNNs really more energy-efficient than ANNs?", *IEEE TETCI* 7(3), 2023.
15. Datasets and statistics: Yang et al., WBCIC-SHU, *Sci. Data* 12:488, 2025; Ma et al., SHU, *Sci. Data* 2022; BCI Competition IV 2a (Tangermann et al., *Front. Neurosci.* 2012); Demšar, *JMLR* 2006 (Friedman/Nemenyi CD diagrams).

If space remains: Zhao et al., threshold modulation for SNN OTTA, *IJCNN* 2025 (mechanistic support for H1); Chevallier et al., MOABB benchmark, arXiv 2024; ISAM-MTL, arXiv 2501.18089, 2025 and SpiTranNet-LIF, *ICAART* 2026 (to show you know the 2025–26 SNN-MI field); one EEG foundation model (CBraMod, *ICLR* 2025, or MIRepNet, arXiv 2025) in one sentence of the intro.

---

## 6. Predicted acceptance probability

- **(a) Plan as written: 4/10.** Reviewers familiar with Zhang 2025 / NiSNN-A will read contributions 1, 2, 5 as re-implementations on datasets they cannot anchor, and the strongest asset (paired statistics vs. 18 ANNs) is buried behind an SNN-architecture narrative.
- **(b) With the recommended framing (non-inferiority + interaction, 2a anchor, honest energy, seeds on SHU): 6/10** — rising to ~6.5 if the interaction is significant with a ≥ 1.5 pp differential; the rigor is unmatched in this sub-literature, and the only remaining exposure is the "no method novelty" reviewer, whom the CD diagram and the interaction figure can outvote.

One organisational note: decide on 9/19 evening, from the SHU pilot, whether the S-version trains stably over 1000 steps and whether H1 shows any signal; that single checkpoint determines whether you write framing 1, A, or B.
