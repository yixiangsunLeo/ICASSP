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
You are a rigorous methodologist and statistician specializing in machine-learning evaluation and biomedical signal classification (paired designs, multiple comparisons, cross-validation leakage, pre-registration, effect sizes; think Demšar / Benavoli / Varoquaux-level expertise). You are reviewing the experimental design of an ICASSP 2027 paper (deadline in 6 days) on spiking EEGNeX for cross-subject motor-imagery EEG decoding.

Read the plan: /share/home/yuan/SYX/eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md (especially sections 1.1, 2.3, 3, 5)
Read how the existing benchmark's statistics are computed: /share/home/yuan/SYX/eeg-mi-online/code/summaries/cross_subject_bench.py and the protocol: /share/home/yuan/SYX/eeg-mi-online/code/experiments/cross_subject_protocols.py (docstring + cell_signature + kfold logic).
Existing paired-test output for reference: /share/home/yuan/SYX/eeg-mi-online/results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1/tables/paired_tests.csv and per_subject.csv (same dir), and /share/home/yuan/SYX/MEETING_BRIEF_2026-09-18.md sections 1-3 for the completed benchmark numbers.

Design facts: 3 cohorts (51 / 25 / 11 subjects). Protocol: subject-level 5-fold (6:2:2 train:val:test by subjects) with two subject-partition seeds (41, 42), one training seed; each subject is a test subject exactly once per partition seed, so the unit of analysis is the subject (per-subject accuracy averaged over the 2 partitions). Model selection = early stopping on validation macro-F1 (validation subjects disjoint from test). 18 ANN baselines were already run on exactly these partitions; new spiking models will be run on the same partitions (deterministic folds from the seed) so all comparisons are paired at the subject level. The plan adds (a) Euclidean Alignment fitted per session on that session's own unlabeled trials (train, val and test subjects each aligned with their own data) and (b) AdaBN re-estimating BatchNorm statistics on each test subject's unlabeled trials at evaluation time. Learning rate will be piloted on one cohort/one partition seed and chosen by validation macro-F1.

Assess with full rigor:
1. Is the paired subject-level analysis valid and what is its power? With 51 subjects and a paired SD of ~4-6pp (observed), what detectable difference results? For 25 and 11 subjects? Is averaging a subject's two partition results before testing correct, or should partition be a factor?
2. The interaction hypothesis H1 (alignment helps SNN more than ANN): specify the exact statistical test (e.g., per-subject difference-of-differences with Wilcoxon or a mixed model with subject random effect), the pre-registered family for Holm correction, and how to report effect sizes / CIs. Warn about any garden-of-forking-paths risk.
3. Leakage audit: EA per session using all trials of the test session (transductive, label-free) and AdaBN on the test subject's trials — are these acceptable, what exactly must be disclosed, and is the comparison against baselines that did NOT use alignment fair? Propose the minimal set of arms to make the comparison fair (e.g., ANN+EA must be run).
4. Model-selection audit: lr chosen by validation macro-F1 on the pilot cohort then reused — fine? Early stopping per cell on validation subjects — fine? Where could test information leak (e.g., per-epoch test curves exist for monitoring; the code claims a guard)?
5. Multiple comparisons: with ~12 new entries + 18 old ones, what is the right pre-registered comparison family for the paper's claims (H0-H5), and what should be exploratory? Friedman+Nemenyi over 30 models vs a small pre-registered set?
6. Reporting: what must the paper state (single training seed, recipe dependence, transductive alignment, cohort floor effects)? Suggest the exact wording of the headline claim that the evidence could support.
7. Anything in cross_subject_bench.py's implementation of Wilcoxon/Holm/Friedman that is statistically questionable.

Return: (A) fatal flaws, (B) ranked improvements, (C) a concrete pre-registered analysis plan (tests, families, alpha, effect sizes, exact per-subject aggregation), (D) power table for the three cohorts, (E) 1-10 score for methodological rigor with justification. Do not modify any files.
```

</details>

# Methodological review — Spiking EEGNeX cross-subject MI plan (ICASSP 2027)

Scope: `PLAN_spiking_eegnex_icassp2027.md` §1.1/2.3/3/5, `cross_subject_bench.py`, `cross_subject_protocols.py`, `e2e_trainer.py`, and the completed 2C / SHU / 3C tables. No files were modified. All numbers below were recomputed from `per_cell.csv` / `per_subject.csv` of the snapshot runs.

## Empirical facts the plan should be built on (recomputed)

| Quantity | 2C (n=51) | SHU (n=25) | 3C (n=11, LOSO, 1 partition) |
|---|---|---|---|
| Between-subject SD (eegnex) | 11.5 pp | 10.0 pp | 13.6 pp |
| Paired SD, eegnex vs near-peer ANN (atcnet / eegnet / deepconvnet / deformer) | **1.36–2.46 pp** | **3.8–6.5 pp** | 3.0–5.7 pp |
| Paired SD, ablation-level variants (atcnet vs @noattn/@tcn8) | 0.9–1.3 pp | 1.7–1.9 pp | — |
| Partition noise: SD of a single-partition paired Δ around the subject mean | 1.0–1.6 pp | 2.5–4.2 pp | n/a |
| Correlation of the paired Δ between split 41 and 42 | 0.33–0.53 (0.83 deepconvnet) | 0.2–0.87 | n/a |
| Variance decomposition of Δ(eegnex−atcnet): stable subject×model vs partition | 1.9 vs 2.4 pp² | 37.6 vs 6.5 pp² | — |
| Fold-cluster ICC of per-subject Δ (shared trained model) | ≤ 0.08 | ≈0 (raw), 0.1–0.24 on partition residual | — |
| Shapiro–Wilk on paired Δ | rejected for most pairs (heavy tails) | not rejected | — |
| Binomial SE of one subject's accuracy | 1.5 pp (600 trials; 3 subjects have only 200) | 2.2 pp (~480 trials) | ~1.6 pp (900 trials) |
| Subjects below 55% (eegnex) | 0/51 | 5/25 (8/25 below 60%) | sub-011 at 39% (chance 33%) |

Two consequences. (i) The "4–6 pp paired SD" you quoted is SHU's; on 2C, peer-model paired SDs are 1.4–2.5 pp, so 2C is 3–4× better powered than the plan assumes. (ii) On 2C, partition (training-set draw) noise is *larger* than the stable subject×model heterogeneity, i.e. half of the per-subject paired Δ variance is retraining noise — which is exactly the component a single training seed leaves unmeasured.

---

## Answers to the seven questions

### 1. Validity and power of the paired subject-level analysis

Valid, with four caveats to disclose. Pairing efficiency is enormous: paired SD 1.4–2.5 pp vs between-subject 11.5 pp is a 20–60× variance reduction. Independence across subjects approximately holds — subjects in the same fold share a trained model, but the fold-cluster ICC is ≤ 0.08 on 2C and the two partitions re-randomize cluster membership, so the design effect on the subject-mean is a few percent at most. Caveats: (a) heteroscedasticity (200- vs 600-trial subjects; unweighted analysis is standard, disclose); (b) `set_seed(seed)` is called once per cell before `build_model`, and no RNG is consumed in between, so **every cell of a model shares the same initialization** — init is a fixed effect confounded with architecture, not a sampled nuisance; (c) SHU has 5–8 subjects at chance whose paired Δ is pure noise (dilutes, does not bias); (d) 3C has one partition (LOSO `seed0`), so no partition averaging.

Averaging the two partitions per subject before testing is correct as the primary analysis: the partition is a nuisance replicate of the same unit, and the subject mean is the natural estimator. Partition should *not* be a fixed factor in the primary test (that would double-count subjects). It belongs in a sensitivity mixed model d_{s,p} = μ + u_s + w_{trained-model(p,f)} + ε with crossed random effects, which yields the same μ̂ under balance but an SE that accounts for fold clustering, plus the variance decomposition above. Note the cheap-lever calculation: going from 2 to 4 partitions reduces the 2C paired SD only from 1.76 to 1.58 pp (MDE 0.70 → 0.63 pp); extra training seeds attack the unmeasured component and are worth more.

Power: see table (D). Headline numbers at 80% power, two-sided α=0.05: n=51, SD 1.5–2.5 pp → MDE 0.6–1.0 pp; SD 4–6 pp → 1.6–2.4 pp. n=25, SD 4–6 pp → 2.3–3.5 pp. n=11, SD 3–6 pp → 2.8–5.6 pp. Under Holm (family of 6, worst case α/6) multiply by ≈1.27.

### 2. H1 (interaction) — exact test, family, effect sizes, forking paths

Per-subject difference-of-differences: g_s = [a(SNN+A, s) − a(SNN, s)] − [a(ANN+A, s) − a(ANN, s)], each a(·, s) averaged over the two partitions. One-sample Wilcoxon signed-rank of g_s against 0 (exact when n ≤ 50, sign-flip permutation with 10⁵ resamples and fixed RNG at n=51 — or simply the permutation method for all cohorts). Under a balanced 2×2×subject design this is identical in estimand to the interaction contrast of a mixed model with subject random effect; the LMM adds a fold-clustering-aware SE and a logit-scale variant, so run it as sensitivity, not primary. Report: mean g with 95% BCa bootstrap CI, Hodges–Lehmann median g with 95% CI, d_z, rank-biserial r, wins/losses, and a 2×2 figure with per-subject lines.

Expected SD of g: if each alignment gain has SD s and the SNN and ANN gains correlate ρ across subjects, SD(g) = s·√(2(1−ρ)); with s≈3 pp, ρ≈0.5 → 3 pp → MDE 1.2 pp (α=.05) / 1.5 pp (Holm) on 2C, 1.75 / 2.25 pp on SHU, 2.8 / 3.8 pp on 3C. So H1 is a 2C-only confirmatory hypothesis.

Garden of forking paths — the plan currently leaves all of these free: SNN variant for the interaction (D, S×{lif,plif,hplif} = 5) × alignment arm (EA only, AdaBN only, EA+AdaBN = 3) × metric (acc, macro-F1, AUROC, bal-acc, κ = 5) × checkpoint (best/last = 2) × cohort (3) = **450 admissible "H1 tests"**. Any single significant one is uninterpretable. Pin exactly one primary: S-EEGNeX-S@hplif, EA+AdaBN vs neither, accuracy (bal-acc on SHU), best-val, 2C. Everything else is exploratory and reported without stars. Second confound: the interaction is trivially positive if the unaligned SNN starts lower than the unaligned ANN (more headroom, regression to the mean, ceiling at 99% subjects). Pre-register a gate: H1 is interpreted as an SNN property only if H0 (unaligned NI) also holds; and analyze g on the logit scale as sensitivity. Pre-register the mechanistic secondary: Spearman correlation between per-subject alignment gain and the subject's covariance shift from the training pool, and the across-subject CV of layer-1 firing rate before/after AdaBN (descriptive).

### 3. Leakage audit of EA and AdaBN

Both are label-free transductive adaptation and acceptable in cross-subject MI *if disclosed as such*. Nothing crosses subjects: EA uses only the session's own trials; AdaBN only the test subject's trials; `assert_disjoint` and the fit-free per-trial normalization are intact. What must be stated verbatim: EA fitted per session on all N≈200 unlabeled trials of that session including the test session; AdaBN re-estimated on all ~600 unlabeled trials of the test subject in one pass, affine parameters frozen, statistics reset per subject (not accumulated across the fold's test subjects — otherwise results depend on subject order); model selection (early stopping) did not use AdaBN; the "first-k-trials" curve for online feasibility. Two subtler points to disclose: (i) AdaBN is sensitive to the test batch's class prior — harmless on 2C (14800:14800) but SHU has 67/125 imbalanced sessions, so on SHU AdaBN implicitly uses the label *distribution*; that is why balanced accuracy should be primary there; (ii) specify the order "per-trial z-score → EA" or "EA replaces z-score", since the EA arm otherwise differs from the zscore arm in two ways.

Fairness: comparing S-EEGNeX+EA+AdaBN against 18 unaligned ANNs is invalid (architecture confounded with preprocessing), and the paper's "18 baselines" framing invites it. AdaBN cannot even be retro-applied to the old baselines — checkpoints were deleted. Minimal fair arm set: (a) every unaligned SNN entry vs the 18 unaligned ANNs (this is where the 18-baseline table belongs); (b) a 2×2 block per cohort: {eegnex, seegnex_stream@hplif} × {none, EA, EA+AdaBN} — already planned; (c) one cheap extra ANN with EA as an external anchor: eegnet_official+EA (3 k params, ~5 GPU-h on 2C; also the ANN twin of the S-EEGNet baseline); atcnet@noattn+EA on 2C only if budget allows. (d) Run the new eegnex at the *same* lr as the old one on at least one cohort: identical seed and partitions make it a direct reproducibility/nondeterminism check, and it tells you whether the old 18-baseline numbers are on the same footing.

### 4. Model-selection audit

Early stopping per cell on disjoint validation subjects: correct, and the code guard is real — `score` is derived from `val_metrics` only; `test_loader` metrics go into `history` and the log line but never into `_is_better`. Two residual leaks, both human: (i) the console prints `acc(train/val/test)` every epoch and `history` stores test curves, so anyone choosing lr/neuron/whether a pilot "worked" while watching those lines is using test information. Set `curves: false` for the pilot (or suppress the test term in the log) and enable curves only in the main run for H4. (ii) lr chosen by validation macro-F1 pooled over the 5 folds of SHU split 41: each SHU subject is a validation subject in some fold and the test subject in another, so this is flat (non-nested) CV on SHU — a small optimistic bias for SHU only (2-point grid, validation signal). For 2C/3C the lr is transferred from another cohort → clean. Make this virtuous: declare SHU the development cohort and 2C the confirmatory cohort, and state that pilot cells are reused in the final table (they will be — same `cell_signature`). Additional asymmetries to state: the 18 old baselines never had lr tuned (fixed 1e-4) while SNNs and the new eegnex get a 2-point validation-selected lr per model; patience 25 was tuned on ANN dynamics — report the fraction of SNN cells that hit `max_epochs=500` (censoring) vs early-stopped. Pre-register HetPLIF-S as the headline variant; if you ever need to swap, use `select_by_validation`, never test accuracy.

### 5. Multiple comparisons — the right families

Confirmatory family (2C only), Holm at α=0.05, five tests: H0-NI, H1-DoD, H2-NI, H3 (Page's L trend test for the ordered LIF ≤ PLIF ≤ HetPLIF — one test instead of two), B1 (S-EEGNeX-S vs S-EEGNet). Non-inferiority tests contribute a one-sided p (H: Δ ≤ −δ), so they fit in the same Holm family. SHU and 3C: same five estimates with CIs, Holm within cohort, explicitly labelled replication cohorts; the paper's claims rest on 2C. Exploratory (no stars): EA-only and AdaBN-only DoDs, D-version interactions, secondary metrics, last checkpoint, H4. Optional secondary cross-cohort test: stratified sign-flip permutation on standardized cohort means (one p for "direction replicates"), never pooled accuracies.

Friedman + Nemenyi over ~30 models is the wrong tool for the paper's claims: at N=51 the CD grows from 3.69 (k=18) to 5.09 (k=24) and 6.54 (k=30) average ranks, so the diagram would say nothing about the top half, and aligned variants of the same architecture are not independent "methods". Pre-register a restricted set: 18 unaligned ANN + 1 declared SNN (S-EEGNeX-S@hplif, unaligned), k=19, and since the question is one-vs-control, use Bonferroni–Dunn with S-EEGNeX as control (CD = 3.33 at N=51 vs 3.92 Nemenyi) or Holm on the one-vs-all rank z-scores (Demšar §3.2.2; García & Herrera 2008). Report S-EEGNeX's average rank and the models inside/outside CD.

### 6. Reporting obligations and headline wording

Must state: one training seed with shared initialization across cells (init variance unmeasured; if 3 seeds are added for the primary contrast, report seed SD); recipe dependence (3C rank order changed between recipes; lr grid and selection cohort; untuned old baselines; patience/censoring); transductive, label-free EA (per session, all trials) and AdaBN (per subject, all trials; class-prior sensitivity on SHU); cohort characteristics (SHU 5/25 subjects at chance, 3C n=11 with one non-decodable subject, both underpowered for < 3 pp effects); fold clustering; primary metric per cohort and best-val checkpoint; that the backbone was chosen on this benchmark's test results (winner's curse makes H0 *conservative*, but the "matches the best ANN" comparison is against the selected winner); energy model assumptions (45 nm Horowitz numbers, no memory traffic, fused BN, measured firing rates with per-subject spread).

Headline claim the design can support if H0 and H1 hold:
> "On 51 held-out subjects (WBCIC-SHU 2C), a temporal-native spiking EEGNeX is non-inferior to its ANN counterpart within a pre-specified 1 pp margin (paired Δ = x.x pp, 95% CI [l, u]) at an estimated k× (range across subjects) reduction in synaptic-operation energy; unsupervised subject-wise alignment (EA + AdaBN) improved the SNN more than the ANN (difference-of-differences x.x pp, 95% CI [l, u], Holm-adjusted p = …). The direction of both effects replicated in two smaller cohorts (n = 25, 11) that are individually underpowered."

If only H0 holds: "…alignment improved both networks by comparable amounts (interaction x.x pp, 95% CI [l, u]); after alignment the SNN reached x.x% vs x.x% for the ANN." Avoid "outperforms 18 baselines", "state of the art", and unqualified "significantly better".

### 7. Statistically questionable points in `cross_subject_bench.py`

1. Estimand mismatch: the p-value is Wilcoxon (symmetric location / pseudo-median), the CI is a Student-t interval on the mean, and d_z is mean/SD. With heavy-tailed 2C differences these can disagree in sign (atcnet vs @noattn: mean 0.29 pp, median 0.08 pp). Add a Hodges–Lehmann estimate with CI (Walsh averages) and a BCa bootstrap CI on the mean; report rank-biserial r. The plan's phrase "Δ and 95% CI from the paired Wilcoxon" is not what the code computes.
2. `method="approx"` forced with `correction=False`: at n=11 the normal approximation is ~10% anti-conservative vs exact (checked: 0.182 vs 0.206), ~4% at n=25. Use exact for n ≤ 50 and permutation at n=51 (or the permutation method everywhere; scipy 1.13 supports `PermutationMethod`). Uniformity across cohorts is not worth an anti-conservative test on the smallest cohort.
3. No one-sided / non-inferiority capability, which H0 and H2 need.
4. `subject_matrix` uses `pivot_table(aggfunc="mean")` with no coverage check: a model with one partition missing for some subjects is silently averaged over 1 partition for those subjects and 2 for the others. Assert `n_cells == n_folds × n_splits` per model before pairing.
5. `n` in the paired table is pre-zero-drop; `zero_method="wilcox"` silently reduces the effective n. Report n_eff.
6. Friedman family is "everything in the run", not pre-registered; Nemenyi (all-pairs) is used where the question is one-vs-control; blocks assumed independent (fold clustering); models with any missing subject are dropped before subjects. Iman–Davenport not used (conservative; irrelevant at p≈10⁻¹¹⁹).
7. `std_splits` is a df=1 SD from two values — not a usable uncertainty estimate.
8. `_holm` with `n_family` fixed at the pre-registered size is correct and conservative under partial sweeps; fine. `select_by_validation` is correct.
9. Pipeline gap for the new design: the alignment arm is `spec.normalization` (in the cell signature, not in the model name) and AdaBN is an eval-time switch that `run_cell` has no column for. `subject_matrix` pivots on `model`, so EA/AdaBN arms will either collide in one output directory or be invisible to the summarizer unless an "entry" key = (model, normalization, adabn) is introduced.

---

## (A) Fatal flaws (as planned, the stated claim would be unsupportable)

1. **H0 is a non-inferiority claim tested with a two-sided superiority test.** A non-significant two-sided Wilcoxon is not evidence of equivalence. It must be a one-sided CI-vs-margin test, and at δ = 1 pp it is only viable on 2C (power 0.76–1.00 at true Δ = 0 for SD 1.5–3 pp; 0.31–0.54 for SD 4–6 pp; SHU/3C essentially cannot demonstrate it).
2. **H1 is unpinned (≈450 admissible tests) and confounded by baseline level.** Without a single pre-declared variant/arm/metric/checkpoint/cohort, and a gate on H0, any reported interaction is post hoc.
3. **Aligned SNN vs unaligned 18 ANN baselines is not a valid comparison**, and AdaBN cannot be applied to the old baselines at all (checkpoints deleted). Only arms trained in the new run can be aligned.
4. **Single training seed with a shared initialization** makes init a fixed effect confounded with architecture. For a claim with a 1 pp margin, one draw of init per architecture is not sufficient evidence about the architecture.

## (B) Ranked improvements

1. Re-specify H0/H2 as non-inferiority (δ = 1 pp, justified as the spread of the top-8 ANNs on 2C = 1.0 pp) with one-sided CI bounds; declare 2C confirmatory, SHU development, 3C replication.
2. Pin H1's single primary test (S-EEGNeX-S@hplif, EA+AdaBN vs none, accuracy, best-val, 2C) + gate on H0; everything else exploratory.
3. Add 2 extra training seeds for the primary 2×2 contrast on 2C (eegnex, seegnex_stream@hplif × {none, EA}: 4 entries × 2 seeds × 10 cells ≈ 30–60 GPU-h); average seeds into the per-subject value; report seed SD. Fund it by dropping D-version LIF/PLIF variants.
4. Two-block main table: unaligned block (18 ANN + SNNs) and aligned 2×2 block; add eegnet_official+EA as a cheap external anchor; rerun eegnex at the old lr on one cohort as a reproducibility check.
5. Fix the test implementation: exact/permutation Wilcoxon, HL + BCa CIs, NI p-values, coverage assertion, entry key for alignment arms.
6. Replace Friedman-over-everything by the restricted k=19 set with Bonferroni–Dunn (control = S-EEGNeX); use Page's L for H3.
7. Disable test curves during the pilot; enable only in the main run for H4 (descriptive).
8. Balanced accuracy as SHU primary metric (imbalance + AdaBN class-prior sensitivity); accuracy for 2C/3C.
9. Mixed-model sensitivity analysis (subject + trained-model crossed random effects, logit scale) reported in one sentence.
10. Report censoring (SNN cells hitting 500 epochs), per-subject energy ratio distribution, and the covariance-shift ↔ alignment-gain correlation as the mechanistic exploratory figure.

## (C) Pre-registered analysis plan (to commit to git before any main-run summary)

**Unit and aggregation.** For entry e and subject s: a(e,s) = mean over partitions p ∈ {41, 42} (and over training seeds where present) of the best-val-checkpoint metric on all pooled trials of s in the fold where s is a test subject under p. Complete cases per contrast (drop a subject only if any entry in that contrast is missing; report n). Primary metric: accuracy (2C, 3C), balanced accuracy (SHU). Secondary: macro-F1, AUROC, κ, last checkpoint — descriptive only.

**Contrasts.** d_s = a(cand,s) − a(base,s). g_s (H1) as defined in Q2.

**Confirmatory family — 2C, Holm, α = 0.05, five tests:**
- H0-NI: mean d_s(S-EEGNeX-S@hplif − eegnex, unaligned); one-sided p for Δ ≤ −1 pp from the BCa bootstrap (B = 10⁴, RNG seed fixed); claim NI if the lower 95% (two-sided) bound > −1 pp.
- H1: Wilcoxon signed-rank on g_s (EA+AdaBN vs none), permutation method; gated on H0.
- H2-NI: S@hplif − D@hplif (unaligned), δ = 1 pp, same procedure as H0; SOP count is deterministic and reported separately.
- H3: Page's L test for the ordered alternative LIF ≤ PLIF ≤ HetPLIF (S version, unaligned); pairwise contrasts as secondary.
- B1: Wilcoxon, S-EEGNeX-S@hplif vs S-EEGNet@plif (unaligned).

**Replication cohorts (SHU, 3C).** Same five estimates and tests, Holm within cohort, exact Wilcoxon (n ≤ 50), labelled replication; no claim in the abstract rests on them. Optional: stratified sign-flip permutation across cohorts for H1 direction.

**Effect sizes per contrast.** Mean Δ (pp) + 95% BCa CI; Hodges–Lehmann median Δ + 95% CI; d_z; rank-biserial r; wins/ties/losses (tie = |Δ| ≤ 1e-12); n_eff after zero-drop.

**Exploratory.** EA-only and AdaBN-only DoDs; D-version interactions; secondary metrics; H4 (train–test gap at best epoch, per-subject Δ of gaps with CI, no test); H5 (per-subject energy ratio median/IQR/min, with and without AdaBN); Friedman over the restricted k = 19 unaligned set with Bonferroni–Dunn CD (control = S-EEGNeX); covariance-shift ↔ gain Spearman; firing-rate CV across subjects before/after alignment.

**Sensitivity.** Paired t on means; mixed model with crossed random effects (subject, trained model = partition×fold) on the pp and logit scales; per-partition results displayed separately.

**Selection rules.** lr from {1e-4, 1e-3} per model by validation macro-F1 on SHU split 41 (pilot cells reused); headline SNN variant fixed as S@hplif; no swap by test; `curves: false` in the pilot.

## (D) Power table

Minimum detectable paired mean difference (pp) at 80% power, paired t (Wilcoxon ≈ ×1.02 under normality, ≤ ×1 under the heavy tails observed on 2C):

| n | SD 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 |
|---|---|---|---|---|---|---|---|
| 51, α=.05 | 0.40 | 0.60 | 0.80 | 1.20 | 1.60 | 2.00 | 2.40 |
| 51, α=.05/6 (Holm worst case) | 0.50 | 0.76 | 1.01 | 1.51 | 2.02 | 2.52 | 3.03 |
| 25, α=.05 | 0.58 | 0.88 | 1.17 | 1.75 | 2.34 | 2.92 | 3.50 |
| 25, α=.05/6 | 0.75 | 1.13 | 1.50 | 2.25 | 3.00 | 3.76 | 4.51 |
| 11, α=.05 | 0.94 | 1.41 | 1.88 | 2.81 | 3.75 | 4.69 | 5.63 |
| 11, α=.05/6 | 1.28 | 1.91 | 2.55 | 3.83 | 5.10 | 6.38 | 7.66 |

Realistic SD ranges: 2C peer-model 1.4–2.5 pp (MDE 0.6–1.0 pp); 2C DoD ≈ 2–3 pp (MDE 0.8–1.2 pp); SHU 3.8–6.5 pp (MDE 2.2–3.5 pp); 3C 3–5.7 pp (MDE 2.8–5.3 pp). 95% CI half-widths: n=51: 0.4 pp (SD 1.5) to 1.7 pp (SD 6); n=25: 0.6–2.5 pp; n=11: 1.0–4.0 pp.

Non-inferiority power at δ = 1 pp (P[lower one-sided 95% bound > −1 pp]):

| n | SD | true Δ = 0 | true Δ = −0.5 | true Δ = −1 |
|---|---|---|---|---|
| 51 | 1.5 / 2 / 3 / 4 / 6 | 1.00 / 0.97 / 0.76 / 0.54 / 0.31 | 0.74 / 0.54 / 0.31 / 0.22 / 0.14 | 0.05 |
| 25 | 1.5 / 2 / 3 / 4 / 6 | 0.94 / 0.78 / 0.48 / 0.32 / 0.19 | 0.48 / 0.32 / 0.19 / 0.14 / 0.10 | 0.05 |
| 11 | 1.5 / 2 / 3 / 4 / 6 | 0.65 / 0.44 / 0.25 / 0.17 / 0.12 | 0.24 / 0.16 / 0.11 / 0.09 / 0.07 | 0.04 |

At δ = 2 pp, n=51 has power ≥ 0.97 for SD ≤ 4 pp at true Δ = 0 and 0.74–0.99 at true Δ = −1 pp. Small-n Wilcoxon floor: at n=11 the smallest exact two-sided p is 0.00098 (11/0 wins) and 0.0020 (10/1), so Holm-corrected significance on 3C requires ≥ 10 of 11 subjects in the same direction. Partition lever on 2C: P = 1/2/4/8 → paired SD 2.07/1.76/1.58/1.48 pp (MDE 0.83/0.70/0.63/0.59 pp) — small return relative to adding training seeds.

## (E) Rigor score: 6 / 10

Credit (what most SNN-MI papers lack): subject-level paired design on 87 subjects across three cohorts; validation subjects disjoint from test with a code-enforced guard; deterministic partitions enabling exact pairing with 18 existing baselines; a pre-registration culture (`PREREGISTERED_COMPARISONS`, `select_by_validation`, signature guards); effect sizes and Holm already in the pipeline; honest floor-effect and single-seed acknowledgements.

Debit: the primary hypothesis is mis-specified (NI tested as superiority); the mechanistic hypothesis has hundreds of unpinned test paths and a baseline-level confound; the fairness structure vis-à-vis the 18 baselines is not yet enforced by the arm set; a single shared-init seed for sub-1 pp claims; anti-conservative approximate Wilcoxon on the smallest cohort; 3C promoted to a "main table" despite MDE ≈ 3–5 pp; Friedman family will become uninformative as k grows; test curves visible during the pilot. Adopting improvements 1–6 moves this to 8/10 — the design would then be more rigorous than the current cross-subject MI literature, and 2C alone is well powered for the claims that matter.
