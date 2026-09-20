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
You are a world-class EEG/BCI researcher (motor-imagery decoding, FBCSP/Riemannian methods, cross-subject transfer, Euclidean/Riemannian alignment, EEGNet/EEGNeX-style deep models; think Wu Dongrui / Lotte / Blankertz-level expertise). You are reviewing a research plan for an ICASSP 2027 paper (deadline in 6 days) that ports EEGNeX to a spiking network and adds unsupervised subject alignment for cross-subject MI decoding.

Read the plan: /share/home/yuan/SYX/eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md
Background facts from the group's own completed benchmark (18 models x 3 cohorts, paired): /share/home/yuan/SYX/MEETING_BRIEF_2026-09-18.md (sections 1-3) and /share/home/yuan/SYX/eeg-mi-online/docs/RESULTS_TRUTH.md (3-class cohort, overfitting table in section 5.1).
The EEGNeX implementation being ported: /share/home/yuan/SYX/eeg-mi-online/code/models/paper_baselines/eegnex_official_torch.py
The existing Euclidean Alignment code that will be reused: /share/home/yuan/SYX/eeg-mi-online/code/methods/session_alignment.py (class EuclideanAlignment, 10% shrinkage toward tr/C * I)
Data: WBCIC-SHU 2C (51 subjects, 58 ch, L/R hand grasp, 3 sessions x 200 trials), SHU 2022 (25 subjects, 32 ch, L/R, 5 sessions ~96 trials), WBCIC-SHU 3C (11 subjects, 58 ch, L/R/feet). Preprocessing: 0.5-40 Hz, 50 Hz notch, 0-4 s epochs, 250 Hz, per-trial channel-wise z-score.

Review as a demanding senior expert. Specifically assess:
1. Section 1.2-1.3 of the plan: is the "EEGNeX = learnable FBCSP + multi-scale envelope dynamics" reading correct? Is the "LIF as envelope detector" and "first LIF at sampling rate = 1-bit level-crossing encoder that preserves phase for downstream spatial filtering" argument physiologically and signal-theoretically sound? Where does it break (e.g., EEGNeX applies ELU BEFORE the spatial depthwise conv; what does that imply)?
2. Alignment protocol: EA fitted per SESSION with all that session's trials (transductive, label-free), applied to train/val/test subjects independently; plus AdaBN per test subject. Is this the accepted standard in cross-subject MI? Leakage or fairness concerns relative to the 18 ANN baselines that used only per-trial z-score? Should EA be applied per session or per subject? Should per-trial z-score be applied AFTER EA or dropped (EA already normalizes scale)? What does the literature say about EA gains for deep models (He & Wu 2020; later works) - expected magnitude on 2-class MI with 51 subjects?
3. Any better or complementary "subject alignment" that is cheap to implement in 1-2 days and known to work for deep MI models (e.g., Riemannian alignment, label-alignment, per-subject spectral normalization for individual alpha frequency, channel-wise covariance re-referencing)? Rank by expected gain and implementation cost.
4. The claim that cross-subject shift is partly "posterior drift" so domain-adversarial invariance fails: do you agree? What does this imply for what the spiking model should and should not try to do?
5. The proposed hypotheses H0-H5 and datasets: what is missing that an EEG reviewer would demand (e.g., BCI IV 2a, within-subject reference, chance-level statistics, per-session breakdown)? Given 6 days, what is the minimum set?
6. Any physiological reason temporal-native spiking could be BETTER than the ANN for MI (e.g., ERD onset latency variability, mu/beta phase, sparsity), or reasons it could be worse (low SNR at single-sample level, 1-bit quantization destroying small ERD amplitude changes)?

Return a structured report: (A) fatal flaws, (B) important improvements ranked by expected impact/effort, (C) concrete protocol recommendations (EA scope, z-score order, AdaBN details, which cohorts/metrics), (D) what to drop given the budget, (E) 1-10 scores for scientific soundness and for expected accuracy gain of the alignment component, each with a one-sentence justification. Be concrete. Do not modify any files.
```

</details>

# Expert review — Spiking EEGNeX + unsupervised subject alignment (ICASSP 2027 plan v1)

Scope of evidence used: the plan; `MEETING_BRIEF_2026-09-18.md` §1–3; `RESULTS_TRUTH.md` §5.1; `eegnex_official_torch.py`; `session_alignment.py`; plus the group's own Phase‑2b no‑learning alignment tables (`2_baseline/{wbci_shu,shu}/alignment_baseline/`), Phase‑2c conclusions (`results.md`), the already‑wired `ea_session[+trial_norm]` grammar in `code/experiments/cross_subject_protocols.py`, `normalize_trials` (has `per_sample_global`), `bn_adaptation.py`, and the literature (Junqueira et al. JNE 2024; Wu group "Revisiting EA" JNE 2025; HR‑SNN / NiSNN‑A / SARNN). No files were modified.

---

## Verdicts on the six questions

**Q1 — §1.2/1.3 reading.** Half right. Correct: depthwise (C×1)×2 per feature map = per‑band spatial filters (FBCSP prior); dilated convs give ~0.5 s / ~1 s receptive fields; flatten preserves time. Wrong or overclaimed:

- "Two 64‑tap convs = a steeper filter bank." ELU sits between conv1 and conv2, so conv1∘ELU∘conv2 is a *nonlinear* spectro‑temporal feature bank, not a 127‑tap linear filter. ELU is asymmetric (identity for x>0, saturating for x<0): on a zero‑mean oscillation it yields a DC term ∝ amplitude (a partial envelope), the preserved carrier, and harmonics. BN1's learnable β sets the operating point: β≫0 → ELU≈identity → the pure linear FBCSP chain is recovered; β<0 → soft rectifier → conv2 smooths → envelope. EEGNeX can *learn* how linear its front end is, per band.
- Consequence for the port (the ELU‑before‑spatial point): the ANN's tolerance of a pre‑spatial nonlinearity does **not** transfer, because ELU has a linear regime and a hard threshold has none. This is the single biggest accuracy risk and the plan treats it as settled.
- "LIF = envelope detector." True only in the short‑τ regime (τ ≲ half a period). Rectification happens at the threshold, not in the integrator: a long‑τ LIF fed a zero‑mean oscillation low‑passes it toward zero (cancellation), it does not integrate its envelope. The envelope path in the S version is LIF₁ (rectify) → conv2 (smooth non‑negative spike trains, whose local mean *is* an envelope estimate) → LIF₂. Describe it that way. Also rate ∝ amplitude (√power) above threshold with a dead zone below, not log(var); the loss of log‑var's gain invariance is precisely why the SNN is scale‑sensitive (this is the *correct* mechanistic basis for H1, stronger than §1.4's wording).
- "1‑bit level‑crossing encoder preserves phase, so spatial filtering remains feasible." Not sound as written. (i) Crossing phase is amplitude‑dependent, φ ≈ arcsin(v_th/A), not fixed. (ii) Half‑wave: negative half‑cycles are discarded. (iii) The Σ‑Δ / time‑encoding‑machine reconstruction argument (Lazar & Tóth) needs subtractive (soft) reset, a positive bias so the neuron keeps firing, and spike density ≳ 2× bandwidth (r₁ ≳ 0.3 at 250 Hz for 40 Hz content) — which directly contradicts the r₁ ≈ 0.04–0.1 energy assumption. A τ=2‑step hard‑reset LIF is a single‑level comparator, not a Σ‑Δ modulator. (iv) Most importantly, CSP‑type spatial filtering is linear unmixing of x = A·s; a per‑electrode nonlinearity before mixing breaks the model, so cancellation of common/volume‑conducted components by weighted subtraction of spike trains does not hold. What survives is coarse spatial weighting of *rates* (lateralized power contrast, C3 vs C4) and coincidence detection. Make this a hypothesis and test it with the encoder‑position ablation (B1).

**Q2 — Alignment protocol.** EA fitted per recording with all its unlabeled trials is the accepted transductive standard (He & Wu 2020 "offline unsupervised"; Junqueira 2024 "offline‑EA", with "online‑EA" = first 24 trials as the realistic variant). Declare it and add the k‑trial curve (evaluation‑only, free). Per **session** is the right scope: your own drift diagnostics show between‑session spatial covariance drift (MMD 0.24–0.36, CSP similarity 0.34–0.42), and 96–200 trials ≫ the 16–32 needed. Fairness vs the 18 baselines: the paired structure survives, but the headline must be (model effect at fixed normalization) and (EA effect at fixed model), never "SNN+EA vs old ANN@zscore"; give the EA arm to EEGNet ANN as well (~5 GPU‑h on 2C) so H1 has two ANN references. Verify the anchor (new `eegnex@zscore` within ±0.5 pp of the old 83.35%) before co‑tabulating. Z‑score: **not** channel‑wise after EA — it re‑normalizes each whitened virtual channel per trial, destroying the diagonal of the aligned covariance EA just set up; use bare EA or a per‑trial *scalar* scale (`per_sample_global`, already in `normalize_trials`; grammar `ea_session+per_sample_global` already parses). Literature magnitude: +5.55 pp (all trials) / +4.33 pp (24 trials) for LOSO shared EEGNet/DeepNet/ShallowNet on 2a and HGD with 8–13 source subjects; EA ≈ RA; fine‑tuning adds nothing after EA; Kostas & Rudzicz report +8.9 pp on 4‑class. Expectation here is smaller: **+0.5–2 pp on 2C** (30 source subjects already teach partial covariance invariance, 83% ceiling, per‑trial z‑score already removes gain), **+1–3 pp on SHU**, 3C too small to resolve. Warning from your own data: cross‑session EA was −1.2 pp (EEGNet), −4.2 pp (DeepConvNet), +1.8 pp (FBCNet) on 2C (`alignment_by_model.csv`) — whitening amplifies low‑variance directions and CNNs pay for it; the variance‑feature model benefits. Two cheap mitigations: log/inspect `cond_` per session and raise shrinkage if >10³; compute R̄ from an 8–30 Hz copy of the trials and apply it to the 0.5–40 Hz input (0.5–40 Hz covariance is dominated by low‑frequency power, so the whitening barely targets the mu/beta subspace; Wu 2025 places EA *after* band‑pass for exactly this reason).

**Q3 — Cheap complementary alignment (ranked).** See table in B.

**Q4 — "Posterior drift".** Disagree with the wording, agree with the conclusion. Your own Phase‑2c result is "within‑class scatter inflation (Fisher 4.58→1.57), **not** centroid collapse": class means and the decision direction survive, spread balloons. That is conditional‑covariance shift plus gain/covariance shift plus illiteracy, not p(y|x) inversion (physiology fixes the contralateral sign). "Posterior drift" would argue against any unsupervised alignment and thus against H1 — the plan contradicts itself. DANN fails here because subject‑identifying features (IAF, mu topography, amplitude) *are* the task features, and with 30–40 domains the discriminator is weak/noisy while gradient reversal removes task information. Implication: the SNN should re‑reference inputs (EA) and re‑center currents (AdaBN) per recording with a shared decoder; no invariance losses, no subject heads; expect nothing for illiterate subjects; plot per‑subject Δ against baseline accuracy.

**Q5 — Missing items.** Per‑subject chance band (binomial one‑sided 95%: ≈53.4% at n=600 for 2C, ≈35.9% at n≈900 for 3C) with counts of subjects above chance; per‑session breakdown of test subjects (EA is per session); seed‑variance estimate (SNN surrogate‑gradient training varies more than ANN); explicit paired SD/CI and a justified non‑inferiority margin (2C paired SDs are ~1–2 pp, so a 1‑pp margin is testable; on 3C, N=11, it is not); within‑subject reference (you have the 3C ceiling; for 2C note that cross‑subject EEGNeX 83.4% exceeds the Phase‑0–2 within‑subject cross‑session EEGNet 71–77% — worth one sentence); SNN prior art (HR‑SNN 77.6% 4‑class 2a; NiSNN‑A; SARNN; the EMBC‑2021 conversion at −1.9 pp) — all within‑subject 2a, which is your differentiation. BCI IV 2a: reviewers expect it; attempt the loader on day 1 while GPUs are queued, hard cut‑off 9/20 morning. Minimum set is in D.

**Q6 — Physiology.** Could be better: LIF is a natural change detector (ERD as a rate drop; beta bursts of ~150 ms are transients an avgpool smears); saturation at ≤1 spike/step clips artifacts so blinks cannot dominate conv2 (the ANN has no clip); heterogeneous τ = learnable integration windows covering ERD‑onset latency variability (0.3–1 s); quantization noise as regularizer (H4). Could be worse: dead zone below threshold (low‑amplitude subjects; ERD pushes toward the dead zone); compressive rate coding of a 10–30% amplitude ERD; half‑wave 1‑bit quantization before unmixing (largest risk); binary readout (A5); BN‑threshold scale coupling at init (tdBN‑style scaling needed); SHU's near‑floor SNR.

---

## (A) Fatal flaws (as written)

1. **Energy is benchmarked against the wrong baseline.** EEGNeX's 993 M MACs are 32× EEGNet's ~31 M (29.7 M temporal + 0.9 M depthwise + ~0.1 M separable) for +0.8 pp on 2C. "15–20× vs EEGNeX" lands at ~50–60 M AC‑equivalent, i.e. EEGNet‑level energy. The defensible claim is "EEGNeX accuracy at EEGNet energy" — report MACs/ACs/energy for EEGNet ANN, S‑EEGNet and every variant in the same table, and state the floor (analog conv1 = 29.7 M MACs ≈ 137 nJ → max ≈33×).
2. **§1.3 encoder theory is incorrect** (Q1). Rewrite as a hypothesis; the encoder‑position ablation (B1) is its test. Also make soft reset the default for LIF₁ if you keep any Σ‑Δ/TEM language.
3. **"Posterior drift" contradicts H1 and Phase‑2c** (Q4). Rewrite as "subject‑specific gain/covariance shift and within‑class scatter inflation".
4. **Streaming/"anytime" overclaim.** Causal convs/pooling do not make the model streaming; the flatten+FC head reads the full 4 s. Drop "anytime" or add a causal readout (FC on truncated features trained with random truncation).
5. **Readout/topology quantization buys nothing and costs accuracy.** Pool→LIF at stages 2 and 3 gives conv3 and the FC binary inputs (the FC sees 504 bits). conv3+conv4+FC are <1% of MACs, so binarizing them saves nothing. Follow the ANN order (LIF→pool: graded counts) or use a membrane‑potential readout; make "all‑binary" the ablation. The energy story lives entirely in conv2, i.e. in r₁.
6. **Process risk.** No SNN code yet, a 1000‑step LIF₁ loop, 4 free GPUs. Use spikingjelly's multi‑step LIF with the cupy backend (or a jit‑fused loop) and set a hard go/no‑go at the smoke test: if s/epoch > 3× `eegnex`, the plan's fallback "LIF₁ after avgpool" *is* the analog‑front‑end variant — conv2 becomes analog and the energy claim dies. Say so explicitly if that happens.

## (B) Important improvements, ranked by impact/effort

| # | Change | Expected impact | Effort |
|---|---|---|---|
| 1 | **Encoder‑position ablation**: LIF₁ before conv2 (plan) vs first spikes after the depthwise spatial conv (analog front end). Decides both the accuracy risk and the energy claim. | Very high (crux of the paper) | ~2 h; the after‑spatial variant is 250 steps, faster |
| 2 | Energy table vs EEGNet ANN and S‑EEGNet; per‑layer MAC/AC + measured r per test subject | High (fixes A1) | Free |
| 3 | LIF→pool order at stages 2–3 or membrane readout (fixes A5) | High | 1 h |
| 4 | EA arm on `eegnet_official` too; anchor reproduction check | High for H1's credibility | ~5 GPU‑h + free |
| 5 | Trial norm after EA: `ea_session` vs `ea_session+per_sample_global` (never channel‑wise); pick on pilot val macro‑F1 | Medium–high for SNN threshold stability | Already supported |
| 6 | Band‑specific EA reference: R̄ from an 8–30 Hz FIR copy, applied to 0.5–40 Hz data | Medium (+0.5–1 pp plausible) | ~30 lines (firwin code exists in `FilterBankReweight`) |
| 7 | Soft (subtractive) reset for LIF₁; signed ±LIF encoder (two polarities, mirrors ADM neuromorphic front ends) | Medium; recovers half‑wave loss | 10 lines each; one pilot config each |
| 8 | AdaBN ablations at eval only: all BN vs bn1‑only; k‑trial calibration curve for both EA and AdaBN | Medium (answers "transductive" and "where does it act") | Free |
| 9 | Per‑subject chance band, above‑chance counts, Δ‑vs‑baseline scatter, per‑session accuracy | Medium (reviewer hygiene) | Free |
| 10 | Log `cond_` per session; shrinkage 0.1→0.2–0.3 if cond>10³ (or Ledoit–Wolf) | Medium (own data shows whitening can hurt CNNs) | Free |
| 11 | Riemannian alignment (log‑Euclidean, implemented) | Low (≈EA in your data and in Junqueira) | Free; robustness footnote only |
| 12 | Retrain one 2C `eegnex` fold and report bn1 β / fraction of negative pre‑activations | Low–medium (evidence for the "learnable linearity" story) | 1–2 h |
| 13 | Two training seeds for the two headline models on SHU | Low–medium (quantifies SNN seed variance) | ~8 GPU‑h |
| 14 | IAF spectral warping / TENT / EA‑to‑reference‑covariance | Promising but risky | Not for this deadline |

Label alignment is not applicable (needs target labels). Common‑average re‑referencing would change the preprocessing shared with the 18 baselines — skip.

## (C) Concrete protocol

- **EA**: per session, all trials, 0.5–40 Hz input; R̄ from an 8–30 Hz copy (B6); shrinkage 0.1 unless `cond_` says otherwise; applied identically to train/val/test subjects; k‑trial curve at k ∈ {10, 20, 40, all} at evaluation. Note SHU's imbalanced sessions bias R̄ toward the majority class — mention it.
- **Trial norm**: anchor arm `per_sample_zscore` (identical to the 18 baselines); EA arm `ea_session` or `ea_session+per_sample_global`, chosen on pilot validation.
- **AdaBN**: per test subject, all trials, `reset=True, cumulative=True, n_passes=1` (exact stats, as implemented); checkpoints selected without AdaBN, then evaluated with and without; ablate bn1‑only; log per‑layer firing rates before/after for the H1 figure; for the SNN, apply the same procedure to BN layers preceding LIF (that is where the "homeostasis" acts).
- **Init**: scale BN before each LIF so pre‑activation std ≈ v_th (tdBN‑style); parametrize τ via sigmoid (PLIF); HetPLIF log‑uniform [1.5, 16] steps is fine.
- **Cohorts/metrics**: 2C main (Acc, macro‑F1, AUROC, balanced Acc), SHU secondary with balanced Acc mandatory, 3C conditional; per‑subject paired Δ with bootstrap CI and fraction improved; three pre‑registered Holm families: (i) SNN vs ANN at fixed norm, (ii) EA vs zscore per model, (iii) interaction (ΔSNN − ΔANN per subject, Wilcoxon). Non‑inferiority margin 1 pp on 2C only.
- **Pilot (SHU, split 41, 5 folds)**: `eegnex`×{z, EA}; `seegnex_stream@hplif`×{z, EA}×{1e‑4, 1e‑3}; `seegnet@plif`×{z, EA}; `seegnex_D@hplif`×{z, EA}; plus single‑config arms: after‑spatial encoder (EA), soft‑reset LIF₁ (EA), `ea_session+per_sample_global`, band‑specific R̄. ≈14 configs ≈ 20–25 GPU‑h. Select everything on val macro‑F1.

## (D) Drop given the budget

- D‑version on 2C/3C (keep it in the pilot; promote only if the S version fails H0).
- LIF/PLIF/HetPLIF three‑way on 2C and 3C → SHU only.
- 3C reduced to {eegnex, seegnex_stream@hplif} × {z, EA}; run last, or drop if the queue slips.
- `spike_all`, `T=8`, and the "anytime" curve.
- H4 as a hypothesis (a lower‑train‑accuracy model trivially has a smaller gap) → report train/test pairs as an observation.
- BCI IV 2a unless the loader works by 9/20 morning.
- Minimum publishable set: 2C + SHU × {eegnex, seegnex_stream@hplif, seegnet@plif} × {z, EA} × AdaBN{off, on}, both split seeds on 2C, plus B1 and the energy table.

## (E) Scores

- **Scientific soundness: 6/10 as written, 8/10 with A1–A5 fixed** — the paired 87‑subject factorial design is excellent, but the encoder theory, the energy baseline and the "posterior drift" framing are each wrong in a way a competent reviewer will catch.
- **Expected accuracy gain of the alignment component: 5/10** — EA+AdaBN should give +0.5–2 pp on 2C and +1–3 pp on SHU with a real chance of a null on 2C, well below the literature's +4–5 pp, because 30 source subjects, per‑trial z‑score and an 83% ceiling already absorb most of what EA fixes; the SNN‑specific interaction (H1) is the more likely positive result.
