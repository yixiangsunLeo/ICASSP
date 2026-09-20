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
You are a leading researcher in spiking neural networks (direct training with surrogate gradients, PLIF/heterogeneous neurons, temporal encoding, energy accounting; think Fang/Zenke/Eshraghian/Yao-level expertise). You are reviewing a research plan for an ICASSP 2027 paper (deadline in 6 days) that ports EEGNeX (a compact CNN for motor-imagery EEG) to a spiking network for cross-subject decoding.

Read the plan: /share/home/yuan/SYX/eeg-mi-online/docs/PLAN_spiking_eegnex_icassp2027.md
Read the ANN being ported: /share/home/yuan/SYX/eeg-mi-online/code/models/paper_baselines/eegnex_official_torch.py
A draft spiking implementation (repeat/direct encoding only; temporal-native 'stream' encoding is NOT yet implemented): /tmp/legacy_peek/SYX/eeg-mi-online/code/models/spiking/neurons.py and /tmp/legacy_peek/SYX/eeg-mi-online/code/models/spiking/models.py

Context you need: EEG trials are [58 or 32 channels x 1000 samples @ 250 Hz], 2 or 3 classes, cross-subject (train on some subjects, test on unseen subjects). Training recipe is fixed: Adam, batch 128, early stopping patience 25 on validation macro-F1, lr to be piloted (1e-4 vs 1e-3). The ANN EEGNeX has 58k params, ~993M MACs/trial (95% in conv2). Hardware: RTX 4090D 24GB, 4 GPUs, ~6 days total including writing.

Review with brutal honesty, as a top expert would. Specifically assess:
1. The temporal-native ("stream") design: LIF1 running at the EEG sampling rate on conv1's output (1000 sequential steps), the "1-bit level-crossing encoder" interpretation, whether phase information really survives for downstream spatial filtering, and whether hard reset vs soft reset vs no-reset (PSN-style parallel neuron) is the right choice here. Is tau=2 steps at 250 Hz sensible for a ~10 Hz oscillation? What tau init range would you use for heterogeneous PLIF at each layer's rate (250 Hz, 62.5 Hz, 15.6 Hz)?
2. Training stability risks: BN placement (BN over folded T*B vs tdBN), threshold 1.0 with BN'd inputs, ATan alpha, detach_reset, dropout on spikes, the rate read-out (mean of FC over steps) vs reading the FC at the last step / membrane-potential readout. Vanishing spikes / dead neurons risk in layer 3 (only 63 steps, 8 channels). Concrete fixes.
3. Compute: is a Python loop of 1000 steps over a [B,8,58] state feasible in training time? Best practical way to accelerate in plain PyTorch 2.7 (torch.jit.script fusion, torch.compile of the step, CUDA graphs) without external libraries. Any correctness pitfalls of the straight-through surrogate formulation inside TorchScript.
4. Energy accounting: is "AC = dense MACs x input firing rate" correct for conv layers with binary input? Is counting conv1 once (analog input) fair? Are 4.6 pJ/MAC and 0.9 pJ/AC (45 nm) the accepted numbers and what should be cited? Does the "chunk"/"stream" claim that SOPs do not grow with T hold?
5. Novelty vs. existing spiking-EEG/MI literature you know (e.g., spiking EEGNet variants, SpikeNet-like models, the 2025 "lightweight SNN for MI" in Neural Networks, ISAM-MTL, spiking transformers for EEG). What exactly would make this stand out, and what would a strong SNN reviewer attack?
6. Anything in the draft code that is wrong or suboptimal (read neurons.py and models.py carefully: neuron dynamics, surrogate, rate bookkeeping, op table, chunk mode).

Return a structured report with: (A) fatal flaws (must fix before running), (B) important improvements ranked by expected impact/effort, (C) specific parameter recommendations (tau init ranges, thresholds, lr, T, alpha), (D) what to drop given the 6-day budget, (E) a 1-10 score for technical soundness and for novelty with one-sentence justification each. Be concrete; cite equations or code lines where relevant. Do not modify any files.
```

</details>

# Expert review — Spiking EEGNeX for cross-subject MI (ICASSP 2027 plan v1)

Scope: read `PLAN_spiking_eegnex_icassp2027.md`, `eegnex_official_torch.py`, `keras_compat.py`, the draft `neurons.py` / `models.py` / `adapter.py`, and the bench recipe (`shu_cross_subject_bench.yaml`: Adam, lr 1e-4, batch 128, wd 0, no clipping, no scheduler, fp32, per-trial z-score). Torch is not installed on this login node, so the code review is static; the quantitative claims below were checked with a small pure-Python simulation of the exact neuron equations in `neurons.py` (deleted afterwards, nothing in the repo was touched).

Bottom line up front: the *evaluation* side of this plan (87 subjects, 3 cohorts, paired Wilcoxon vs 18 ANN entries) is far stronger than anything in the SNN-MI literature. The *neuron* side has six concrete defects that will make the fixed recipe (lr 1e-4, patience 25 on val macro-F1) silently fail or make H3 a null result by construction. All are cheap to fix; none of them should cost GPU time.

---

## 1. Temporal-native ("stream") design

**1.1 What LIF1 at 250 Hz actually is.** With `decay_input=True` (`neurons.py` L151: `h = v + (x − v)·k`), the membrane is a first-order low-pass with unity DC gain, cutoff \(f_c = -\ln(1-1/\tau)\, f_s / 2\pi\). Followed by threshold + hard reset it is a **half-wave, dead-zone, refractory threshold detector**, not a level-crossing ADC:

| τ (steps @250 Hz) | gain @10 Hz | gain @20 Hz | f_c |
|---|---|---|---|
| 1.5 | 0.98 | 0.92 | 43.7 Hz |
| 2 | 0.94 | 0.82 | 27.6 Hz |
| 4 | 0.76 | 0.50 | 11.5 Hz |
| 8 | 0.47 | 0.26 | 5.3 Hz |
| 16 | 0.25 | 0.13 | 2.6 Hz |

Simulating a unit-variance BN'd input (10 Hz sinusoid of amplitude A + noise) through the exact `neurons.py` dynamics, θ = 1, hard reset:

| τ | A=0 | A=1.0 | A=1.4 | A=2.0 |
|---|---|---|---|---|
| 2 | 0.036 | 0.058 | 0.085 | 0.120 |
| 4 | 0.004 | 0.011 | 0.038 | 0.080 |
| 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | 0.000 | 0.000 | 0.000 | 0.000 |

So: **τ = 2 at 250 Hz is sensible** (it is the level-crossing regime; the neuron tracks the oscillation and fires a phase-locked burst at each positive peak). **τ ≥ 8 is dead** for a zero-mean oscillation: a leaky integrator averages a zero-mean sinusoid to ≈0, so it never crosses θ regardless of amplitude. This is the central physics error in §1.3(1) of the plan: "LIF = rectifier + leaky integrator + threshold ⇒ envelope detector" has the order wrong. Rectification happens *after* integration, so long-τ LIF on a *linear* conv1 output does not compute band power — it computes nothing. Envelope behaviour only arises when τ is short relative to the half-period (burst length ∝ time above threshold), or when the input already carries a positive mean (spike *rates* from a previous layer, which is why longer τ *is* meaningful at LIF2/LIF3).

Consequence: the plan's HetPLIF init range [1.5, 16] applied at LIF1 gives, for 8 channels log-uniform, ~3 of 8 conv1 filters born silent — conv2 loses 3/8 of its input at initialisation. Fatal for the S version (see A2).

**1.2 Spike count vs amplitude (pure 10 Hz sinusoid, τ = 2, hard reset): spikes/cycle = 0 (A=1.05), 1 (1.2), 2 (1.5), 3 (2.0), 7 (3.0).** It is a monotone, roughly linear-above-threshold *amplitude* code with a dead zone — not power, not log-variance. Do not claim FBCSP `log(var)` equivalence in the paper; say "monotone amplitude code".

**1.3 Does phase survive?** Yes, but only in the fundamental. Spike times are locked to the peak (first spike of the burst at the up-crossing asin(θ/A) — note this crossing time shifts with amplitude, a mild phase–amplitude coupling artefact). conv2 (64-tap FIR per (filter, electrode), no cross-electrode mixing) reconstructs a band-limited version whose phase is preserved, so the depthwise spatial filter can still exploit inter-electrode phase/polarity relations (a dipole's +s/−s electrodes produce half-period-shifted spike trains whose fundamentals subtract correctly). What is lost is **linear superposition**: (i) sub-threshold electrodes contribute exactly 0, not a scaled copy; (ii) the DC/rate component is unsigned, so spatial filters acting on rate-channels combine *amplitudes*, not signed signals; (iii) common-mode artefacts that saturate all electrodes cannot be cancelled linearly afterwards. Two consequences: the "1-bit level-crossing encoder" wording is loose (LC-ADC and the neuromorphic ADM front-ends emit *signed* UP/DOWN events; cite Lazar & Tóth 2004 TCAS-I for time-encoding machines, Corradi & Indiveri 2015 TBioCAS and Sharifshazileh et al. 2021 Nat. Commun. for ADM EEG front-ends, and Petro/Kasabov/Kiss 2020 TNNLS for TBR/BSA EEG encoders), and there is a cheap fix if the pilot shows a gap: **ON/OFF channels** (feed ±BN1 output to two populations → 16 spike channels; conv2 in_channels 8→16, +16k params, ACs unchanged because each polarity fires half the time). A more elegant fix that also has theory behind it: a positive bias current (BN1 β > 0). Lazar–Tóth: with bias b > max|x| an IF neuron encodes a band-limited signal losslessly in its ISIs. β trades energy (rate) against information — a nice explicit knob for the paper. Verify empirically with a spike–phase-locking histogram (Hilbert phase of conv1 output at spike times) on a trained model; it is a cheap and convincing figure.

**1.4 Hard vs soft vs no reset.**
- LIF1: hard reset is right. It gives one phase-locked burst per cycle with a 2-sample refractory (max rate 0.5) and the lowest firing rate (energy). Soft reset ≈ same at A ≤ 1.4, higher at A = 2 (0.20 vs 0.12).
- LIF2/LIF3 (rate regime, feeding linear temporal integrators conv3/conv4): **soft reset** (reset-by-subtraction) keeps rate ≈ linear in input (Rueckauer 2017; Han et al. RMP-SNN 2020); hard reset saturates (rate ∈ {1/k}). Make `hard_reset` per-layer, default hard@LIF1, soft@LIF2/3, and ablate.
- No reset (PSN/sliding-PSN style, Fang et al. NeurIPS 2023): rate becomes the duty cycle above threshold (0.32 at A=2, 2–3× hard), but the membrane is then a *linear* filter → **fully parallel, no 1000-step loop**. Since conv1 is already a learnable FIR, "no-reset leaky integrator + Heaviside" is just Θ(exp-kernel ∗ conv1(x) − θ). This is the compute escape hatch (§3). Reviewers will call it a binarised activation; PSN got through NeurIPS with the same critique, so report it as an ablation either way — if `none ≈ hard`, be honest that the reset is not doing work.

**1.5 τ init ranges per layer rate (log-uniform, in steps):**
- LIF1 @250 Hz: **[1.5, 4]** (6–16 ms). Must pass 8–30 Hz; τ = 4 already halves 20 Hz. HetPLIF adds little here anyway (redundant with conv1's learnable taps) — put heterogeneity where the multi-timescale story lives.
- LIF2 @62.5 Hz: **[1.5, 8]** (24–128 ms). Short-τ channels stay phase-locked (10 Hz period = 6.25 samples), long-τ channels integrate rate/envelope inputs from conv2's low-pass kernels. τ = 16 (gain 0.07 at 10 Hz) is dead unless the channel carries DC; cap at 8.
- LIF3 @15.6 Hz: **[1.5, 8]** (96–512 ms) — ERD/ERS time scales. If you adopt the LI (non-spiking) readout recommended in §2, no dead-neuron risk and [2, 16] is fine.

---

## 2. Training stability

**2.1 BN placement.** `_seq_apply` (`models.py` L77–80) folds T into batch = tdBN (Zheng et al. AAAI 2021) with α·V_th = 1. Fine for the D version. In the S version there is no folding at all: time is the tensor's last axis and BN is the ANN's plain `BatchNorm2d` — good, but note it for the paper (no "tdBN" claim needed there). TEBN is not worth the time.

**2.2 Threshold 1.0 with BN'd inputs — LIF3 is born dead.** BN(σ=1) → avgpool(4) → leaky integration shrinks the pre-threshold std to 0.29–0.54 depending on temporal correlation, so θ = 1 sits at **1.8–3.5 σ**: P(h > θ) ≈ 0.03 %–3 %. The surrogate at h−θ ≈ −1 with α = 2 is 1/(1+π²) ≈ 0.09, so gradients exist but BN5 β must travel ~0.8; Adam at lr 1e-4 needs ~8 000 steps (~100+ epochs at ~60–80 batches/epoch) → **patience 25 on val macro-F1 kills the run at chance**. At lr 1e-3 it takes ~800 steps, so the lr pilot would "choose" 1e-3 for the wrong reason (initialisation) rather than optimisation. Fixes, cheapest first: (a) init BN β = +0.5 (≈θ/2) for `bn1`, `bn3`, `bn5` (the BNs feeding a neuron) and γ = 2 for `bn3`, `bn5` to undo the pooling shrinkage; or (b) a one-batch data-dependent threshold/BN calibration at init targeting r ≈ 0.10–0.20 per layer; plus (c) a cheap homeostatic regulariser \(\lambda\sum_l (\bar r_l - r^\*)^2\), λ ≈ 1e-3, r* = 0.1 (Zenke & Vogels 2021 use activity bounds the same way) as a safety net for dead/saturated layers. Do (a)+(c) before the pilot; otherwise the pilot is uninterpretable.

**2.3 ATan α = 2**: correct SpikingJelly default; g'(0) = 1, half-width 0.32 — appropriate for σ ≈ 1 pre-activations. Not a lever. `detach_reset=True` with the Zenke & Vogels 2021 citation is right (SpikingJelly's default is False; you differ deliberately — say so).

**2.4 Dropout on spikes — an SNN-specific train/test shift.** `nn.Dropout(0.5)` after `lif3`/`lif5` (`models.py` L232, L237) rescales kept spikes to 2. Downstream conv3 then sees {0,2} at rate r/2 in training and {0,1} at rate r at test: means match, but variance is 2r−r² vs r−r², so BN4's running variance is ≈2× the test variance → at test the LIF3 input is ≈0.71× the training scale. The ANN tolerates a 30 % scale change (ELU is quasi-linear); a threshold turns it into a several-fold firing-rate change (see the σ-table above). Fix: mask-only dropout on spikes (no 1/(1−p) rescale), or p = 0.25, or apply dropout to the input current before the neuron. This applies to both D and S.

**2.5 Read-out.** D version: `classifier(feats).mean(0)` = FC(mean rate) since FC is linear — correct and standard; last-step readout would be worse at T = 4 (τ = 2 transient: 0.5, 0.75, 0.875, 0.94 of steady state). S version: FC on a flattened **8×63 binary** map with ~3–5 % rate = ~15–25 ones per trial — a severe bottleneck compared with 504 analog features. Worse, `constrain_max_norm(classifier, 0.25)` (`models.py` L212–213) bounds logits by 0.25·√k ≈ 1.25 for k = 25 active features → max softmax prob ≈ 0.78 (2-class) / lower for 3C → CE cannot fall and the head is permanently under-confident. **Recommendation: make the last neuron a non-spiking leaky integrator (LI) and read its membrane** (standard: SpyTorch/snnTorch, Cramer et al. SHD baselines, Yin et al. Nat. MI 2021). FC is 1 k MACs, energetically irrelevant; the "fully spiking LIF3" becomes the ablation. If you keep spikes at layer 3, raise the FC max_norm to ≥ 1 or drop it.

**2.6 Gradient flow.** Through the membrane the gradient decays as \((1-1/\tau)^k\prod(1-s)\): contractive, no explosion (no clipping needed), effectively ≤10–30 steps of temporal credit assignment — fine, conv2/3/4 do the long-range integration. No BPTT truncation needed.

**2.7 AdaBN on an SNN needs iterating.** Re-estimating BN1 changes LIF1 spikes, which changes BN2's correct statistics, etc. Do 2–3 full passes with cumulative averaging (or layer-sequential), not the single ANN-style pass. Also note a mechanistic prediction that strengthens H1: BN1 has one threshold per filter *shared across 58 electrodes*; EA whitening equalises per-electrode scale before that shared threshold, something the ANN's BN+ELU never needed. Write that down as the hypothesised mechanism.

---

## 3. Compute

**3.1 Feasibility.** Eager loop over `[128, 8, 58]`: ~10 launches/step forward + ~20 backward + Python overhead → ~250–300 ms per batch for LIF1 alone, i.e. 3–6× the ANN's batch time; LIF2/LIF3 loops are launch-bound but small (~20 ms). Memory is fine (1000 × 2 saved tensors × 237 KB ≈ 0.5 GB). So the plan's "S ≈ 2×" holds only with fusion; unfused it is 4–7×. The **D version** estimate ("2.5×") is optimistic: conv2 runs on `[T·B, 8, 58, 1000]`, i.e. 4× the ANN's conv work (~4× time), and the activations (`[512, 32, 58, 1000]` fp32 = 3.8 GB per tensor, several alive in backward) push 15–20 GB at batch 128 on a 24 GB 4090D — use `micro_batch_size: 64` (the trainer supports accumulation; BN-per-micro-batch caveat is already documented) or expect OOM.

**3.2 Best practical acceleration in plain torch 2.7, in order:**
1. **CUDA graphs** via `torch.cuda.make_graphed_callables(lif1, (sample,))` on the neuron module (training only): removes launch overhead for forward+backward (~5–10× on this loop). Requirements: static shapes (`drop_last=True` or pad the last batch), no CPU syncs inside (the `float()` in `mean_rate` is eval-only, fine), no RNG inside (dropout is outside, fine). Fall back to eager in eval where batch sizes vary.
2. **`torch.jit.script` the loop as a free function** (not the module; `autograd.Function` and parametrized modules are not scriptable). NNC fuses the body to ~2–3 kernels/step; the loop itself stays a TorchScript loop (no unrolling). Warm-up: the profiling executor runs 2 unfused iterations first; shape changes trigger re-profiling — use `torch._C._jit_set_fusion_strategy([("DYNAMIC", 20)])` or fixed shapes.
3. **Do not `torch.compile` the loop**: Dynamo unrolls 1000 iterations into one graph (minutes-to-hours compile, recompiles on the ragged last batch). Compiling only the step function costs ~20–50 µs of guard overhead per call, which negates the gain on 59 k-element tensors.
4. **Zero-loop path**: no-reset LIF1 (exponential kernel as a truncated depthwise conv1d over time, learnable per-channel τ) — parallel, exact for τ ≤ 4 with ~20 taps.

**3.3 Surrogate inside TorchScript — pitfalls.** Use the straight-through form `s_hard = (h − θ ≥ 0).to(dtype); s = s_hard + (g(h−θ) − g(h−θ).detach())` with \(g(x) = \tfrac{1}{\pi}\arctan(\tfrac{\pi}{2}\alpha x) + \tfrac12\), whose derivative is exactly the ATan surrogate in `neurons.py` L47. Forward is exactly binary (the correction term is identically 0 in fp32). Three traps: (i) for `detach_reset=True` the **reset must use `s_hard`** and the output `s`; using the STE spike in the reset silently gives `detach_reset=False`; (ii) hoist `k = sigmoid(w)` and its reshape out of the loop (the draft calls `self.decay(x_seq[t])` on every step, L150 — 1000 redundant sigmoid+view kernels); (iii) keep fp32 — under bf16 autocast the threshold comparison and the 0.3-wide surrogate degrade. Validate once: scripted vs eager `ATanSurrogate` gradients on random inputs must agree to 1e-6.

---

## 4. Energy accounting

- **AC = dense MACs × input firing rate** is correct for a conv with binary input: each spike costs fan-out = C_out × k_taps ACs (conv2: 2048), so ACs = N_spikes × 2048 = 950 M × r₁. 'same'-padding edge effects make it a ≤6 % overestimate; standard, ignore. Two omissions to footnote: the folded-BN bias (1 add per output per step: conv2 1.86 M ACs ≈ 2 % of the spike-driven ACs at r₁ = 0.1) and neuron updates (S version 481 k per trial ≈ 2 µJ, negligible but count them — reviewers ask).
- **Counting conv1 once, analog, at 4.6 pJ/MAC is fair and necessary** (it is the encoder; excluding it is the classic sin of direct-encoding papers). In the S version it is simply an FIR front-end at the sample rate — exactly what any EEG DSP or ADM chain needs.
- **The numbers**: Horowitz, "1.1 Computing's energy problem (and what we can do about it)", ISSCC 2014, 45 nm: FP32 add 0.9 pJ, FP32 mult 3.7 pJ → MAC 4.6 pJ, AC 0.9 pJ. Cite Horowitz directly; cite Yao et al. NeurIPS 2023 (Spike-driven Transformer) / Zhou et al. ICLR 2023 (Spikformer) / Rathi & Roy only as examples of the convention. Note some papers use INT32 (MAC 3.2, AC 0.1 pJ) — yours is the conservative choice; say so. Anticipate the attack that a deployed EEGNeX would be INT8 (≈0.2 pJ/MAC): report SOP/MAC counts as the primary, hardware-agnostic metric and pJ as secondary, and state that memory traffic is excluded.
- **Arithmetic check** (C = 58): conv1 29.7 M, conv2 950.3 M (95.7 %), dw 3.7 M, conv3 8.2 M, conv4 1.0 M, total 992.9 M → E_ANN = 4.57 mJ. S version analog floor (conv1 + dw + conv4) = 158 µJ → **ratio ceiling 28.8×**; at (r₁, r₃) = (0.05, 0.05) 22.7×, (0.1, 0.1) 18.7×, (0.2, 0.1) 13.8×, (0.3, 0.2) 11.0×. D version T = 4: 8.0× (r₁ = 0.1), 5.0× (0.2), 3.6× (0.3); T = 8 roughly halves that. The plan's "15–20×" and "~5×" are consistent; put ≥10× (H5) as the claim, not 20×.
- **Hybrid honesty**: the plan text says "LIF after avgpool ensures all downstream inputs are 0/1", but depthwise (input BN2, analog) and conv4 (input BN4, analog) are analog layers — the op table handles it correctly (`spike_input=self.spike_all`, L265–269), the prose does not. Say "hybrid; 96.5 % of MACs are spike-driven".
- **"SOPs do not grow with T"**: true by construction for the stream design (there is no T). It is also true for the draft's *chunk* mode — but see A6: chunk mode is a different, weaker model, so the claim is vacuous there.

---

## 5. Novelty and expected attacks

Individually, every component exists: repeat-encoding spiking EEGNet/ShallowConvNet ports (2022–2025, typically within-subject BCI-IV-2a, SpikingJelly, T = 4–16, same 4.6/0.9 pJ arithmetic — the 2025 Neural Networks "lightweight SNN for MI" is in this family), fixed-encoder Kasabov-lineage work (NeuCube, TBR/BSA/delta encoding of EEG at the sample rate), per-neuron learnable τ (Perez-Nieves et al. Nat. Commun. 2021; Yin et al. Nat. MI 2021; PLIF Fang et al. ICCV 2021; GLIF), time-encoding of band-limited signals (Lazar–Tóth 2004), EA (He & Wu TBME 2020) and AdaBN (Li et al. 2016). ISAM-MTL did cross-subject MI with spiking features but on BCI-IV-2a only; spiking transformers for EEG are within-subject and energy-motivated. Nobody has (i) trained the analog front-end + sample-rate LIF encoder end-to-end for EEG, (ii) evaluated an SNN-MI model cross-subject at this scale with paired statistics against a full ANN zoo, or (iii) asked whether subject alignment interacts with the threshold nonlinearity. That combination is publishable at ICASSP; it is not a new mechanism.

What makes it stand out: the 87-subject paired evaluation; the spike-phase-locking / rate-vs-amplitude figures that make the "learned 1-bit encoder" tangible; H1 with the firing-rate-shift mechanism (even a clean null with CIs is informative); honest hybrid energy accounting that includes the encoder.

What a strong SNN reviewer will attack, in order: (1) "a binarised-activation CNN with three analog layers; τ ≈ 2 means no dynamics" — pre-empt with learned-τ histograms and the reset/no-reset ablation; (2) theoretical FP32 energy, no memory, no hardware; (3) no BCI-IV-2a → no anchor to any published SNN-MI number, "you beat your own S-EEGNet"; (4) transductive EA/AdaBN; (5) one training seed; (6) HetPLIF ≠ novel, encoder ≠ novel unless the lineage is cited; (7) "streaming" while using trial-level z-scoring, session-level EA and non-causal 'same' padding — call it "sample-synchronous with a ≤0.5 s look-ahead", not "online".

Also reframe H2. The D version keeps the full analog conv1 map at 2-bit rate resolution; the S version imposes a 1-bit-per-sample bottleneck. Expect **D ≥ S in accuracy and S ≫ D in energy**; state H2 as "S within 1 pp of D at ~4× fewer SOPs", not "S ≥ D".

---

## 6. Draft code — defects found

`neurons.py`
- L107–114: HetPLIF (`learnable="channel"`) uses `torch.full((n_channels,), init_w)` — **all channels start at the same τ**. Heterogeneity must emerge from training at lr 1e-4, which it will not → HetPLIF ≡ PLIF → H3 null by construction. Needs per-channel log-uniform init with per-layer ranges (§1.5).
- L150: `self.decay(x_seq[t])` recomputed every step (hoist).
- L148–158: no per-layer reset mode, no LI/no-threshold mode, no bias; state discarded at the end (fine for trial-wise).
- L160–163: rate bookkeeping accumulates in *every* eval-mode forward, including validation passes during training; `pop_spike_stats()` resets only *after* reading, so the first held-out subject's stats include all validation-epoch spikes. Reset before each subject pass.
- The surrogate/dynamics themselves match SpikingJelly (`H = V + (X − (V − V_reset))/τ`, ATan backward) — correct.

`models.py`
- L123–126 + L219–222 + L256–273: **chunk mode is not the stream design.** Each 250-sample window is zero-padded per window for 64-tap conv1/conv2 (25 % of positions boundary-corrupted), conv4's 61-tap dilated receptive field on 63-sample pooled windows corrupts nearly every output, LIF steps only T = 4 times, `feature_dim` is 128 per window instead of 504, and logits are averaged across windows (killing the absolute time position that `RESULTS_TRUTH §4.1` says matters). Delete it; do not let it appear as "streaming" anywhere.
- L212–213: FC `max_norm(0.25)` on binary sparse features → logit ceiling (§2.5).
- L232, L237: rescaling dropout on spikes (§2.4).
- L262–272: op table is arithmetically right (reproduces 992.9 M MACs); missing neuron-update and BN-bias rows; needs a `steps=1, Lt=L` path for the stream model and an LI-readout FC row (MACs).
- L143–163 `energy_report(rates={})` in `describe()`: an empty dict is falsy, so it falls back to `self.firing_rates()` and only returns the "dense upper bound" if no eval pass has happened yet — mislabels measured numbers as an upper bound after any eval.
- Global `hard_reset`/`tau` for all layers (needs per-layer).
- `_infer` with all-zero input is fine for shape inference only.

---

## (A) Fatal — fix before any run

1. **HetPLIF init is not heterogeneous** (`neurons.py` L114) → H3 dead on arrival.
2. **τ range [1.5, 16] at LIF1 (250 Hz)** → ~3/8 encoder channels born silent; a zero-mean oscillation never crosses threshold for τ ≥ 8. Per-layer ranges (§1.5).
3. **LIF3 (and LIF2) start 2–3.5 σ below threshold** → chance-level for 100+ epochs at lr 1e-4 → killed by patience 25. BN β/γ init (or data-dependent calibration) + rate regulariser.
4. **Readout bottleneck + FC max_norm(0.25) logit ceiling** on binary features → LI membrane readout (or relax max_norm).
5. **Rescaling dropout on spikes** → systematic test-time firing-rate shift through BN4/BN5. Mask-only dropout.
6. **"Chunk" mode is a different, boundary-corrupted model** — remove; implement the real stream path (convs on the full `[B,C,H,L]` tensor exactly as the ANN, only the three neuron loops sequential, op table with steps = 1).
7. **D version memory/time**: 15–20 GB and ~4× ANN time at batch 128 — set `micro_batch_size: 64` or the main-table jobs OOM/overrun.

## (B) Important improvements, ranked by impact/effort

1. CUDA-graph or TorchScript the neuron loop (§3.2); hoist `sigmoid(w)`. High impact, ~half a day.
2. Per-layer reset: hard@LIF1, soft@LIF2/3; add `none` for LIF1 as the parallel fallback/ablation. Low effort.
3. Fix rate bookkeeping (reset before each subject pass; `describe()` upper bound). Low effort, protects the energy table.
4. Iterative (2–3 pass) AdaBN for the SNN; document the electrode-scale mechanism for H1.
5. Keep 'same' padding as the main S model (isolates the spiking effect from the causal-padding shift), declare the look-ahead; causal only if time permits.
6. Spike-phase-locking and rate-vs-amplitude figures from the trained encoder (cheap, makes the encoder claim real).
7. ON/OFF (±) encoder channels — only if the pilot shows S > 2 pp below D.
8. Cut the lr pilot to {seegnex_stream@hplif, seegnex@hplif} × {1e-4, 1e-3} × zscore only (20 cells, ~8 GPU-h) — EA does not interact with lr; the eegnex reference at 1e-4 already exists.
9. Turn `curves: false` for SNN entries (each eval pass re-runs the loops).

## (C) Parameter recommendations

- **τ init (log-uniform, steps)**: LIF1 [1.5, 4]; LIF2 [1.5, 8]; LIF3 [1.5, 8] (or [2, 16] if LI readout). Plain LIF/PLIF: τ = 2 at all layers is acceptable. Weight decay must stay 0 (the bench already has it) so `w` is not pulled back to τ = 2.
- **Threshold**: θ = 1.0 everywhere, v_reset = 0, with BN β = +0.5 and γ = 2 on the BNs feeding LIF2/LIF3 (β = +0.5, γ = 1 on BN1), or a one-batch calibration targeting r ≈ 0.10–0.20. Add \(\lambda\sum_l(\bar r_l - 0.1)^2\), λ = 1e-3.
- **Surrogate**: ATan α = 2 (α = 1 if a layer's pre-activations stay far from threshold). `detach_reset=True`.
- **Reset**: hard@LIF1, soft@LIF2/3.
- **Dropout**: p = 0.5 mask-only on spikes (or p = 0.25 with rescale).
- **lr**: pilot {1e-4, 1e-3} as planned but *after* the init fixes; prior expectation 1e-3 (3e-4 as a tie-breaker if both look poor). Adam, no clipping needed.
- **T (D version)**: 4; T = 8 only as an ablation. `micro_batch_size` 64.
- **Readout**: LI membrane at layer 3 (S); mean-over-T rate (D). FC max_norm 0.25 with analog readout, ≥ 1.0 (or none) if spiking.
- **Energy**: Horowitz 2014 numbers as planned; report SOPs as primary; footnote BN-bias and neuron-update ops; claim ≥10×.

## (D) Drop given six days

- Chunk mode (entirely). Causal padding as default (make it an ablation, or drop). Anytime curves. `spike_all`, `T=8`, BCI-IV-2a.
- 3C cohort for anything but {eegnex, seegnex_stream@hplif, seegnex@hplif} × {zscore, EA}.
- LIF/PLIF/HetPLIF ablation on 2C + SHU only; S-EEGNet baseline on 2C only, one config.
- The "first-k-trial calibration" curve — keep only because it is eval-only; do it last.
- Budget reality: with D at ~4× rather than 2.5×, P3 is ~265 GPU-h ≈ 2.8 days wall on four uncontended 4090Ds. Fine if the cluster cooperates; the list above is what to shed first if it doesn't.

## (E) Scores

- **Technical soundness: 6/10** — the statistical framework and energy arithmetic are sound, but the neuron design has one bug that nullifies a hypothesis (HetPLIF init), a physics error about long-τ LIF on zero-mean input, and three initialisation/readout issues that would make the fixed recipe fail silently; all fixable in a day → 8/10 afterwards.
- **Novelty: 5/10** — every mechanism (temporal-native LIF encoding, PLIF/heterogeneous τ, EA/AdaBN, SOP accounting) already exists; the contribution is the end-to-end learned encoder on EEGNeX plus an unprecedentedly rigorous cross-subject evaluation and the SNN-alignment interaction question, which is enough for ICASSP if H1 or the energy/accuracy trade-off comes out clean.
