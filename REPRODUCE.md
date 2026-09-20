---
title: "Reproducing the cross-subject benchmark, the ATCNet x S4ERP fusion, and the spiking round"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-08-27"
updated: "2026-09-18"
status: "active"
---

# Reproducing the cross-subject benchmark

Every number in the benchmark comes from two commands per dataset: one training sweep and
one summarize pass. This file lists them, what they should produce, and how long they take.
The design rationale is in [`docs/PLAN_cross_subject_benchmark_and_fusion.md`](docs/PLAN_cross_subject_benchmark_and_fusion.md).

## 1. What is being reproduced

**18 model entries x 2 datasets x 5 folds x 2 subject-partition seeds = 360 cells.**

| group | entries | what it is |
|:---|:---|:---|
| baselines | `eegnet_official`, `eegnex`, `eeg_deformer`, `atcnet` | published models, **original authors' code only** |
| classic | `deepconvnet`, `fbcnet` | pre-deep-learning-era reference points |
| ours (S4/DINO) | `s4erp`, `dualcd_s4_flatten`, `dualcd_transformer`, `dualcd_s4_timepatch` | diagonal-SSM family |
| fusion A | `s4erp_dw@f64d2`, `s4erp_dw@f16d8`, `s4erp_dw@f16d2p` | S4ERP + ATCNet's depthwise spatial front-end |
| fusion B | `atcnet_s4` | ATCNet + one shared causal SSM instead of 5 TCNs |
| ablations | `atcnet@tcn8`, `atcnet@nwin1`, `atcnet@noattn` | which ATCNet component carries its advantage |

All 18 run in **one sweep per dataset** on identical `(split_seed, fold, subject)` triples.
That is not a convenience: paired per-subject tests are the only way to detect differences
of a few percentage points against a 10pp between-subject spread, and pairing requires the
identical splits.

## 2. Environment

```bash
conda env create -f environment.yml          # or: use the existing `mi_torch_cu118`
conda activate mi_torch_cu118
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.7.1+cu118 True
```

CPU-only works for the smoke test (`--device cpu`) but not for full runs — one `s4erp`
step takes about 10 s on CPU.

## 3. Data

No EEG is shipped with this repository. Obtain both datasets from their sources
(see [`LICENSE`](LICENSE) § DATA), preprocess, then point the local config at the result:

```bash
cp code/configs/paths.example.yaml code/configs/paths.local.yaml   # then edit the paths
# WBCIC-SHU 2C: the `eog_ecg_clean` npz tree + processed_manifest.csv is produced by
# code/preprocessing/{pipeline,eog_ecg_clean}.py from the provider's BIDS derivatives (see code/preprocessing/README.md);
# on this cluster it already exists at /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/.
python scripts/preprocess_shu.py         # SHU 2022 -> processed/npz_clean/processed_manifest.csv
python scripts/preprocess_wbci_3c.py     # WBCIC-SHU 3C -> outputs/processed/wbci_shu_3c_mat_clean/ (~7 min)
```

Expected manifests (verify before training — a wrong manifest is the single most expensive
mistake here):

| dataset | ok sessions | subjects | shape | trials |
|:---|---:|---:|:---|---:|
| WBCIC-SHU 2C | 148 / 153 | 51 | `[n, 58, 1000]` @ 250 Hz | 29,600 |
| SHU 2022 | 125 / 125 | 25 | `[n, 32, 1000]` @ 250 Hz | 11,988 |

```bash
python3 -c "
import pandas as pd, sys, yaml
p = yaml.safe_load(open('code/configs/paths.local.yaml'))['manifests']
for key in ('processed_manifest', 'shu_processed_manifest'):
    d = pd.read_csv(p[key]); ok = d[d.status == 'ok']
    print(f\"{key:26s} {len(ok):4d} ok sessions  {ok.subject_id.nunique():3d} subjects  \"
          f\"{ok.n_trials.sum():6d} trials  {sorted(ok.n_channels.unique())}ch\")
"
```

## 4. Smoke test first (about 3 hours, 1 GPU)

Never skip this. A bug found here costs 3 hours; the same bug found after the full sweep
costs 5 days.

```bash
sbatch -J bench_smoke scripts/slurm/bench_gpu.sbatch \
  code/configs/experiments/wbci_2c_cross_subject_bench.yaml \
  --folds-subset 0 \
  --models eegnet_official,atcnet,s4erp,dualcd_transformer,dualcd_s4_timepatch,s4erp_dw@f64d2,s4erp_dw@f16d2p,atcnet_s4,atcnet_s4@noattn,atcnet@tcn8 \
  --max-epochs 5 \
  --out outputs/experiments/wbci_shu/wbci_2c_bench_smoke \
  --ckpt-dir checkpoints/wbci_shu/wbci_2c_bench_smoke
```

Acceptance checks — all must pass before the full sweep:

```bash
python3 - <<'PY'
import json, glob, numpy as np
cells = sorted(glob.glob("outputs/experiments/wbci_shu/wbci_2c_bench_smoke/cells/*/result.json"))
assert cells, "no cells produced"
for c in cells:
    d = json.load(open(c)); desc = d["model_desc"]; f = d["fold"]
    # 1. no zero parameter counts (deepconvnet/fbcnet used to report 0)
    assert desc.get("n_params", 0) > 0, f"{d['model']}: n_params={desc.get('n_params')}"
    # 2. strict zero-shot: the test subject is in neither train nor val
    tr, va, te = set(f["train_subjects"]), set(f["val_subjects"]), set(f["test_subjects"])
    assert not (tr & te) and not (va & te), f"{d['model']}: subject leak"
    # 3. 6:2:2 subject ratio
    assert len(va) == 10, f"{d['model']}: n_val_subjects={len(va)} (expected 10)"
    # 4. Cohen's Kappa present
    assert "cohen_kappa" in d["rows"][0], "kappa missing"
    # 5. per-trial predictions written and aligned
    p = np.load(c.replace("result.json", "predictions.npz"))
    n = p["y_true"].size
    assert p["probs_best"].shape == (n, d["data_dims"]["n_classes"])
    assert set(np.unique(p["session"])) <= {"ses-01", "ses-02", "ses-03"}
    # 6. atcnet_s4's SSM timescales rescaled for Tc=17, NOT the 6.8-546 default
    if d["model"] in ("atcnet_s4", "atcnet_s4@noattn"):
        assert desc["tau_max_tokens"] <= desc["Tc"] * 1.5, desc
        assert desc["readout_idx"] == [12, 13, 14, 15, 16], desc
        if d["model"] == "atcnet_s4@noattn":
            assert desc.get("attention") in (None, "null", "None") or not desc.get("attention")
print(f"all {len(cells)} smoke cells pass")
PY
```

Then read the measured per-cell cost out of the run and compare against the plan's estimate
(§7 of the plan). If it differs by more than ~50%, re-budget before launching the sweep:

```bash
python3 -c "
import json, glob, collections
a = collections.defaultdict(list)
for f in glob.glob('outputs/experiments/wbci_shu/wbci_2c_bench_smoke/cells/*/result.json'):
    d = json.load(open(f)); ti = d['train_info']
    a[d['model']].append(ti['train_seconds'] / max(ti['epochs_run'], 1))
for m, v in sorted(a.items(), key=lambda kv: -sum(kv[1])):
    print(f'{m:22s} {sum(v)/len(v):7.1f} s/epoch')
"
```

## 5. Full sweep (about 3.2 days, 4 GPUs)

```bash
bash scripts/slurm/submit_bench.sh wbci_2c      # 180 cells, ~227 GPU-h
bash scripts/slurm/submit_bench.sh shu          # 180 cells, ~83 GPU-h
```

Those totals are measured, not estimated: per-epoch cost comes from the smoke run and the
epoch count from a completed run on the same recipe. See the plan's §7 for the breakdown.

Each dataset is split into 4 groups (one GPU each, the QOS limit) and each group is chained
across 4 jobs with `--dependency=afterany`. Every link runs the same command; the resume
logic skips finished cells and continues an interrupted one from `last.pt`, so a group can
exceed the 48 h wall clock. Re-running the exact same command is always safe.

```bash
# monitor (the 2026-08/09 bench logs were deleted on 09-13; the pattern below is what a re-run would produce)
squeue -u $USER -o '%.10i %.14j %.2t %.11M %.11L %R'
tail -f logs/slurm/b2c_g4_1-*.out

# progress (a re-run writes here; the completed 2026-09 sweep lives read-only in results_snapshot_2026-09-13/)
find outputs/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1/cells -name result.json | wc -l   # -> 180

# if gpu2node is full and a job sits PENDING (Resources)
scontrol update jobid=<id> partition=gpu3node
```

One partition at a time: this cluster pins a partition in the association, so a list
(`partition=gpu2node,gpu3node`) is rejected with "Multiple partition job request not
supported". Only jobs that are still PENDING can be moved.

Anything that failed is in `<out>/failures__<tag>.json` (written by every job, `n_cells_failed`
tells you whether anything actually failed), not buried in the Slurm logs.

## 6. Tables, statistics, figures

```bash
# the completed sweep is read-only under results_snapshot_2026-09-13/ (a re-run would be under outputs/)
python scripts/summarize_cross_subject.py \
  --run results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1 \
  --run results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1 \
  --out-dir outputs/analysis/bench_tables
```

Per run: `tables/{per_cell,per_model,per_subject,coverage,paired_tests,variant_selection}.csv`,
`tables/friedman_nemenyi.json`, `figures/cd_diagram.png`, `REPORT_TABLE.md`. Across runs:
`tables/main_table.csv`.

Three things to read carefully:

* **`coverage.csv`** — any model with fewer subjects than the rest must not be compared
  against the complete rows. (A 7-of-11-fold row once made a model look competitive; see
  `docs/RESULTS_TRUTH.md` §8.)
* **`paired_tests.csv`** — the pre-registered comparisons, Holm-corrected, with Cohen's
  `d_z` and a 95% t-interval on the paired Δ. A significant p-value with `|d_z| < 0.2`
  is a small effect however small the p. Wilcoxon uses `method="approx"` on both
  datasets so n=25 and n=51 are the same test.
* **`variant_selection.csv`** — the `s4erp_dw` representative is chosen on **validation**
  macro-F1. If validation and test rankings disagree, that is the mechanism working, not a
  problem.

## 7. Determinism

* `cross_subject.split_seeds` decides the subject partition. `train.seeds` decides
  initialization and batch order. They are different knobs; the benchmark sweeps the first
  and fixes the second at 0, so reported spread is the spread of the partition.
* Each epoch reseeds with `epoch_seed_base + epoch`, so a run interrupted and resumed sees
  the same data order as an uninterrupted one.
* Bitwise reproducibility is **not** claimed: cuDNN kernel selection is nondeterministic and
  `torch.use_deterministic_algorithms` is not enabled (it would slow the sweep down without
  changing any conclusion at the resolution reported here).
* `cell_signature` (v2) fingerprints dataset, model, params, subject lists, and the full
  recipe **including `max_epochs` and `early_stopping_patience`**. Re-running with any of
  them changed into the same output directory fails loudly instead of silently mixing
  recipes. Use a fresh `--out` for a new setting.

## 8. Tests

```bash
pytest tests/ -q
pytest tests/models -q            # the fusion models, alias resolution, param counts
```

## 9. Spiking round (ICASSP 2027, 2026-09-18 →)

Design and decisions: `docs/PLAN_spiking_eegnex_icassp2027.md` (§0.5 pilot results), `docs/DECISIONS.md`
D-001–D-026. The benchmark tables above are the paired ANN baselines; they now live read-only in
`results_snapshot_2026-09-13/experiments/…`.

```bash
# configs (24 files = {shu,wbci2c,wbci3c} x {zs,ea,eamu,eaz} x {lr1e3,lr1e4}; edit the grid in the generator, never the YAMLs)
python scripts/make_snn_configs.py

# submit one config as N single-GPU groups, each a 48 h relay chain (DRY_RUN=1 to preview)
bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_wbci2c_zs_lr1e4.yaml  c2z \
     "seegnex@s_hplif,seegnex@hplif" "eegnex,eegnet_official,seegnet@plif"
bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_wbci2c_eaz_lr1e4.yaml c2e \
     "seegnex@s_hplif,seegnex@hplif" "eegnex,eegnet_official,seegnet@plif"
PARTITION=gpu3node bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_wbci3c_zs_lr1e4.yaml c3z \
     "eegnex,eegnet_official,seegnex@hplif,seegnex@s_hplif,seegnet@plif"
# … SHU adds seegnex@s_lif,seegnex@s_plif (H3). Job ids -> logs/slurm/snn_<tag>_job_ids.txt

# analysis: pre-registered paired tests, H1 difference-of-differences, energy, Friedman; +bench pairs the 18 ANNs
python scripts/analyze_snn_round.py --dataset wbci_shu \
    --run zs=outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1 \
    --run eaz=outputs/experiments/wbci_shu/snn_wbci2c_eaz_lr1e4_v1 --aligned-arm eaz \
    --bench results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1 \
    --out outputs/analysis/snn_round/wbci_shu
# `--aligned-arm eaz` maps the eaz run onto the canonical role `ea` that the pre-registered family and the
# H1 interaction are written against; without it the script now aborts (it used to write a report without H1).
pytest tests/models/test_spiking.py -q      # 13 tests: neuron numerics, EA whitening, all variants fwd/bwd, energy
pytest tests -q                             # whole suite after the 2026-09-18 prune: 106 passed, 1 skipped
```

Expected cell counts: 2C 50 per arm (5 entries × 5 folds × 2 splits); 3C 55 per arm (5 × 11 LOSO);
SHU 70 per arm (7 × 5 × 2). Only the lr 1e-4 runs enter the final SHU table (`snn_shu_zs_lr1e4_v1` split-41
pilot cells included — renamed to `__split41__` so job 71034 skips them, D-030). The lr 1e-3 pilot arms
(`snn_shu_{zs,ea,eamu,eaz}_lr1e3_v1`) are a separate pilot/ablation table: the afe / spk / noinit ablations exist
**only at lr 1e-3** and are paired against the lr 1e-3 S-EEGNeX-S. Pilot analyses:
`outputs/analysis/snn_round/shu_pilot_lr1e4/` and `shu_pilot_lr1e3/` (regenerated 2026-09-18 with the arm mapping).
