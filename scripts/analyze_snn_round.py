#!/usr/bin/env python
"""Analysis of the spiking-EEGNeX round: ANN vs SNN x alignment, paired per subject.

    python scripts/analyze_snn_round.py --dataset shu \
        --run zs=outputs/experiments/shu/snn_shu_zs_lr1e3_v1 \
        --run ea=outputs/experiments/shu/snn_shu_ea_lr1e3_v1 \
        --bench results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1 \
        --out outputs/analysis/snn_round/shu

Every ``--run`` is ``<arm>=<run_dir>`` where ``arm`` names the normalization arm. The
pre-registered comparisons and the H1 interaction are written against two canonical roles,
``zs`` (unaligned baseline) and ``ea`` (aligned arm). Which of your labels plays which role
is set with ``--baseline-arm`` / ``--aligned-arm`` (defaults ``zs`` / ``ea``); the main run's
aligned arm is ``eaz`` (EA + per-channel z-score, D-021), so pass ``--aligned-arm eaz`` or
simply label the run ``ea=``. Any other label is kept as an extra arm (per-entry table only).
The script aborts if the pre-registered family matches nothing, instead of silently writing a
report without H0/H1. Entries are addressed as ``<model>+<role>`` and, for the AdaBN-scored
copy, ``<model>+<role>+adabn``. The optional ``--bench`` run contributes the 18 ANN baselines
as ``<model>+bench`` (same subject folds, so everything is paired).

Outputs (all under ``--out``):
  per_model.csv          mean +/- std over subjects for every entry (+ params, epochs, GPU-h)
  paired_tests.csv       pre-registered family (Holm within family) + exploratory extras
  interaction_H1.csv     per-subject difference-of-differences tests: (SNN_ea-SNN_zs)-(ANN_ea-ANN_zs)
  energy.csv             firing rates + energy estimate per spiking entry (median over subjects)
  overfit.csv            train/val/test at the best epoch (from curves) per entry
  subject_matrix.csv     subjects x entries accuracy matrix (the unit of every test)
  REPORT.md              everything above, readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.summaries.cross_subject_bench import (  # noqa: E402
    _holm, friedman_nemenyi, load_run, paired_tests,
)

METRICS = ["accuracy", "macro_f1", "auc", "balanced_accuracy", "cohen_kappa"]
MAIN = "seegnex@s_hplif"
ANN = "eegnex"

#: Pre-registered family (Holm within this family; everything else is exploratory).
#: (baseline_entry, candidate_entry, question)
PREREGISTERED: List[Tuple[str, str, str]] = [
    (f"{ANN}+zs", f"{MAIN}+zs", "H0 non-inferiority: stream S-EEGNeX vs EEGNeX, no alignment"),
    (f"{ANN}+ea", f"{MAIN}+ea", "H0 non-inferiority: stream S-EEGNeX vs EEGNeX, with EA"),
    (f"{ANN}+zs", f"{ANN}+ea", "EA effect on the ANN"),
    (f"{MAIN}+zs", f"{MAIN}+ea", "EA effect on the SNN"),
    (f"{ANN}+ea", f"{ANN}+ea+adabn", "AdaBN effect on the ANN (after EA)"),
    (f"{MAIN}+ea", f"{MAIN}+ea+adabn", "AdaBN effect on the SNN (after EA)"),
    ("seegnex@hplif+ea", f"{MAIN}+ea", "H2: temporal-native vs direct (T=4) encoding"),
    ("seegnet@plif+ea", "seegnex@hplif+ea", "S-EEGNet vs S-EEGNeX (same direct encoding)"),
]
EXPLORATORY: List[Tuple[str, str, str]] = [
    ("seegnex@s_lif+ea", "seegnex@s_plif+ea", "H3: learnable tau (PLIF) vs fixed (LIF)"),
    ("seegnex@s_plif+ea", f"{MAIN}+ea", "H3: heterogeneous tau vs one tau per layer"),
    ("seegnex@s_lif+zs", "seegnex@s_plif+zs", "H3 (no EA): PLIF vs LIF"),
    ("seegnex@s_plif+zs", f"{MAIN}+zs", "H3 (no EA): heterogeneous vs per-layer tau"),
    (f"{ANN}+zs", f"{ANN}+zs+adabn", "AdaBN alone on the ANN"),
    (f"{MAIN}+zs", f"{MAIN}+zs+adabn", "AdaBN alone on the SNN"),
    # no-alignment arm: direct-encoding SNN, S-EEGNet, and the single-factor ablations
    (f"{ANN}+zs", "seegnex@hplif+zs", "direct (T=4) S-EEGNeX vs EEGNeX, no alignment"),
    ("seegnex@hplif+zs", f"{MAIN}+zs", "H2 (no EA): temporal-native vs direct (T=4) encoding"),
    ("eegnet_official+zs", "seegnet@plif+zs", "S-EEGNet vs EEGNet, no alignment"),
    (f"{MAIN}+zs", "seegnex@s_hplif_afe+zs", "ablation: analog front-end (spikes after spatial filter)"),
    (f"{MAIN}+zs", "seegnex@s_hplif_spk+zs", "ablation: binary spike readout instead of LI"),
    (f"{MAIN}+zs", "seegnex@s_hplif_noinit+zs", "ablation: default BN init before neurons"),
    (f"{MAIN}+zs", "seegnex@s_hplif_causal+zs", "ablation: causal padding"),
    ("eegnex+bench", f"{ANN}+zs", "sanity: re-run EEGNeX vs benchmark EEGNeX (same recipe/splits)"),
    ("atcnet+bench", f"{MAIN}+ea+adabn", "vs strongest benchmark ANN (ATCNet)"),
    ("eegnex+bench", f"{MAIN}+ea+adabn", "vs benchmark EEGNeX"),
    ("atcnet@noattn+bench", f"{MAIN}+ea+adabn", "vs benchmark #1 on 2C (ATCNet w/o attention)"),
]


# --------------------------------------------------------------------------- #
def load_arm(run_dir: Path, arm: str) -> Tuple[pd.DataFrame, List[Dict]]:
    """Rows with entry names ``<model>+<arm>`` (+ AdaBN copies) and per-subject spike stats."""
    df = load_run(run_dir)
    df["entry"] = df["model"].astype(str) + "+" + arm
    frames = [df]
    if "adabn_accuracy" in df.columns and df["adabn_accuracy"].notna().any():
        ad = df.copy()
        for m in METRICS:
            col = f"adabn_{m}"
            if col in ad.columns:
                ad[m] = pd.to_numeric(ad[col], errors="coerce")
        ad = ad[ad["accuracy"].notna()]
        ad["entry"] = ad["model"].astype(str) + "+" + arm + "+adabn"
        frames.append(ad)
    # spike / energy statistics live in the per-subject detail of result.json
    stats: List[Dict] = []
    for p in sorted((run_dir / "cells").glob("*/result.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        model = d.get("model")
        split = d.get("split_seed")
        for det in d.get("per_subject") or d.get("per_subject_detail") or []:
            for key, suffix in (("spike_stats", ""), ("spike_stats_adabn", "+adabn")):
                ss = det.get(key)
                if not ss:
                    continue
                rec = {"entry": f"{model}+{arm}{suffix}", "model": model, "split_seed": split,
                       "test_subject": det.get("test_subject")}
                rec.update({f"rate_{k}": v for k, v in (ss.get("firing_rates") or {}).items()})
                rec.update(ss.get("energy") or {})
                stats.append(rec)
    return pd.concat(frames, ignore_index=True), stats


def load_bench(run_dir: Path) -> pd.DataFrame:
    df = load_run(run_dir)
    df["entry"] = df["model"].astype(str) + "+bench"
    return df


def entry_matrix(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """subjects x entries, averaged over split seeds (and train seeds) first."""
    return df.pivot_table(index="test_subject", columns="entry", values=metric,
                          aggfunc="mean").sort_index()


def per_entry_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entry, g in df.groupby("entry"):
        subj = g.groupby("test_subject")[[m for m in METRICS if m in g.columns]].mean()
        rec: Dict[str, object] = {"entry": entry, "model": g["model"].iloc[0],
                                  "n_subjects": int(len(subj)),
                                  "n_cells": int(g["cell"].nunique()),
                                  "split_seeds": sorted(set(g["split_seed"].dropna().astype(int)))}
        for m in METRICS:
            if m in subj.columns:
                rec[m] = float(subj[m].mean())
                rec[f"{m}_std"] = float(subj[m].std(ddof=1)) if len(subj) > 1 else float("nan")
        rec["val_macro_f1"] = float(pd.to_numeric(g["best_score"], errors="coerce").mean())
        rec["n_params"] = int(pd.to_numeric(g["n_params"], errors="coerce").max() or 0)
        rec["best_epoch_median"] = float(pd.to_numeric(g["best_epoch"], errors="coerce").median())
        rec["gpu_hours"] = float(pd.to_numeric(g["train_seconds"], errors="coerce")
                                 .groupby(g["cell"]).first().sum() / 3600.0)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("entry").sort_values("accuracy", ascending=False)


def interaction_tests(mat: pd.DataFrame, pairs: Sequence[Tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """Difference-of-differences per subject: (cand_B - cand_A) - (base_B - base_A).

    ``pairs`` = (base_A, base_B, cand_A, cand_B, question). Wilcoxon signed-rank on the
    per-subject interaction contrast; t-based 95% CI; Holm within this family.
    """
    import inspect

    from scipy.stats import t as student_t
    from scipy.stats import wilcoxon

    kw: Dict[str, object] = dict(zero_method="wilcox", alternative="two-sided")
    if "method" in inspect.signature(wilcoxon).parameters:
        kw["method"] = "approx"
    out = []
    for a0, a1, c0, c1, q in pairs:
        cols = [a0, a1, c0, c1]
        if any(c not in mat.columns for c in cols):
            continue
        sub = mat[cols].dropna()
        if len(sub) < 3:
            continue
        base_gain = sub[a1] - sub[a0]
        cand_gain = sub[c1] - sub[c0]
        inter = (cand_gain - base_gain).to_numpy(dtype=float)
        n = len(inter)
        sd = float(inter.std(ddof=1))
        se = sd / np.sqrt(n) if sd > 0 else float("nan")
        tcrit = float(student_t.ppf(0.975, n - 1))
        try:
            _, p = wilcoxon(inter, **kw)
        except ValueError:
            p = 1.0
        out.append({"question": q, "n": n,
                    "gain_baseline": float(base_gain.mean()), "gain_candidate": float(cand_gain.mean()),
                    "interaction": float(inter.mean()), "interaction_median": float(np.median(inter)),
                    "ci95_lo": float(inter.mean() - tcrit * se), "ci95_hi": float(inter.mean() + tcrit * se),
                    "d_z": float(inter.mean() / sd) if sd > 0 else float("nan"),
                    "wins": int((inter > 0).sum()), "losses": int((inter < 0).sum()),
                    "p_value": float(p)})
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["p_holm"] = _holm(df["p_value"].tolist(), n_family=len(pairs))
    return df


def energy_table(stats: List[Dict]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame()
    df = pd.DataFrame(stats)
    num = [c for c in df.columns if c.startswith("rate_") or c.endswith("_uJ")
           or c in ("snn_macs", "snn_acs", "ann_macs", "energy_ratio_ann_over_snn")]
    agg = df.groupby("entry")[num].median()
    agg["n_subject_rows"] = df.groupby("entry").size()
    return agg.sort_values("energy_ratio_ann_over_snn", ascending=False)


def overfit_table(run_dirs: Dict[str, Path]) -> pd.DataFrame:
    """train / val / test accuracy at the best epoch from the per-epoch curves (if logged)."""
    rows = []
    for arm, run_dir in run_dirs.items():
        for p in sorted((run_dir / "cells").glob("*/result.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            hist = (d.get("train") or {}).get("history") or d.get("history") or []
            best_epoch = (d.get("train") or {}).get("best_epoch", d.get("best_epoch"))
            if not hist or best_epoch is None:
                continue
            h = None
            for rec in hist:
                if int(rec.get("epoch", -1)) == int(best_epoch):
                    h = rec
                    break
            if h is None:
                continue
            rows.append({"entry": f"{d.get('model')}+{arm}",
                         "train_acc": h.get("train_accuracy", h.get("train_acc")),
                         "val_acc": h.get("val_accuracy", h.get("accuracy")),
                         "test_acc": h.get("test_accuracy", h.get("test_acc"))})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="ignore")
    g = df.groupby("entry")[["train_acc", "val_acc", "test_acc"]].mean()
    g["gap_train_test"] = g["train_acc"] - g["test_acc"]
    return g.sort_values("gap_train_test")


def _fmt(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "_(no data yet)_\n"
    return df.to_markdown(floatfmt=".4f") + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", action="append", required=True, help="<arm>=<run_dir>")
    ap.add_argument("--bench", default=None, help="benchmark run dir (18 ANN baselines)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="accuracy", choices=METRICS)
    ap.add_argument("--baseline-arm", default="zs",
                    help="which --run label is the unaligned baseline role 'zs' (default zs)")
    ap.add_argument("--aligned-arm", default="ea",
                    help="which --run label is the aligned role 'ea' (main run: eaz; default ea)")
    ap.add_argument("--allow-empty-family", action="store_true",
                    help="do not abort when no pre-registered comparison can be formed")
    args = ap.parse_args()

    # User labels -> canonical roles used by PREREGISTERED / EXPLORATORY / interaction lists.
    role_of = {args.baseline_arm: "zs", args.aligned_arm: "ea"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, stats, run_dirs, arm_map = [], [], {}, {}
    for spec in args.run:
        label, _, rd = spec.partition("=")
        rd = Path(rd)
        if not (rd / "cells").exists():
            print(f"[skip] {rd}: no cells yet")
            continue
        arm = role_of.get(label, label)
        if arm in run_dirs:
            raise SystemExit(f"two --run entries map to the same role '{arm}' "
                             f"({run_dirs[arm]} and {rd}); check --baseline-arm/--aligned-arm")
        df, st = load_arm(rd, arm)
        frames.append(df)
        stats += st
        run_dirs[arm] = rd
        arm_map[arm] = f"{label}={rd}"
    if args.bench:
        frames.append(load_bench(Path(args.bench)))
    if not frames:
        raise SystemExit("nothing to analyse")
    df = pd.concat(frames, ignore_index=True)

    table = per_entry_table(df)
    mat = entry_matrix(df, args.metric)
    pre = paired_tests(mat, PREREGISTERED)
    if pre.empty and not args.allow_empty_family:
        raise SystemExit(
            "no pre-registered comparison could be formed: the runs are labelled "
            f"{sorted(run_dirs)} but the family needs roles 'zs' and/or 'ea' with entries "
            f"{ANN!r} and {MAIN!r}. Use --baseline-arm/--aligned-arm to map your labels "
            "(main run: --aligned-arm eaz), or --allow-empty-family for a table-only report.")
    if not pre.empty:
        pre.insert(0, "family", "preregistered")
    expl = paired_tests(mat, EXPLORATORY)
    if not expl.empty:
        expl.insert(0, "family", "exploratory")
        expl["p_holm"] = float("nan")
        expl["significant_holm_05"] = pd.NA
    tests = pd.concat([pre, expl], ignore_index=True) if not (pre.empty and expl.empty) else pd.DataFrame()

    inter = interaction_tests(mat, [
        (f"{ANN}+zs", f"{ANN}+ea", f"{MAIN}+zs", f"{MAIN}+ea",
         "H1: EA gain SNN(stream) minus EA gain ANN"),
        (f"{ANN}+ea", f"{ANN}+ea+adabn", f"{MAIN}+ea", f"{MAIN}+ea+adabn",
         "H1b: AdaBN gain SNN(stream) minus AdaBN gain ANN (after EA)"),
        (f"{ANN}+zs", f"{ANN}+ea", "seegnex@hplif+zs", "seegnex@hplif+ea",
         "H1 (direct SNN): EA gain SNN(direct) minus EA gain ANN"),
    ])
    energy = energy_table(stats)
    overfit = overfit_table(run_dirs)

    # omnibus over the spiking round + benchmark, complete cases only
    fried = {}
    try:
        cc = mat.dropna(axis=1, thresh=int(0.9 * len(mat))).dropna(axis=0)
        if cc.shape[1] >= 3 and len(cc) >= 5:
            fried = friedman_nemenyi(cc)
    except Exception as e:  # pragma: no cover
        fried = {"error": str(e)}

    table.to_csv(out / "per_model.csv")
    mat.to_csv(out / "subject_matrix.csv")
    if not tests.empty:
        tests.to_csv(out / "paired_tests.csv", index=False)
    if not inter.empty:
        inter.to_csv(out / "interaction_H1.csv", index=False)
    if not energy.empty:
        energy.to_csv(out / "energy.csv")
    if not overfit.empty:
        overfit.to_csv(out / "overfit.csv")

    show = ["model", "n_subjects", "n_cells", "accuracy", "accuracy_std", "macro_f1", "auc",
            "val_macro_f1", "n_params", "best_epoch_median", "gpu_hours"]
    arm_note = "; ".join(f"role `{k}` <- `{v}`" for k, v in sorted(arm_map.items())) or "none"
    lines = [f"# Spiking round — {args.dataset}\n",
             f"metric for tests: **{args.metric}**; unit = subject (mean over split seeds first)\n",
             f"arms: {arm_note}" + (f"; bench <- `{args.bench}`" if args.bench else "") + "\n",
             "> NOTE: the 'preregistered' family below is the 8-comparison list coded before D-012 "
             "(PLAN §0.3 fixes a 5-test family: H0-NI, H1, H2-NI, H3 Page's L, B1). Non-inferiority "
             "bounds, exact/permutation Wilcoxon, HL + BCa CI and Page's L are still pending "
             "(STATUS §3.3); treat p-values here as approximate.\n",
             "## Per-entry table (mean over subjects)\n", _fmt(table[[c for c in show if c in table.columns]]),
             "## Pre-registered paired tests (Holm within family) + exploratory\n",
             _fmt(tests[[c for c in ["family", "baseline", "candidate", "question", "n", "mean_baseline",
                                     "mean_candidate", "delta", "delta_ci95_lo", "delta_ci95_hi", "d_z",
                                     "wins", "ties", "losses", "p_value", "p_holm", "significant_holm_05"]
                         if c in tests.columns]] if not tests.empty else tests),
             "## Interaction (H1): does alignment help the SNN more than the ANN?\n", _fmt(inter),
             "## Energy / firing rates (median over subjects)\n", _fmt(energy),
             "## Overfitting at best epoch (train / val / test)\n", _fmt(overfit),
             "## Friedman + Nemenyi (complete cases)\n",
             "```\n" + json.dumps({k: (v if not hasattr(v, "to_dict") else v.to_dict())
                                   for k, v in fried.items()}, indent=1, default=str)[:4000] + "\n```\n"]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print((out / "REPORT.md").read_text(encoding="utf-8")[:6000])


if __name__ == "__main__":
    main()
