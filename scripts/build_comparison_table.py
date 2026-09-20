#!/usr/bin/env python
"""Build the paper comparison table (models x cohorts) from script-generated CSVs only.

Sources (never hand-typed numbers):
  * 2C  : results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1/tables/per_model.csv
  * SHU : results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1/tables/per_model.csv
  * 3C  : results_snapshot_2026-09-13/experiments/wbci_shu/paper_baseline_3c_821_v1/tables/per_cell.csv
          (mean / SD with ddof=1 over the 11 LOSO folds, same SD convention as the bench tables;
           the per_model.csv of that run used ddof=0)
  * SNN : any ``per_model.csv`` / ``energy.csv`` written by ``scripts/analyze_snn_round.py``
          (default: the SHU pilot at lr 1e-4, split 41 only). Pass ``--snn-2c / --snn-shu / --snn-3c``
          once the main-run analyses exist; the table then refreshes without touching the script.

Outputs (``--out``):
  comparison_long.csv   one row per (model, cohort) cell with provenance (source file, row key)
  comparison_table.md   wide table for the report (Model | Year | 2C | SHU | 3C | Params | uJ/trial)
  tab_main_rows.tex     the tabular body for paper/main.tex (same column layout as tab:main)

CPU-only, safe on the login node.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "results_snapshot_2026-09-13" / "experiments"

BENCH_2C = SNAP / "wbci_shu" / "wbci_2c_cross_subject_bench_v1" / "tables" / "per_model.csv"
BENCH_SHU = SNAP / "shu" / "shu_cross_subject_bench_v1" / "tables" / "per_model.csv"
CELLS_3C = SNAP / "wbci_shu" / "paper_baseline_3c_821_v1" / "tables" / "per_cell.csv"
PILOT_SHU = ROOT / "outputs" / "analysis" / "snn_round" / "shu_pilot_lr1e4"

# Roster: display name, year, bib key, group, and which cohorts it was run on.
# ``group``: published = named architecture with official code; inhouse = this project's own entries
# (report only, never in the paper table); snn = this paper's spiking models.
ROSTER: List[Dict] = [
    dict(key="deepconvnet", name="DeepConvNet", year=2017, cite="schirrmeister2017deep", group="published"),
    dict(key="eegnet_official", name="EEGNet", year=2018, cite="lawhern2018eegnet", group="published"),
    dict(key="fbcnet", name="FBCNet", year=2021, cite="mane2021fbcnet", group="published"),
    dict(key="atcnet", name="ATCNet", year=2023, cite="altaheri2023atcnet", group="published"),
    dict(key="eegnex", name="EEGNeX", year=2024, cite="chen2024eegnex", group="published"),
    dict(key="eeg_deformer", name="EEG-Deformer", year=2025, cite="ding2025deformer", group="published"),
    dict(key="s4erp", name="S4ERP", year=None, cite=None, group="inhouse"),
    dict(key="s4erp_dw@f64d2", name="S4ERP-DW (f64d2)", year=None, cite=None, group="inhouse"),
    dict(key="s4erp_dw@f16d8", name="S4ERP-DW (f16d8)", year=None, cite=None, group="inhouse"),
    dict(key="s4erp_dw@f16d2p", name="S4ERP-DW (f16d2p)", year=None, cite=None, group="inhouse"),
    dict(key="dualcd_transformer", name="DualCD-Transformer", year=None, cite=None, group="inhouse"),
    dict(key="dualcd_s4_flatten", name="DualCD-S4-Flatten", year=None, cite=None, group="inhouse"),
    dict(key="dualcd_s4_timepatch", name="DualCD-S4-TimePatch", year=None, cite=None, group="inhouse"),
    dict(key="atcnet_s4", name="ATCNet-S4", year=None, cite=None, group="inhouse"),
    dict(key="atcnet_s4@noattn", name="ATCNet-S4 (no attn)", year=None, cite=None, group="inhouse"),
    dict(key="atcnet@tcn8", name="ATCNet (tcn8)", year=None, cite=None, group="inhouse"),
    dict(key="atcnet@nwin1", name="ATCNet (nwin1)", year=None, cite=None, group="inhouse"),
    dict(key="atcnet@noattn", name="ATCNet (no attn)", year=None, cite=None, group="inhouse"),
    dict(key="seegnet@plif", name="S-EEGNet (PLIF)", year=None, cite=None, group="snn",
         tex=r"S-EEGNet (PLIF)"),
    dict(key="seegnex@hplif", name="S-EEGNeX-D (HetPLIF, T=4)", year=None, cite=None, group="snn",
         tex=r"S-EEGNeX-D (HetPLIF, $T{=}4$)"),
    dict(key="seegnex@s_hplif", name="S-EEGNeX-S (HetPLIF, temporal-native)", year=None, cite=None,
         group="snn", tex=r"\textbf{S-EEGNeX-S (HetPLIF, temporal-native)}"),
]

# Parameter counts at 58 channels / 2 classes (the 2C basis used for the whole Params column).
# ANN values are the ``n_params`` column of the 2C bench table; SNN values were counted with
# ``code.models.registry.build_model`` on CPU (2026-09-18) because no 58-ch SNN cell has finished yet.
PARAMS_58CH_SNN = {"seegnet@plif": 3028, "seegnex@hplif": 58850, "seegnex@s_hplif": 58850}


def _fmt_pm(mean: float, sd: float, digits: int = 1) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return ""
    return f"{100 * mean:.{digits}f} ± {100 * sd:.{digits}f}"


def _fmt_pm_tex(mean: float, sd: float, digits: int = 1) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return r"\todo{}"
    return f"{100 * mean:.{digits}f} $\\pm$ {100 * sd:.{digits}f}"


def _fmt_params(n: Optional[float]) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "--"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.1f}k"


def load_bench(path: Path, cohort: str) -> Dict[str, Dict]:
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[r["model"]] = dict(
            cohort=cohort, arm="zs", protocol="kfold_subject 5-fold, splits 41+42",
            n_subjects=int(r["n_subjects"]), n_cells=10, acc=float(r["accuracy"]),
            acc_sd=float(r["accuracy_std"]), macro_f1=float(r["macro_f1"]),
            bal_acc=float(r["balanced_accuracy"]), n_params=int(r["n_params"]),
            gpu_hours=float(r["gpu_hours"]), source=str(path.relative_to(ROOT)),
            source_key=f"model={r['model']}", flag="",
        )
    return out


def load_3c_cells(path: Path) -> Dict[str, Dict]:
    df = pd.read_csv(path)
    out = {}
    for model, g in df.groupby("model"):
        g = g.sort_values("fold")
        out[model] = dict(
            cohort="3C", arm="zs", protocol="LOSO 11-fold, paper recipe (Adam 1e-4, batch 128, patience 100)",
            n_subjects=int(g["test_subject"].nunique()), n_cells=int(len(g)),
            acc=float(g["accuracy"].mean()), acc_sd=float(g["accuracy"].std(ddof=1)),
            macro_f1=float(g["macro_f1"].mean()), bal_acc=float(g["balanced_accuracy"].mean()),
            n_params=int(g["n_params"].iloc[0]), gpu_hours=float(g["train_seconds"].sum() / 3600.0),
            source=str(path.relative_to(ROOT)), source_key=f"model={model} (11 rows, ddof=1)",
            flag="paper_recipe",
        )
    return out


def load_snn(per_model: Path, cohort: str, arm: str = "zs", energy: Optional[Path] = None) -> Dict[str, Dict]:
    """Rows written by analyze_snn_round.py: entry = '<model>+<arm>[+adabn]' or '<model>+bench'."""
    df = pd.read_csv(per_model)
    en = pd.read_csv(energy).set_index("entry") if energy and energy.exists() else None
    out = {}
    for _, r in df.iterrows():
        entry = str(r["entry"])
        if entry != f"{r['model']}+{arm}":
            continue
        seeds = str(r["split_seeds"])
        single_split = seeds.count(",") == 0
        rec = dict(
            cohort=cohort, arm=arm,
            protocol=f"kfold_subject 5-fold, splits {seeds}" if cohort != "3C" else f"LOSO, splits {seeds}",
            n_subjects=int(r["n_subjects"]), n_cells=int(r["n_cells"]), acc=float(r["accuracy"]),
            acc_sd=float(r["accuracy_std"]), macro_f1=float(r["macro_f1"]),
            bal_acc=float(r["balanced_accuracy"]), n_params=int(r["n_params"]),
            gpu_hours=float(r["gpu_hours"]), source=str(per_model.relative_to(ROOT)),
            source_key=f"entry={entry}", flag="pilot_single_split" if single_split else "",
            energy_uJ=float("nan"),
        )
        if en is not None and entry in en.index:
            row = en.loc[entry]
            rec["energy_uJ"] = float(row["snn_energy_uJ"])
            rec["energy_source"] = str(energy.relative_to(ROOT))
            # The matching ANN energy (dense MACs) is on the same row; keep it for the ANN parent.
            rec["ann_energy_uJ"] = float(row["ann_energy_uJ"])
        out[str(r["model"])] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snn-2c", type=Path, default=None, help="per_model.csv from analyze_snn_round (2C)")
    ap.add_argument("--snn-shu", type=Path, default=PILOT_SHU / "per_model.csv")
    ap.add_argument("--snn-3c", type=Path, default=None)
    ap.add_argument("--energy-shu", type=Path, default=PILOT_SHU / "energy.csv")
    ap.add_argument("--digits", type=int, default=1, help="decimals in the LaTeX table")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    out_rel = args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out

    cohorts = {
        "2C": load_bench(BENCH_2C, "2C"),
        "SHU": load_bench(BENCH_SHU, "SHU"),
        "3C": load_3c_cells(CELLS_3C),
    }
    snn = {"2C": {}, "SHU": {}, "3C": {}}
    if args.snn_2c and args.snn_2c.exists():
        snn["2C"] = load_snn(args.snn_2c, "2C")
    if args.snn_shu and args.snn_shu.exists():
        snn["SHU"] = load_snn(args.snn_shu, "SHU", energy=args.energy_shu)
    if args.snn_3c and args.snn_3c.exists():
        snn["3C"] = load_snn(args.snn_3c, "3C")

    # Energy per trial (uJ): SNN rows from energy.csv; the two ANN parents from the ann_energy_uJ of
    # their spiking counterpart (dense MACs x 4.6 pJ, identical layer shapes). Others: not computed.
    energy: Dict[str, float] = {}
    energy_note = ""
    for m, rec in snn["SHU"].items():
        if not math.isnan(rec.get("energy_uJ", float("nan"))):
            energy[m] = rec["energy_uJ"]
            energy_note = rec.get("energy_source", "")
    if "seegnex@s_hplif" in snn["SHU"] and "ann_energy_uJ" in snn["SHU"]["seegnex@s_hplif"]:
        energy["eegnex"] = snn["SHU"]["seegnex@s_hplif"]["ann_energy_uJ"]
    if "seegnet@plif" in snn["SHU"] and "ann_energy_uJ" in snn["SHU"]["seegnet@plif"]:
        energy["eegnet_official"] = snn["SHU"]["seegnet@plif"]["ann_energy_uJ"]

    long_rows = []
    wide_rows = []
    for m in ROSTER:
        key = m["key"]
        cells = {}
        for c in ("2C", "SHU", "3C"):
            rec = snn[c].get(key) if m["group"] == "snn" else cohorts[c].get(key)
            if rec is None and m["group"] != "snn":
                rec = snn[c].get(key)  # ANN parents re-run inside the SNN round (only if not in bench)
            cells[c] = rec
            if rec is not None:
                long_rows.append(dict(model_key=key, model=m["name"], year=m["year"], group=m["group"],
                                      **{k: v for k, v in rec.items() if k not in ("ann_energy_uJ",)}))
        n_params_58 = None
        if m["group"] == "snn":
            n_params_58 = PARAMS_58CH_SNN.get(key)
        elif cells["2C"] is not None:
            n_params_58 = cells["2C"]["n_params"]
        wide_rows.append(dict(model=m, cells=cells, n_params_58=n_params_58, energy=energy.get(key)))

    long_df = pd.DataFrame(long_rows)
    long_df["year"] = long_df["year"].astype("Int64")
    long_df.to_csv(args.out / "comparison_long.csv", index=False)

    # ---- Pilot-internal paired block: the only apples-to-apples SNN-vs-ANN comparison so far ----
    # (same split, same folds, same recipe; the bench rows above are on splits 41+42 and cannot be
    #  paired with single-split pilot cells).
    if args.snn_shu and args.snn_shu.exists():
        pm = pd.read_csv(args.snn_shu)
        pm = pm[pm["entry"].str.endswith("+zs")].copy()
        pm["acc_pm"] = [_fmt_pm(a, s, 2) for a, s in zip(pm["accuracy"], pm["accuracy_std"])]
        pm["bacc_pm"] = [_fmt_pm(a, s, 2) for a, s in zip(pm["balanced_accuracy"], pm["balanced_accuracy_std"])]
        lines = [f"| entry | n_subj | n_cells | splits | acc ± SD | balanced acc ± SD | macro-F1 | params | GPU-h |",
                 "|:---|---:|---:|:---|---:|---:|---:|---:|---:|"]
        for _, r in pm.sort_values("accuracy", ascending=False).iterrows():
            lines.append(f"| `{r['entry']}` | {int(r['n_subjects'])} | {int(r['n_cells'])} | {r['split_seeds']} | "
                         f"{r['acc_pm']} | {r['bacc_pm']} | {100 * r['macro_f1']:.2f} | {int(r['n_params'])} | "
                         f"{r['gpu_hours']:.2f} |")
        pt_path = args.snn_shu.parent / "paired_tests.csv"
        if pt_path.exists():
            pt = pd.read_csv(pt_path)
            lines += ["", "| family | baseline → candidate | Δacc (pp) | 95% CI | wins/ties/losses | p (Wilcoxon) |",
                      "|:---|:---|---:|:---|:---|---:|"]
            for _, r in pt.iterrows():
                lines.append(f"| {r['family']} | `{r['baseline']}` → `{r['candidate']}` | {100 * r['delta']:+.2f} | "
                             f"[{100 * r['delta_ci95_lo']:+.2f}, {100 * r['delta_ci95_hi']:+.2f}] | "
                             f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])} | {r['p_value']:.3f} |")
        src = f"{args.snn_shu.relative_to(ROOT) if args.snn_shu.is_relative_to(ROOT) else args.snn_shu}"
        lines.insert(0, f"Source: `{src}` and `paired_tests.csv` next to it (analyze_snn_round.py).\n")
        (args.out / "pilot_paired.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Markdown (report) ----
    def md_cell(rec):
        if rec is None:
            return "—"
        s = _fmt_pm(rec["acc"], rec["acc_sd"], 2)
        if rec["flag"] == "pilot_single_split":
            s += " †"
        elif rec["flag"] == "paper_recipe":
            s += " ‡"
        return s

    md = ["| Model | Year | 2C (51) acc ± SD | SHU (25) acc ± SD | 3C (11) acc ± SD | Params @58ch | µJ/trial (SHU, 32ch) |",
          "|:---|:---:|---:|---:|---:|---:|---:|"]
    for grp, title in (("published", "已发表 ANN 基线（官方代码，同协议）"),
                       ("snn", "本文 SNN（先导：SHU split 41；主实验在跑）"),
                       ("inhouse", "自研 / 消融条目（只进报告，不进论文表）")):
        md.append(f"| **{title}** | | | | | | |")
        for w in wide_rows:
            if w["model"]["group"] != grp:
                continue
            mm = w["model"]
            yr = str(mm["year"]) if mm["year"] else ("ours" if grp == "snn" else "—")
            e = w["energy"]
            e_s = f"{e:.0f}" if e is not None and not math.isnan(e) else "—"
            md.append(f"| {mm['name']} | {yr} | {md_cell(w['cells']['2C'])} | {md_cell(w['cells']['SHU'])} | "
                      f"{md_cell(w['cells']['3C'])} | {_fmt_params(w['n_params_58'])} | {e_s} |")
    md.append("")
    md.append("† SHU 先导：lr 1e-4、`zs` 臂、**只有 split 41（5 cell，每被试测 1 次）**；bench 行是 split 41+42（10 cell，每被试测 2 次取平均），"
              "两者不在同一划分集合上，**逐被试配对比较只能用先导自己重跑的 EEGNeX/EEGNet（见报告 §4.2）**。主实验补齐 split 42 后由本脚本重算。")
    md.append("‡ 3C：`paper_baseline_3c_821_v1`，LOSO 11 折，论文配方（patience 100，其余同 bench）；SD 为 ddof=1（该 run 自带 per_model.csv 用 ddof=0）。"
              "主实验 3C 组正在用 bench 配方重跑 EEGNeX / EEGNet + SNN，出齐后替换。")
    md.append("能耗：45 nm，MAC 4.6 pJ / AC 0.9 pJ，逐层 dense MACs × 实测发放率，SHU 32 通道、4 s trial；ANN 两行为同形状 dense MACs × 4.6 pJ；"
              "未做 MAC 核算的模型为 —。")
    (args.out / "comparison_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- LaTeX rows (paper/main.tex tab:main) ----
    def tex_cell(rec, digits):
        if rec is None:
            return r"\todo{}"
        s = _fmt_pm_tex(rec["acc"], rec["acc_sd"], digits)
        if rec["flag"] == "pilot_single_split":
            s += r"$^\dagger$"
        elif rec["flag"] == "paper_recipe":
            s += r"$^\ddagger$"
        return s

    tex = [f"% generated by scripts/build_comparison_table.py -> {out_rel}/tab_main_rows.tex",
           "% columns: Model & 2C (51) & SHU (25) & 3C (11) & Params & uJ/trial"]
    for grp in ("published", "snn"):
        if grp == "snn":
            tex.append(r"    \midrule")
        for w in wide_rows:
            mm = w["model"]
            if mm["group"] != grp:
                continue
            if grp == "published":
                label = f"{mm['name']}~\\cite{{{mm['cite']}}} ({mm['year']})"
            else:
                label = mm.get("tex", mm["name"])
            e = w["energy"]
            e_s = f"{e:.0f}" if e is not None and not math.isnan(e) else "--"
            c3 = w["cells"]["3C"]
            c3_s = tex_cell(c3, args.digits) if (c3 is not None or grp == "snn") else "--"
            tex.append(f"    {label} & {tex_cell(w['cells']['2C'], args.digits)} & "
                       f"{tex_cell(w['cells']['SHU'], args.digits)} & {c3_s} & "
                       f"{_fmt_params(w['n_params_58'])} & {e_s} \\\\")
    (args.out / "tab_main_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print("\n".join(md))
    print("\nLaTeX rows ->", args.out / "tab_main_rows.tex")
    if energy_note:
        print("energy source:", energy_note)


if __name__ == "__main__":
    main()
