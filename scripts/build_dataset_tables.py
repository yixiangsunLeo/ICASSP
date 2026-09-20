#!/usr/bin/env python
"""Build the per-cohort comparison tables (2C / SHU / 3C) of the manuscript from script-generated CSVs.

This complements ``scripts/build_comparison_table.py`` (one models x cohorts table for paper/main.tex):
here every cohort gets its own table in the layout of the team's ICASSP 2026 paper (``old_reference.tex``:
model name with a venue line underneath, ``\\multirow`` value cells, ``\\midrule`` between models), with
several metrics per cohort. Numbers are never typed by hand. A compact single ``table*`` with all cohorts
(single-line rows) is written as well and can be spliced instead with ``--layout compact``.

Sources
  2C  : results_snapshot_2026-09-13/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1/tables/per_model.csv
  SHU : results_snapshot_2026-09-13/experiments/shu/shu_cross_subject_bench_v1/tables/per_model.csv
  3C  : results_snapshot_2026-09-13/experiments/wbci_shu/paper_baseline_3c_821_v1/tables/per_cell.csv
        (LOSO, paper recipe = patience 100; mean and SD with ddof=1 over the 11 folds = 11 subjects)
  SNN / recipe-matched re-runs / external baselines run into the same run directories
      : ``per_model.csv`` written by scripts/analyze_snn_round.py, rows ``<model>+zs`` (``--snn-2c/-shu/-3c``).
        For 3C these rows REPLACE the paper-recipe rows of the same model (recipe-matched run preferred);
        for 2C/SHU the bench rows stay canonical and analyze rows only add models that are not in the bench.
  other CSVs (``--extra COHORT=path``; bench per_model / analyze per_model / per_cell format, auto-detected)
      : may only ADD models; a key that already exists is an error unless ``--allow-override``.

Row semantics
  pending  = being run in the parallel session (docs/PROTOCOL_external_baselines.md) -> ``\\tbd`` until data exist
  optional = printed only when data exist (CBraMod, the proposed substitute for EEGMoE)
  a ticked baseline without data on a cohort (FBCNet on 3C) -> ``--`` with the legend ``--: not run``
  pretrained = initialised from a public checkpoint -> superscript P on the name + legend

Outputs (``--out``)
  tab_2c.tex, tab_shu.tex, tab_3c.tex   complete ``table`` environments (2026-paper layout)
  tab_compact_all.tex                   one ``table*`` with all cohorts, single-line rows
  dataset_tables_long.csv               one row per (cohort, model) with every metric and its provenance
  dataset_tables.md                     Markdown version (reports)

``--splice <manuscript.tex>`` replaces everything between the two marker lines (each must occur exactly once)
  % >>> AUTO-GENERATED COHORT TABLES BEGIN ...
  % <<< AUTO-GENERATED COHORT TABLES END
The manuscript preamble must define ``\\tbd`` and load ``booktabs`` + ``multirow``.
``--final`` refuses to write a block that still contains a ``\\tbd``.

CPU-only; safe on the login node.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_comparison_table import BENCH_2C, BENCH_SHU, CELLS_3C  # noqa: E402  (single source of paths)

COHORTS = ("2C", "SHU", "3C")
EXPECTED_N_SUBJECTS = {"2C": 51, "SHU": 25, "3C": 11}  # AGENTS.md §1; a table with another count is refused
MARK_BEGIN = "% >>> AUTO-GENERATED COHORT TABLES BEGIN"
MARK_END = "% <<< AUTO-GENERATED COHORT TABLES END"

# Headers are kept short so that the tables fit the 86 mm column at 9 pt (measured with tectonic 2026-09-18:
# 2C/3C 198 pt, SHU 224 pt of 244.7 pt with 4 columns). Units and long names go to the caption legend.
METRICS: Dict[str, Dict] = {
    "acc": dict(col="accuracy", sd="accuracy_std", header="Acc.", md="Acc (%)", pct=True,
                legend="Acc.: accuracy (\\%)"),
    "bal_acc": dict(col="balanced_accuracy", sd="balanced_accuracy_std", header="BAcc.", md="BAcc (%)", pct=True,
                    legend="BAcc.: balanced accuracy (\\%)"),
    "macro_f1": dict(col="macro_f1", sd="macro_f1_std", header="F1", md="Macro-F1 (%)", pct=True,
                     legend="F1: macro-F1 (\\%)"),
    "auc": dict(col="auc", sd="auc_std", header="AUC", md="AUC", pct=False, legend="AUC in $[0,1]$"),
    "kappa": dict(col="cohen_kappa", sd="cohen_kappa_std", header=r"$\kappa$", md="kappa", pct=False,
                  legend="$\\kappa$: Cohen's kappa"),
}
# 2C and 3C are class-balanced (balanced accuracy == accuracy) and on SHU accuracy tracks balanced accuracy within
# a few hundredths of a pp for every benchmark entry, so each cohort shows ONE primary metric (+- SD) plus F1 and
# AUC. Cohen's kappa exists for 2C/SHU but not in the historical 3C per-cell table. Override: --metrics[-<cohort>].
DEFAULT_METRICS = {"2C": "acc+sd,macro_f1,auc", "SHU": "bal_acc+sd,macro_f1,auc", "3C": "acc+sd,macro_f1,auc"}

# Roster (2026-09-18, senior's tick marks + the five baselines of the parallel session). Venues / years verified
# 2026-09-18 (entries in paper/refs.bib). ``include`` rows are off by default and switched on with --include KEY
# (EEGNeX / DeepConvNet were not ticked; the reviewers' strongest objection is the missing EEGNeX row).
ROSTER: List[Dict] = [
    dict(key="eegnet_official", name="EEGNet", venue="J. Neural Eng. 2018", cite="lawhern2018eegnet", year=2018),
    dict(key="deepconvnet", name="DeepConvNet", venue="Hum. Brain Mapp. 2017", cite="schirrmeister2017deep",
         year=2017, include=False),
    dict(key="fbcnet", name="FBCNet", venue="arXiv 2021", cite="mane2021fbcnet", year=2021),
    dict(key="atcnet", name="ATCNet", venue="IEEE TII 2023", cite="altaheri2023atcnet", year=2023),
    dict(key="eegnex", name="EEGNeX", venue="Biomed. Signal Process. Control 2024", cite="chen2024eegnex",
         year=2024, include=False),
    dict(key="labram", name="LaBraM", venue="ICLR 2024", cite="jiang2024labram", year=2024, pending=True,
         pretrained=True),
    dict(key="eegpt", name="EEGPT", venue="NeurIPS 2024", cite="wang2024eegpt", year=2024, pending=True,
         pretrained=True),
    dict(key="eeg_deformer", name="EEG-Deformer", venue="IEEE JBHI 2025", cite="ding2025deformer", year=2025),
    dict(key="tcformer", name="TCFormer", venue="Sci. Rep. 2025", cite="altaheri2025tcformer", year=2025,
         pending=True),
    dict(key="mdgeeg", name="MDGEEG", venue="Neurocomputing 2025", cite="song2025mdgeeg", year=2025, pending=True),
    dict(key="cbramod", name="CBraMod", venue="ICLR 2025", cite="wang2025cbramod", year=2025, pending=True,
         optional=True, pretrained=True),
    dict(key="eegmoe", name="EEGMoE", venue="IEEE TNNLS 2026", cite="gao2026eegmoe", year=2026, pending=True,
         pretrained=True),
]
# The spiking candidates. Pre-registration (AGENTS.md §4.3, PLAN §0.3): main model = seegnex@s_hplif; the D
# version is the documented fallback (STATUS §4). The choice must NOT be made on test metrics, so this script
# prints candidate NAMES only as comments; their numbers stay in dataset_tables_long.csv / dataset_tables.md.
CANDIDATE_NAMES = {
    "seegnex@s_hplif": "S-EEGNeX-S (HetPLIF, temporal-native)",
    "seegnex@hplif": "S-EEGNeX-D (HetPLIF, $T{=}4$)",
    "seegnet@plif": "S-EEGNet (PLIF)",
}

COHORT_TEXT = {
    "2C": dict(label="tab:comparison_2c", title="WBCIC-SHU 2C",
               desc="{n} subjects, 58 channels, left vs.\\ right hand; chance 50\\%",
               protocol="Five subject folds with the validation subjects carved from the training subjects "
                        "(train/validation/test 6:2:2), two fold seeds (every subject tested twice); identical "
                        "optimiser and early-stopping rule for all rows. Mean over subjects; $\\pm$: SD across "
                        "subjects of the per-subject score averaged over the two seeds."),
    "SHU": dict(label="tab:comparison_shu", title="SHU 2022",
                desc="{n} subjects, 32 channels, left vs.\\ right hand",
                protocol="Protocol as in Table~\\ref{{tab:comparison_2c}}. Balanced accuracy is the primary metric "
                         "because of session-level class imbalance{gap_note}."),
    "3C": dict(label="tab:comparison_3c", title="WBCIC-SHU 3C",
               desc="{n} subjects, 58 channels; left hand / right hand / feet; chance 33.3\\%",
               protocol="Leave-one-subject-out on the dataset's official pre-processed derivatives, with the same "
                        "per-trial z-scoring, optimiser and early-stopping rule as Table~\\ref{{tab:comparison_2c}}. "
                        "AUC: macro one-vs-rest. Mean over the held-out subjects; $\\pm$: SD across subjects."),
}
BLOCK_PREFIX = "Unaligned block (per-trial z-scoring only; no EA/AdaBN): "
FLAG_NOTE = {
    "pilot_single_split": "pilot run, split seed 41 only (each subject tested once)",
    "paper_recipe": "historical LOSO run with patience 100 instead of 25 (not recipe-matched)",
}
FLAG_MARK = {"pilot_single_split": r"$^\dagger$", "paper_recipe": r"$^\ddagger$"}
PRETRAINED_MARK = r"$^{\mathrm{P}}$"
PRETRAINED_NOTE = "P: initialised from the public pre-trained checkpoint and fine-tuned with the common recipe"


# ----------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------
def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _vals_from_row(r: pd.Series) -> Dict[str, Tuple[float, float]]:
    return {m: (_f(r[s["col"]]) if s["col"] in r.index else float("nan"),
                _f(r[s["sd"]]) if s["sd"] in r.index else float("nan")) for m, s in METRICS.items()}


def load_bench(path: Path, cohort: str, canonical: bool) -> Dict[str, Dict]:
    """bench-format per_model.csv (one row per model, SD across subjects already computed)."""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[str(r["model"])] = dict(
            cohort=cohort, values=_vals_from_row(r), n_subjects=int(r["n_subjects"]),
            n_cells=10 if canonical else float("nan"),
            protocol=("kfold_subject 5-fold 6:2:2, splits 41+42, bench recipe" if canonical
                      else f"bench-format per_model.csv ({_rel(path)}); protocol as documented with that run"),
            source=_rel(path), source_key=f"model={r['model']}", flag="")
    return out


def load_per_cell(path: Path, cohort: str, canonical_3c: bool) -> Dict[str, Dict]:
    """per_cell.csv (one row per cell). Per-subject score = mean over that subject's cells (splits), then mean
    and SD (ddof=1) across subjects -- the same convention as the bench per_model tables."""
    df = pd.read_csv(path)
    out = {}
    for model, g in df.groupby("model"):
        per_subj = g.groupby("test_subject").mean(numeric_only=True)
        vals = {}
        for m, s in METRICS.items():
            if s["col"] in per_subj.columns:
                vals[m] = (float(per_subj[s["col"]].mean()), float(per_subj[s["col"]].std(ddof=1)))
            else:
                vals[m] = (float("nan"), float("nan"))
        n_splits = int(g["split_seed"].nunique()) if "split_seed" in g.columns else 1
        out[str(model)] = dict(
            cohort=cohort, values=vals, n_subjects=int(per_subj.shape[0]), n_cells=int(len(g)),
            protocol=("LOSO 11-fold, paper recipe (Adam 1e-4, batch 128, patience 100)" if canonical_3c
                      else f"per_cell.csv ({_rel(path)}): {len(g)} cells, {n_splits} split(s), per-subject mean"),
            source=_rel(path), source_key=f"model={model} ({len(g)} rows, per-subject mean, ddof=1)",
            flag="paper_recipe" if canonical_3c else "")
    return out


def load_analyze(path: Path, cohort: str, arm: str = "zs") -> Dict[str, Dict]:
    """per_model.csv of analyze_snn_round.py: entry = '<model>+<arm>[+adabn]' or '<model>+bench'."""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        entry = str(r["entry"])
        if entry != f"{r['model']}+{arm}":
            continue
        seeds = str(r["split_seeds"])
        n_seeds = len([s for s in re.findall(r"\d+", seeds)])
        single = cohort != "3C" and n_seeds == 1  # LOSO has no split seeds; a lone seed there is not a pilot
        out[str(r["model"])] = dict(
            cohort=cohort, values=_vals_from_row(r), n_subjects=int(r["n_subjects"]), n_cells=int(r["n_cells"]),
            protocol=(f"LOSO {int(r['n_cells'])}-fold" if cohort == "3C" else f"kfold_subject 5-fold, splits {seeds}")
                     + f", arm {arm}, bench recipe",
            source=_rel(path), source_key=f"entry={entry}", flag="pilot_single_split" if single else "")
    return out


def load_any(path: Path, cohort: str) -> Dict[str, Dict]:
    cols = set(pd.read_csv(path, nrows=0).columns)
    if "entry" in cols:
        return load_analyze(path, cohort)
    if {"test_subject", "fold"} <= cols:
        return load_per_cell(path, cohort, canonical_3c=False)
    if {"model", "accuracy", "n_subjects"} <= cols:
        return load_bench(path, cohort, canonical=False)
    raise SystemExit(f"unrecognised CSV format: {path}")


# ----------------------------------------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------------------------------------
def parse_metrics(spec: str) -> List[Tuple[str, bool]]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        with_sd = tok.endswith("+sd")
        name = tok[:-3] if with_sd else tok
        if name not in METRICS:
            raise SystemExit(f"unknown metric '{name}'; choose from {sorted(METRICS)}")
        out.append((name, with_sd))
    return out


def fmt_value(m: str, mean: float, sd: float, with_sd: bool, tex: bool, pct_digits: int) -> Optional[str]:
    s = METRICS[m]
    if mean is None or math.isnan(mean):
        return None
    scale, d = (100.0, pct_digits) if s["pct"] else (1.0, 3)
    txt = f"{scale * mean:.{d}f}"
    if with_sd and sd is not None and not math.isnan(sd):
        txt += (" $\\pm$ " if tex else " ± ") + f"{scale * sd:.{d}f}"
    return txt


def rounded(m: str, mean: float, pct_digits: int) -> str:
    s = METRICS[m]
    scale, d = (100.0, pct_digits) if s["pct"] else (1.0, 3)
    return f"{scale * mean:.{d}f}"


def best_per_metric(rows: List[Dict], metrics: List[Tuple[str, bool]], pct_digits: int) -> Dict[str, str]:
    """Best printed value per metric (ties at printed precision are all bold)."""
    best = {}
    for m, _ in metrics:
        vals = [r["rec"]["values"][m][0] for r in rows
                if r["rec"] is not None and not math.isnan(r["rec"]["values"][m][0])]
        if vals:
            best[m] = rounded(m, max(vals), pct_digits)
    return best


def table_rows(merged: Dict[str, Dict], ours: Optional[str], ours_name: str, aliases: Dict[str, str],
               include: List[str]) -> List[Dict]:
    rows = []
    roster = sorted(ROSTER, key=lambda d: (d["year"] is None, d["year"] or 0, ROSTER.index(d)))
    for m in roster:
        if not m.get("include", True) and m["key"] not in include:
            continue
        rec = merged.get(aliases.get(m["key"], m["key"]))
        if rec is None and m.get("optional"):
            continue
        rows.append(dict(model=m, rec=rec, ours=False))
    ours_row = dict(model=dict(key=ours, name=ours_name, venue="(Ours)", cite=None, year=None, pending=True),
                    rec=None, ours=True)
    if ours:
        ours_row["rec"] = merged.get(aliases.get(ours, ours))
    rows.append(ours_row)
    return rows


def check_coverage(cohort: str, rows: List[Dict]) -> None:
    exp = EXPECTED_N_SUBJECTS[cohort]
    bad = [(r["model"]["name"], r["rec"]["n_subjects"]) for r in rows
           if r["rec"] is not None and r["rec"]["n_subjects"] != exp]
    if bad:
        raise SystemExit(f"{cohort}: rows with n_subjects != {exp} (partial run?): {bad}")


def legend_parts(rows: List[Dict]) -> Tuple[bool, List[str]]:
    flags = {r["rec"]["flag"] for r in rows if r["rec"] is not None}
    mixed = len(flags) > 1
    parts = []
    if mixed:
        parts += [f"{FLAG_MARK[f]}: {FLAG_NOTE[f]}" for f in sorted(flags) if f]
    elif len(flags) == 1 and next(iter(flags)):
        parts.append(f"all rows: {FLAG_NOTE[next(iter(flags))]}")
    if any(r["model"].get("pretrained") for r in rows):
        parts.append(PRETRAINED_NOTE)
    if any(r["rec"] is None and r["model"].get("pending") for r in rows):
        parts.append("\\tbd{}: not available yet")
    if any(r["rec"] is None and not r["model"].get("pending") for r in rows):
        parts.append("--: not run")
    return mixed, parts


def shu_gap_note(rows: List[Dict], pct_digits: int) -> str:
    gaps = [abs(r["rec"]["values"]["acc"][0] - r["rec"]["values"]["bal_acc"][0]) for r in rows
            if r["rec"] is not None and not math.isnan(r["rec"]["values"]["acc"][0])
            and not math.isnan(r["rec"]["values"]["bal_acc"][0])]
    if not gaps:
        return ""
    return f" (it differs from accuracy by at most {100 * max(gaps):.2f}~pp for the rows shown)"


def caption_for(cohort: str, rows: List[Dict], metrics: List[Tuple[str, bool]], pct_digits: int) -> Tuple[str, bool]:
    txt = COHORT_TEXT[cohort]
    n_subj = next((r["rec"]["n_subjects"] for r in rows if r["rec"] is not None), None)
    n_txt = str(n_subj) if n_subj is not None else "\\tbd{}"
    protocol = txt["protocol"].format(gap_note=shu_gap_note(rows, pct_digits) if cohort == "SHU" else "")
    mixed, parts = legend_parts(rows)
    cap = (f"{BLOCK_PREFIX}cross-subject classification on {txt['title']} ({txt['desc'].format(n=n_txt)}). "
           f"{protocol} " + "; ".join(METRICS[m]["legend"] for m, _ in metrics) + ". Best value per column in bold")
    cap += ("; " + "; ".join(parts) if parts else "") + "."
    return cap, mixed


def name_cell(r: Dict) -> Tuple[str, str]:
    m = r["model"]
    if r["ours"]:
        return f"\\textbf{{{m['name']}}}", "\\textbf{(Ours)}"
    first = f"{m['name']}~\\cite{{{m['cite']}}}" if m.get("cite") else m["name"]
    if m.get("pretrained"):
        first += PRETRAINED_MARK
    # \small = 9 pt = the kit's minimum font size (the 2026 paper used \footnotesize = 8 pt, not allowed)
    return f"{{{first}}}", f"\\small {m['venue']}"


def value_cells(r: Dict, metrics, best, mixed, pct_digits, multirow=True) -> List[str]:
    rec = r["rec"]
    wrap = (lambda s: f"\\multirow{{2}}{{*}}{{{s}}}") if multirow else (lambda s: s)
    out = []
    for m, with_sd in metrics:
        if rec is None:
            out.append(wrap("\\tbd" if r["model"].get("pending") else "--"))
            continue
        mean, sd = rec["values"][m]
        s = fmt_value(m, mean, sd, with_sd, True, pct_digits)
        if s is None:
            out.append(wrap("--"))
            continue
        if m in best and rounded(m, mean, pct_digits) == best[m]:
            s = f"\\textbf{{{s}}}"
        if mixed and rec["flag"]:
            s += FLAG_MARK[rec["flag"]]
        out.append(wrap(s))
    return out


def render_table(cohort: str, rows: List[Dict], metrics, candidates: List[str], pct_digits: int, note: str) -> str:
    check_coverage(cohort, rows)
    best = best_per_metric(rows, metrics, pct_digits)
    caption, mixed = caption_for(cohort, rows, metrics, pct_digits)
    ncol = len(metrics)
    lines = [f"% {note}",
             f"% cohort {cohort}: values from the CSVs listed in dataset_tables_long.csv (same directory).",
             "\\begin{table}[htb]",  # the kit's own float example uses [htb]
             f"  \\caption{{{caption}}}",
             f"  \\label{{{COHORT_TEXT[cohort]['label']}}}",
             "  \\centering",
             "  \\setlength{\\tabcolsep}{3pt}",
             f"  \\begin{{tabular}}{{l{'c' * ncol}}}",
             "    \\toprule",
             "    Model & " + " & ".join(METRICS[m]["header"] for m, _ in metrics) + " \\\\",
             "    \\midrule"]
    for i, r in enumerate(rows):
        first, second = name_cell(r)
        lines.append(f"    {first} & " + " & ".join(value_cells(r, metrics, best, mixed, pct_digits)) + " \\\\")
        lines.append(f"    {second} " + "&" * ncol + " \\\\")
        lines.append("    \\bottomrule" if i == len(rows) - 1 else "    \\midrule")
    lines.append("  \\end{tabular}")
    if candidates:
        lines.append("  % (Ours) row: pre-registered main model = seegnex@s_hplif (S version); the D version is the")
        lines.append("  % documented fallback. Do not pick by test metrics. Candidates (regenerate with --ours <key>):")
        for c in candidates:
            lines.append(f"  %   {c}: {CANDIDATE_NAMES.get(c, c)}")
        lines.append("  % Their current numbers: dataset_tables.md / dataset_tables_long.csv next to this fragment.")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def render_compact(rows_by_cohort: Dict[str, List[Dict]], metrics_by_cohort, pct_digits: int, note: str) -> str:
    """One table* with a column group per cohort and single-line rows (fallback if the page budget is tight)."""
    cohorts = [c for c in COHORTS if c in rows_by_cohort]
    header1, header2, spans, col_spec, pos = ["Model"], [""], [], "l", 2
    for c in cohorts:
        mets = metrics_by_cohort[c]
        n = next((r["rec"]["n_subjects"] for r in rows_by_cohort[c] if r["rec"] is not None), "?")
        header1.append(f"\\multicolumn{{{len(mets)}}}{{c}}{{{COHORT_TEXT[c]['title']} ({n})}}")
        header2 += [METRICS[m]["header"] for m, _ in mets]
        spans.append(f"\\cmidrule(lr){{{pos}-{pos + len(mets) - 1}}}")
        pos += len(mets)
        col_spec += "c" * len(mets)
    # one model list across cohorts (union, roster order); a model absent from a cohort shows -- / \tbd
    names = []
    for c in cohorts:
        for r in rows_by_cohort[c]:
            k = r["model"].get("key")
            if all(k != x["model"].get("key") for x in names):
                names.append(r)
    legends, all_parts = [], []
    for c in cohorts:
        _, parts = legend_parts(rows_by_cohort[c])
        for p in parts:
            if p not in all_parts:
                all_parts.append(p)
        legends += [METRICS[m]["legend"] for m, _ in metrics_by_cohort[c] if METRICS[m]["legend"] not in legends]
    caption = (f"{BLOCK_PREFIX}cross-subject classification on the three cohorts (number of subjects in "
               "parentheses). 2C/SHU: five subject folds (6:2:2), two fold seeds; 3C: leave-one-subject-out on the "
               "official derivatives. Identical per-trial z-scoring, optimiser and early-stopping rule for all rows. "
               "Mean over subjects; $\\pm$: SD across subjects. " + "; ".join(legends) +
               ". Best value per column in bold" + ("; " + "; ".join(all_parts) if all_parts else "") + ".")
    lines = [f"% {note}", "% compact alternative: one table* (single-line rows); splice with --layout compact",
             "\\begin{table*}[t]", f"  \\caption{{{caption}}}", "  \\label{tab:comparison_all}", "  \\centering",
             "  \\setlength{\\tabcolsep}{4pt}", f"  \\begin{{tabular}}{{{col_spec}}}", "    \\toprule",
             "    " + " & ".join(header1) + " \\\\", "    " + " ".join(spans),
             "    " + " & ".join(header2) + " \\\\", "    \\midrule"]
    for r0 in names:
        m = r0["model"]
        if r0["ours"]:
            label = f"\\textbf{{{m['name']} (Ours)}}"
        else:
            label = f"{m['name']}~\\cite{{{m['cite']}}}" if m.get("cite") else m["name"]
            label += PRETRAINED_MARK if m.get("pretrained") else ""
            label += f" ({m['year']})" if m.get("year") else ""
        cells = []
        for c in cohorts:
            rows = rows_by_cohort[c]
            best = best_per_metric(rows, metrics_by_cohort[c], pct_digits)
            mixed = legend_parts(rows)[0]
            r = next((x for x in rows if x["model"].get("key") == m.get("key")), None)
            if r is None:
                cells += ["--"] * len(metrics_by_cohort[c])
            else:
                cells += value_cells(r, metrics_by_cohort[c], best, mixed, pct_digits, multirow=False)
        if r0["ours"]:
            lines.append("    \\midrule")
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def render_markdown(cohort: str, rows: List[Dict], metrics, pct_digits: int) -> str:
    best = best_per_metric(rows, metrics, pct_digits)
    hdr = "| Model | Venue | " + " | ".join(METRICS[m]["md"] + (" ± SD" if sd else "") for m, sd in metrics) + " |"
    out = [f"### {COHORT_TEXT[cohort]['title']}", "", hdr, "|:---|:---|" + "---:|" * len(metrics)]
    for r in rows:
        m = r["model"]
        cells = []
        for mm, with_sd in metrics:
            if r["rec"] is None:
                cells.append("TBD" if m.get("pending") else "— (not run)")
                continue
            mean, sd = r["rec"]["values"][mm]
            s = fmt_value(mm, mean, sd, with_sd, False, pct_digits)
            if s is None:
                cells.append("—")
                continue
            if mm in best and rounded(mm, mean, pct_digits) == best[mm]:
                s = f"**{s}**"
            if r["rec"]["flag"]:
                s += " †" if r["rec"]["flag"] == "pilot_single_split" else " ‡"
            cells.append(s)
        name = f"**{m['name']} (Ours)**" if r["ours"] else m["name"] + (" (P)" if m.get("pretrained") else "")
        out.append(f"| {name} | {'' if r['ours'] else m['venue']} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def splice(tex_path: Path, block: str) -> None:
    text = tex_path.read_text(encoding="utf-8")
    b = re.compile("^" + re.escape(MARK_BEGIN) + r"[^\n]*\n", re.M)
    e = re.compile("^" + re.escape(MARK_END) + r"[^\n]*$", re.M)
    bs, es = list(b.finditer(text)), list(e.finditer(text))
    if len(bs) != 1 or len(es) != 1 or es[0].start() < bs[0].end():
        raise SystemExit(f"{tex_path}: need exactly one BEGIN and one END marker line (found {len(bs)} / {len(es)})")
    tex_path.write_text(text[:bs[0].end()] + block + text[es[0].start():], encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bench-2c", type=Path, default=BENCH_2C)
    ap.add_argument("--bench-shu", type=Path, default=BENCH_SHU)
    ap.add_argument("--cells-3c", type=Path, default=CELLS_3C)
    ap.add_argument("--snn-2c", type=Path, default=None, help="per_model.csv from analyze_snn_round.py (2C)")
    ap.add_argument("--snn-shu", type=Path, default=None, help="per_model.csv from analyze_snn_round.py (SHU)")
    ap.add_argument("--snn-3c", type=Path, default=None, help="per_model.csv from analyze_snn_round.py (3C)")
    ap.add_argument("--extra", action="append", default=[], metavar="COHORT=CSV",
                    help="additional CSV for a cohort (adds models only; repeatable)")
    ap.add_argument("--allow-override", action="store_true", help="let --extra replace existing model rows")
    ap.add_argument("--alias", action="append", default=[], metavar="ROSTERKEY=CSVKEY",
                    help="map a roster key (e.g. mdgeeg) to the model key used in the CSVs; repeatable")
    ap.add_argument("--include", action="append", default=[], help="switch on an off-by-default roster row, e.g. eegnex")
    ap.add_argument("--ours", default=None, help="model key of the proposed row, e.g. seegnex@s_hplif")
    ap.add_argument("--ours-name", default="S-EEGNeX")
    ap.add_argument("--candidates", default="seegnex@s_hplif,seegnex@hplif,seegnet@plif",
                    help="comma list of candidate keys named (as comments) under the (Ours) row; '' to disable")
    ap.add_argument("--metrics", default=None,
                    help=f"comma list for all cohorts; suffix +sd shows the SD; metrics: {sorted(METRICS)}. "
                         f"Default per cohort: {DEFAULT_METRICS}")
    for c in COHORTS:
        ap.add_argument(f"--metrics-{c.lower()}", default=None, help=f"override for {c} (default {DEFAULT_METRICS[c]})")
    ap.add_argument("--pct-digits", type=int, default=2, help="decimals for percentage metrics (AUC/kappa: 3)")
    ap.add_argument("--layout", choices=("per-cohort", "compact"), default="per-cohort",
                    help="what --splice inserts: three tables (default) or the single table*")
    ap.add_argument("--splice", type=Path, default=None, help="manuscript .tex with the two marker lines")
    ap.add_argument("--final", action="store_true", help="refuse to splice a block that still contains \\tbd")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if "_" in args.ours_name or "&" in args.ours_name or "%" in args.ours_name:
        raise SystemExit("--ours-name must be LaTeX-safe (no _ & %)")

    aliases = dict(a.split("=", 1) for a in args.alias)
    extras: Dict[str, List[Path]] = {c: [] for c in COHORTS}
    for e in args.extra:
        c, p = e.split("=", 1)
        if c.upper() not in extras:
            raise SystemExit(f"--extra cohort must be one of {COHORTS}")
        extras[c.upper()].append(Path(p))

    base = {"2C": load_bench(args.bench_2c, "2C", canonical=True),
            "SHU": load_bench(args.bench_shu, "SHU", canonical=True),
            "3C": load_per_cell(args.cells_3c, "3C", canonical_3c=True)}
    snn_paths = {"2C": args.snn_2c, "SHU": args.snn_shu, "3C": args.snn_3c}
    merged: Dict[str, Dict[str, Dict]] = {}
    for c in COHORTS:
        d = dict(base[c])
        p = snn_paths[c]
        if p is not None:
            if not Path(p).exists():
                raise SystemExit(f"--snn-{c.lower()} file not found: {p}")
            an = load_analyze(Path(p), c)
            if c == "3C":
                d.update(an)  # recipe-matched LOSO re-runs replace the paper-recipe rows
            else:
                for k, rec in an.items():
                    d.setdefault(k, rec)  # bench numbers stay canonical for the ANN baselines
        for p in extras[c]:
            if not p.exists():
                raise SystemExit(f"--extra file not found: {p}")
            ex = load_any(p, c)
            clash = sorted(set(ex) & set(d))
            if clash and not args.allow_override:
                raise SystemExit(f"--extra {c}={p}: keys already present {clash}; use --allow-override to replace")
            d.update(ex)
        merged[c] = d

    metrics_by_cohort = {c: parse_metrics(getattr(args, f"metrics_{c.lower()}") or args.metrics or DEFAULT_METRICS[c])
                         for c in COHORTS}
    candidates = [x for x in args.candidates.split(",") if x] if args.candidates else []
    out_rel = _rel(args.out)
    note = f"generated by scripts/build_dataset_tables.py -> {out_rel}/ ; regenerate, do not edit numbers by hand"

    long_rows, tex_blocks, md_blocks, rows_by_cohort = [], [], [], {}
    for c, fname in (("2C", "tab_2c.tex"), ("SHU", "tab_shu.tex"), ("3C", "tab_3c.tex")):
        rows = table_rows(merged[c], args.ours, args.ours_name, aliases, args.include)
        rows_by_cohort[c] = rows
        tex = render_table(c, rows, metrics_by_cohort[c], candidates, args.pct_digits, note)
        (args.out / fname).write_text(tex, encoding="utf-8")
        tex_blocks.append(tex)
        md_blocks.append(render_markdown(c, rows, metrics_by_cohort[c], args.pct_digits))
        for r in rows:
            m, rec = r["model"], r["rec"]
            row = dict(cohort=c, roster_key=m.get("key"), model=m["name"] + (" (Ours)" if r["ours"] else ""),
                       venue="" if r["ours"] else m["venue"],
                       status="filled" if rec is not None else ("TBD" if m.get("pending") else "not_run"))
            for mm in METRICS:
                row[f"{mm}_mean"], row[f"{mm}_sd"] = rec["values"][mm] if rec else (float("nan"), float("nan"))
            row.update(dict(n_subjects=rec["n_subjects"] if rec else "", n_cells=rec["n_cells"] if rec else "",
                            protocol=rec["protocol"] if rec else "", source=rec["source"] if rec else "",
                            source_key=rec["source_key"] if rec else "", flag=rec["flag"] if rec else ""))
            long_rows.append(row)
        for cand in candidates:  # provenance of the candidates (numbers live here, not in the manuscript)
            rec = merged[c].get(aliases.get(cand, cand))
            if rec is None:
                continue
            row = dict(cohort=c, roster_key=cand, model=f"candidate: {CANDIDATE_NAMES.get(cand, cand)}", venue="",
                       status="candidate")
            for mm in METRICS:
                row[f"{mm}_mean"], row[f"{mm}_sd"] = rec["values"][mm]
            row.update(dict(n_subjects=rec["n_subjects"], n_cells=rec["n_cells"], protocol=rec["protocol"],
                            source=rec["source"], source_key=rec["source_key"], flag=rec["flag"]))
            long_rows.append(row)

    compact = render_compact(rows_by_cohort, metrics_by_cohort, args.pct_digits, note)
    (args.out / "tab_compact_all.tex").write_text(compact, encoding="utf-8")
    pd.DataFrame(long_rows).to_csv(args.out / "dataset_tables_long.csv", index=False)
    md = ["# Per-cohort comparison tables", "",
          f"Generated by `scripts/build_dataset_tables.py` -> `{out_rel}/`. † pilot (split 41 only); "
          "‡ 3C historical run (patience 100, not recipe-matched); (P) fine-tuned from a public checkpoint; "
          "TBD = pending (parallel session); — (not run) = never scheduled.", ""] + md_blocks
    cand_rows = [r for r in long_rows if r["status"] == "candidate"]
    if cand_rows:
        md += ["### Candidates for the (Ours) row (not printed in the manuscript; pre-registered main = seegnex@s_hplif)", "",
               "| cohort | candidate | acc ± SD | bal_acc | macro_f1 | auc | protocol | flag |", "|:--|:--|--:|--:|--:|--:|:--|:--|"]
        for r in cand_rows:
            md.append(f"| {r['cohort']} | {r['model']} | {100 * r['acc_mean']:.2f} ± {100 * r['acc_sd']:.2f} | "
                      f"{100 * r['bal_acc_mean']:.2f} | {100 * r['macro_f1_mean']:.2f} | {r['auc_mean']:.3f} | "
                      f"{r['protocol']} | {r['flag']} |")
        md.append("")
    (args.out / "dataset_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if args.splice:
        block = compact if args.layout == "compact" else "\n".join(tex_blocks)
        if args.final and "\\tbd" in block:
            raise SystemExit("--final: the generated block still contains \\tbd placeholders; not splicing")
        splice(args.splice, block)
        print(f"spliced {'1 compact table*' if args.layout == 'compact' else '3 tables'} into {args.splice}")
    print("\n".join(md_blocks))
    print("tex ->", args.out)


if __name__ == "__main__":
    main()
