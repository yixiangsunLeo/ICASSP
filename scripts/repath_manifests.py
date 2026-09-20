#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite absolute prefixes inside processed_manifest.csv and verify every ok npz exists.

Training loads ``npz_path`` as-is (no re-rooting). After you copy a processed tree to a
new machine you MUST rewrite the three path columns (npz_path, meta_path, report_path)
and then verify. This script is the only supported way to do that.

Examples
--------
Dry-run (print what would change, then verify current paths)::

    python scripts/repath_manifests.py \\
        --csv /data/eog_ecg_clean/processed_manifest.csv \\
        --rewrite /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean=/data/eog_ecg_clean \\
        --expect-ok 148 --expect-subjects 51 --dry-run --verify

Apply + verify::

    python scripts/repath_manifests.py \\
        --csv /data/eog_ecg_clean/processed_manifest.csv \\
        --rewrite OLDPREFIX=NEWPREFIX \\
        --expect-ok 148 --expect-subjects 51 --verify
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PATH_COLS = ("npz_path", "meta_path", "report_path")


def _parse_rewrite(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--rewrite must be OLD=NEW, got: {raw!r}")
        old, new = raw.split("=", 1)
        if not old:
            raise SystemExit(f"empty OLD prefix in {raw!r}")
        if not new or new in ("/", "."):
            raise SystemExit(
                f"NEW prefix is empty or unsafe in {raw!r}. "
                "Did you forget to export DATA_ROOT / REPO?"
            )
        if not new.startswith("/"):
            raise SystemExit(f"NEW prefix must be an absolute path, got: {new!r}")
        out.append((old, new))
    return out


def _rewrite_value(value: str, pairs: list[tuple[str, str]]) -> str:
    if not value:
        return value
    for old, new in pairs:
        if value.startswith(old):
            return new + value[len(old) :]
    return value


def _load(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{csv_path}: empty CSV")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _write(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)


def _ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if (r.get("status") or "").strip() == "ok"]


def _verify(
    csv_path: Path,
    rows: list[dict[str, str]],
    *,
    expect_ok: int | None,
    expect_subjects: int | None,
    expect_rows: int | None,
) -> int:
    n_fail = 0
    ok = _ok_rows(rows)
    subjects = sorted({(r.get("subject_id") or "").strip() for r in ok})
    print(f"[verify] {csv_path}")
    print(f"  rows={len(rows)} ok={len(ok)} failed={len(rows) - len(ok)} subjects={len(subjects)}")
    if expect_rows is not None and len(rows) != expect_rows:
        print(f"  FAIL: expect-rows {expect_rows}, got {len(rows)}")
        n_fail += 1
    if expect_ok is not None and len(ok) != expect_ok:
        print(f"  FAIL: expect-ok {expect_ok}, got {len(ok)}")
        n_fail += 1
    if expect_subjects is not None and len(subjects) != expect_subjects:
        print(f"  FAIL: expect-subjects {expect_subjects}, got {len(subjects)} {subjects}")
        n_fail += 1

    missing = []
    leftover = Counter()
    for r in ok:
        p = (r.get("npz_path") or "").strip()
        if not p:
            missing.append(("<empty npz_path>", r.get("subject_id"), r.get("session_id")))
            continue
        if p.startswith("/share/workspace2/") or p.startswith("/share/home/yuan/"):
            leftover[p.rsplit("/", 3)[0] if "/" in p else p] += 1
        if not Path(p).is_file():
            missing.append((p, r.get("subject_id"), r.get("session_id")))
    if leftover:
        print("  FAIL: still contains old-cluster prefixes (copy+rewrite incomplete):")
        for pref, n in leftover.most_common(8):
            print(f"    {n} rows still under {pref}...")
        n_fail += 1
    if missing:
        print(f"  FAIL: {len(missing)} ok-row npz missing on this machine")
        for p, sub, ses in missing[:12]:
            print(f"    {sub} {ses} -> {p}")
        if len(missing) > 12:
            print(f"    ... {len(missing) - 12} more")
        n_fail += 1
    if n_fail == 0:
        print("  OK")
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="processed_manifest.csv to rewrite / verify")
    ap.add_argument(
        "--rewrite",
        action="append",
        default=[],
        help="OLDPREFIX=NEWPREFIX (repeatable). Applied to npz_path, meta_path, report_path.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print counts, do not write")
    ap.add_argument("--verify", action="store_true", help="after rewrite (or alone) check every ok npz exists")
    ap.add_argument("--expect-ok", type=int, default=None)
    ap.add_argument("--expect-subjects", type=int, default=None)
    ap.add_argument("--expect-rows", type=int, default=None)
    args = ap.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        print(f"FAIL: manifest not found: {csv_path}", file=sys.stderr)
        return 2

    fieldnames, rows = _load(csv_path)
    for col in ("npz_path", "status", "subject_id"):
        if col not in fieldnames:
            print(f"FAIL: {csv_path} missing column {col}", file=sys.stderr)
            return 2

    pairs = _parse_rewrite(args.rewrite) if args.rewrite else []
    changed = 0
    if pairs:
        for r in rows:
            for col in PATH_COLS:
                if col not in r:
                    continue
                old = r[col] or ""
                new = _rewrite_value(old, pairs)
                if new != old:
                    r[col] = new
                    changed += 1
        print(f"[rewrite] {csv_path}: {changed} path-cell(s) updated, {len(pairs)} prefix pair(s)")
        if args.dry_run:
            print("[rewrite] dry-run: not writing")
        elif changed:
            bak = csv_path.with_suffix(csv_path.suffix + ".bak")
            if not bak.exists():
                bak.write_bytes(csv_path.read_bytes())
                print(f"[rewrite] backup {bak}")
            _write(csv_path, fieldnames, rows)
            print("[rewrite] wrote")
        else:
            print("[rewrite] nothing matched — prefixes already new, or --rewrite OLD is wrong")

    n_fail = 0
    if args.verify or not pairs:
        n_fail = _verify(
            csv_path,
            rows,
            expect_ok=args.expect_ok,
            expect_subjects=args.expect_subjects,
            expect_rows=args.expect_rows,
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
