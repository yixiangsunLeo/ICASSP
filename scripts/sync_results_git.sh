#!/usr/bin/env bash
# Stage (and optionally commit + push) the small result files that GitHub is allowed
# to carry. Training still writes to the same relative paths; this script is the
# only supported way to put those files on GitHub.
#
#   bash scripts/sync_results_git.sh              # stage + print status
#   SYNC_COMMIT=1 bash scripts/sync_results_git.sh
#   SYNC_COMMIT=1 SYNC_PUSH=1 bash scripts/sync_results_git.sh
#
# NEVER stages: *.pt *.pth *.ckpt *.npz paths.local.yaml weights/ checkpoints/ logs/
# NEVER force-pushes. NEVER amends.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -d .git ] || { echo "SYNC FAIL: not a git repo"; exit 2; }

# Honour .gitignore (result.json etc. are re-included there). -f is a safety net
# if a new nested directory is not yet un-ignored.
_find_exp() {
  find outputs/experiments \( -path '*/_smoke/*' -o -path '*/_smoke' \) -prune -o "$@" -print
}
mapfile -t FILES < <(
  {
    _find_exp -name result.json
    _find_exp -name 'failures__*.json'
    _find_exp -name 'resolved_config__*.yaml'
    _find_exp -name 'resolved_model_params__*.json'
    _find_exp -name 'meta_cross_subject__*.json'
    _find_exp -name 'cross_subject_folds__*.json'
    find outputs/analysis \( -name '*.csv' -o -name '*.md' \) 2>/dev/null
    find results_snapshot_2026-09-13 -path '*/tables/*' \( -name '*.csv' -o -name '*.json' \) 2>/dev/null
    find results_snapshot_2026-09-13 -path '*/cells/*/result.json' 2>/dev/null
  } | sort -u
)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "SYNC: no result files found under outputs/ or results_snapshot_"
  echo "      train first, or the copy of processed cells did not arrive."
  exit 0
fi

echo "SYNC: staging ${#FILES[@]} small result files"
# No -f: _smoke and binaries must stay ignored. If git refuses a path, fix
# .gitignore instead of force-adding.
git add -- "${FILES[@]}"

# Living docs the other side must see after a pull.
for extra in docs/STATUS.md progress.md docs/DECISIONS.md; do
  [ -f "$extra" ] && git add -- "$extra"
done

echo "---- staged (pathspec) ----"
git diff --cached --stat
echo "---- check: no binaries ----"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|npz|bdf|edf|mat)$'; then
  echo "SYNC FAIL: binary slipped into the index. Unstage it and fix .gitignore."
  exit 2
fi
if git diff --cached --name-only | grep -E 'paths\.local\.ya?ml$'; then
  echo "SYNC FAIL: paths.local.yaml must never be committed."
  exit 2
fi

n_cells="$(find outputs/experiments \( -path '*/_smoke/*' -o -path '*/_smoke' \) -prune -o -name result.json -print 2>/dev/null | wc -l | tr -d ' ')"
echo "on-disk result.json count: $n_cells"

if [ "${SYNC_COMMIT:-0}" != "1" ]; then
  echo "SYNC: staged only. Re-run with SYNC_COMMIT=1 to commit, SYNC_PUSH=1 to push."
  exit 0
fi

if git diff --cached --quiet; then
  echo "SYNC: nothing new to commit"
else
  git commit -m "$(cat <<EOF
Sync experiment result.json and analysis tables (${n_cells} cells on disk).

EOF
)"
fi

if [ "${SYNC_PUSH:-0}" = "1" ]; then
  # 3090 clone: origin is yixiangsunLeo/ICASSP. yuan tree: remote name is icassp.
  if git remote get-url origin 2>/dev/null | grep -q 'yixiangsunLeo/ICASSP'; then
    git push origin HEAD
  elif git remote get-url icassp >/dev/null 2>&1; then
    git push icassp HEAD
  else
    echo "SYNC FAIL: no remote pointing at yixiangsunLeo/ICASSP"
    echo "  git remote add icassp git@github.com:yixiangsunLeo/ICASSP.git"
    exit 2
  fi
fi
echo "SYNC OK"
