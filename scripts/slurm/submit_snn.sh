#!/bin/bash
# Submit one spiking-round experiment config as N single-GPU jobs (one per model group),
# each relay-chained across `N_LINKS` 48 h jobs so long groups survive the QOS wall clock.
#
#   bash scripts/slurm/submit_snn.sh <config.yaml> <tag> "<group1>" ["<group2>" ...] [-- extra run.py args]
#
#   # SHU pilot: 4 models on one GPU, split 41 only
#   bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_shu_zs_lr1e3.yaml pz3 \
#        "eegnex,seegnex@hplif,seegnex@s_hplif,seegnet@plif" -- --split-seeds 41
#
#   # 2C main run: 3 groups -> 3 GPUs, both split seeds (config default)
#   bash scripts/slurm/submit_snn.sh code/configs/experiments/snn_wbci2c_ea_lr1e3.yaml c2e \
#        "eegnex,seegnet@plif" "seegnex@hplif,seegnex@s_lif" "seegnex@s_plif,seegnex@s_hplif"
#
# Environment: N_LINKS (default 2), PARTITION (default gpu2node), DRY_RUN=1 prints only.
# Every job writes into the config's output dir under its own --tag-suffix; finished cells
# are skipped on resume, so re-submitting the same command is always safe.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIG="${1:?config path required}"; shift
TAG="${1:?short tag for job names required}"; shift
[ -f "$CONFIG" ] || { echo "ERROR: missing $CONFIG" >&2; exit 2; }

GROUPS_ARR=()
EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
  GROUPS_ARR+=("$1"); shift
done
[ ${#GROUPS_ARR[@]} -gt 0 ] || { echo "ERROR: at least one model group required" >&2; exit 2; }

N_LINKS="${N_LINKS:-2}"
PARTITION="${PARTITION:-gpu2node}"
DRY_RUN="${DRY_RUN:-0}"

# Every listed model must resolve (typos would otherwise surface hours later on the GPU).
python3 - "${GROUPS_ARR[@]}" <<'PY'
import sys
from code.models.model_specs import resolve_model_params, split_alias
from code.models.registry import MODEL_NAMES
names = [m for g in sys.argv[1:] for m in g.split(",") if m]
dupes = {m for m in names if names.count(m) > 1}
if dupes:
    sys.exit(f"ERROR: duplicated entries across groups: {sorted(dupes)}")
for n in names:
    base, _ = split_alias(n)
    if base not in MODEL_NAMES:
        sys.exit(f"ERROR: unknown model '{n}'")
    resolve_model_params(n)          # raises on an undefined @variant
print(f"roster check: {len(names)} entries OK")
PY

mkdir -p logs/slurm
IDS_FILE="logs/slurm/snn_${TAG}_job_ids.txt"
: > "$IDS_FILE"
echo "config  : $CONFIG"
echo "groups  : ${#GROUPS_ARR[@]}  links/group: $N_LINKS  partition: $PARTITION"
echo "extra   : ${EXTRA[*]:-<none>}"

ALL_IDS=()
for i in "${!GROUPS_ARR[@]}"; do
  models="${GROUPS_ARR[$i]}"
  g="g$((i + 1))"
  prev=""
  for link in $(seq 1 "$N_LINKS"); do
    name="${TAG}_${g}_${link}"
    dep=()
    [ -n "$prev" ] && dep=(--dependency="afterany:${prev}")
    cmd=(sbatch -J "$name" -p "$PARTITION" "${dep[@]}" scripts/slurm/bench_gpu.sbatch
         "$CONFIG" --models "$models" --tag-suffix "${TAG}_${g}" "${EXTRA[@]}")
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] ${cmd[*]}"; prev="DRY_${name}"; continue
    fi
    out="$("${cmd[@]}")"
    id="${out##* }"
    echo "$name -> $id ${prev:+(after $prev)}   [$models]"
    echo "$name $id $models" >> "$IDS_FILE"
    ALL_IDS+=("$id"); prev="$id"
  done
done
[ "$DRY_RUN" = "1" ] && exit 0
echo "job ids -> $IDS_FILE"
echo "monitor : squeue -u \$USER -o '%.10i %.14j %.2t %.11M %R'"
echo "if PENDING (Resources) on $PARTITION: scontrol update jobid=<id> partition=gpu3node"
