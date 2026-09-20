#!/bin/bash
# Submit the external-baseline round (D-034) from a plan file, one relay-chained single-GPU
# job per plan line, with several `code/run.py` processes packed on each GPU.
#
#   bash scripts/slurm/submit_baselines.sh [scripts/slurm/baselines_plan.txt]
#
# Environment: N_LINKS (default 2 relay links of 47.5 h, `afterany`), PARTITION (default
# gpu2node), NICE (default 50: the spiking main run's jobs keep priority within our 4-GPU
# QOS), DRY_RUN=1 prints the sbatch commands only, ONLY=<regex> submits matching job names.
#
# Before anything is submitted every model on the plan is resolved for its cohort
# (`resolve_model_params(name, dataset=<data.name>)`), so a missing adapter, yaml or
# per_dataset block fails here and not hours later on a GPU. Re-running the same plan is
# always safe: finished cells are skipped, interrupted cells resume from last.pt.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PLAN="${1:-scripts/slurm/baselines_plan.txt}"
[ -f "$PLAN" ] || { echo "ERROR: missing plan $PLAN" >&2; exit 2; }
N_LINKS="${N_LINKS:-2}"
PARTITION="${PARTITION:-gpu2node}"
NICE="${NICE:-50}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"

# ---- parse plan ---------------------------------------------------------------------------
declare -a JOB_NAMES JOB_SPECS
while IFS= read -r line; do
  line="${line%%#*}"; line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -n "$line" ] || continue
  name="$(echo "$line" | cut -d';' -f1 | tr -d '[:space:]')"
  rest="$(echo "$line" | cut -d';' -f2-)"
  specs=()
  IFS=';' read -ra parts <<< "$rest"
  for p in "${parts[@]}"; do p="$(echo "$p" | tr -d '[:space:]')"; [ -n "$p" ] && specs+=("$p"); done
  [ ${#specs[@]} -gt 0 ] || { echo "ERROR: no specs on plan line: $line" >&2; exit 2; }
  if [ -n "$ONLY" ] && ! [[ "$name" =~ $ONLY ]]; then continue; fi
  JOB_NAMES+=("$name"); JOB_SPECS+=("$(IFS=$'\t'; echo "${specs[*]}")")
done < "$PLAN"
[ ${#JOB_NAMES[@]} -gt 0 ] || { echo "ERROR: nothing to submit (ONLY='$ONLY')" >&2; exit 2; }

# ---- roster check: every (config, model) must resolve for its cohort -----------------------
python3 - "${JOB_SPECS[@]}" <<'PY'
import sys, yaml
from pathlib import Path
from code.models.model_specs import resolve_model_params, split_alias
from code.models.registry import MODEL_NAMES
seen = set(); n = 0
for job in sys.argv[1:]:
    for spec in job.split("\t"):
        cfg, models, tag = spec.split("|")
        p = Path(cfg)
        if not p.exists():
            sys.exit(f"ERROR: missing config {cfg}")
        c = yaml.safe_load(p.read_text())
        ds = c["data"]["name"]; roster = set(c.get("models") or [])
        for m in models.split(","):
            if split_alias(m)[0] not in MODEL_NAMES:
                sys.exit(f"ERROR: unknown model '{m}' ({cfg})")
            if m not in roster:
                sys.exit(f"ERROR: '{m}' is not in the models list of {cfg}")
            resolve_model_params(m, dataset=ds)     # raises on undefined variant / bad yaml
            key = (cfg, m)
            if key in seen:
                sys.exit(f"ERROR: ({cfg}, {m}) appears twice in the plan")
            seen.add(key); n += 1
print(f"roster check: {n} (config, model) entries OK")
PY

mkdir -p logs/slurm
IDS_FILE="${IDS_FILE:-logs/slurm/bl_job_ids.txt}"
echo "plan    : $PLAN  jobs: ${#JOB_NAMES[@]}  links/job: $N_LINKS  partition: $PARTITION  nice: $NICE"

for i in "${!JOB_NAMES[@]}"; do
  name="${JOB_NAMES[$i]}"
  IFS=$'\t' read -ra specs <<< "${JOB_SPECS[$i]}"
  prev=""
  for link in $(seq 1 "$N_LINKS"); do
    jname="${name}_${link}"
    dep=(); [ -n "$prev" ] && dep=(--dependency="afterany:${prev}")
    # relay links skip the smoke (the first link already validated the models)
    exp=(--export="ALL,SMOKE_FIRST=$([ "$link" = 1 ] && echo 1 || echo 0)")
    cmd=(sbatch -J "$jname" -p "$PARTITION" --nice="$NICE" "${dep[@]}" "${exp[@]}"
         scripts/slurm/baseline_gpu.sbatch "${specs[@]}")
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] ${cmd[*]}"; prev="DRY_${jname}"; continue
    fi
    out="$("${cmd[@]}")"
    id="${out##* }"
    echo "$jname -> $id ${prev:+(after $prev)}   [${specs[*]}]"
    echo "$jname $id ${specs[*]}" >> "$IDS_FILE"
    prev="$id"
  done
done
[ "$DRY_RUN" = "1" ] && exit 0
echo "job ids appended to $IDS_FILE"
echo "monitor : squeue -u \$USER -o '%.10i %.14j %.2t %.11M %R'"
echo "progress: for r in wbci_shu/snn_wbci2c_zs_lr1e4_v1 shu/snn_shu_zs_lr1e4_v1 wbci_shu/snn_wbci3c_zs_lr1e4_v1; do ls outputs/experiments/\$r/cells | sed 's/__.*//' | sort | uniq -c; done"
