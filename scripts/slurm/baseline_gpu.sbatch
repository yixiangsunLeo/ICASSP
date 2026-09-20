#!/bin/bash
# One GPU job of the external-baseline round (D-034): N independent `code/run.py` processes
# sharing ONE GPU (small models are launch-bound and leave a 4090 mostly idle, so packing
# them is how we get more out of the 4-GPU QOS cap). Same resources as bench_gpu.sbatch.
#
#   sbatch -J <name> scripts/slurm/baseline_gpu.sbatch "<config>|<models>|<tag>" ["<config>|<models>|<tag>" ...]
#
# Each argument is one process: the experiment config, a comma-separated model list and the
# --tag-suffix under which its failures/resolved files are written. All processes run
# concurrently; the job exits non-zero if any of them does.
#
# SMOKE_FIRST=1 (default): before the full runs, every process first trains fold 0 of the
# first split for 2 epochs into outputs/experiments/_smoke/<tag> (fresh, --no-resume). Any
# failure there (import, shapes, CUDA OOM at the recipe batch size) aborts the whole job
# within minutes instead of after hours in the queue. Relay links pass SMOKE_FIRST=0.
#
# Normally submitted through scripts/slurm/submit_baselines.sh, not by hand.
#
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 47:30:00
#SBATCH --mail-type=FAIL

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh" ]; then
  # shellcheck disable=SC1091
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh"
else
  # shellcheck disable=SC1091
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
fi

[ $# -ge 1 ] || { echo "ERROR: at least one '<config>|<models>|<tag>' spec required" >&2; exit 2; }
SMOKE_FIRST="${SMOKE_FIRST:-1}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

echo "=== job ${SLURM_JOB_NAME:-?} / ${SLURM_JOB_ID:-?} ==="
echo "Project : $PROJECT_ROOT"
echo "Env     : $MI_ENV"
echo "Node    : $(hostname)"
echo "Specs   : $*"
echo "Smoke   : SMOKE_FIRST=$SMOKE_FIRST epochs=$SMOKE_EPOCHS"
echo "Start   : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) dirty file(s)"

mi_activate_conda
nvidia-smi || true
mi_cuda_failfast

# Each process gets a fair share of the job's CPUs (data loaders use 2 workers each).
N_PROC=$#
THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / N_PROC )); [ "$THREADS" -ge 1 ] || THREADS=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

parse_spec() {  # "<config>|<models>|<tag>" -> CONFIG MODELS TAG
  IFS='|' read -r CONFIG MODELS TAG <<< "$1"
  [ -f "$CONFIG" ] || { echo "ERROR: missing config $CONFIG" >&2; return 2; }
  [ -n "$MODELS" ] && [ -n "$TAG" ] || { echo "ERROR: bad spec '$1'" >&2; return 2; }
}

# ---- 1. smoke: sequential, fresh, tiny ------------------------------------------------
if [ "$SMOKE_FIRST" = "1" ]; then
  for spec in "$@"; do
    parse_spec "$spec" || exit 2
    echo "--- smoke [$TAG] $MODELS ($(date -Is))"
    python -u code/run.py --config "$CONFIG" --device cuda --models "$MODELS" \
      --tag-suffix "${TAG}_smoke" --folds-subset 0 --max-epochs "$SMOKE_EPOCHS" --no-resume \
      --out "outputs/experiments/_smoke/${TAG}" --ckpt-dir "checkpoints/_smoke/${TAG}"
    rc=$?
    if [ $rc -ne 0 ]; then echo "SMOKE FAILED [$TAG] rc=$rc — aborting job"; exit 3; fi
    python - "outputs/experiments/_smoke/${TAG}/failures__${TAG}_smoke.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); n = int(d.get("n_cells_failed", 0))
print(f"smoke summary: n_cells_failed={n}")
sys.exit(3 if n else 0)
PY
    [ $? -eq 0 ] || { echo "SMOKE FAILED [$TAG]: failed cells — aborting job"; exit 3; }
  done
  echo "--- all smokes passed ($(date -Is))"
fi

# ---- 2. full runs: concurrent on the one GPU ---------------------------------------------
declare -a PIDS TAGS
for spec in "$@"; do
  parse_spec "$spec" || exit 2
  echo "--- start [$TAG] $MODELS ($(date -Is)) -> logs/slurm/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-0}__${TAG}.log"
  python -u code/run.py --config "$CONFIG" --device cuda --models "$MODELS" --tag-suffix "$TAG" \
    > "logs/slurm/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-0}__${TAG}.log" 2>&1 &
  PIDS+=($!); TAGS+=("$TAG")
done

RC=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"; rc=$?
  echo "--- done  [${TAGS[$i]}] rc=$rc ($(date -Is))"
  [ $rc -eq 0 ] || RC=$rc
done

echo "End     : $(date -Is)"
echo "Exit    : $RC"
if command -v sacct >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,JobName%18,State%13,Elapsed%11,Timelimit%11,ReqMem%9,MaxRSS%9,AllocTRES%40 \
    | tee "logs/slurm/sacct_${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}.txt"
fi
exit $RC
