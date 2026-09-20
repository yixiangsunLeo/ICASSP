#!/usr/bin/env bash
# 3090 / new-machine preflight. The migration AI MUST run this from the repo root
# after paths.local.yaml exists and manifests have been rewritten.
# Exit 0 and print "PREFLIGHT OK" only when every gate passes. Otherwise print
# "PREFLIGHT FAIL: ..." and exit 1. Do not train until this is OK.

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(pwd)"
fail() { echo "PREFLIGHT FAIL: $1"; exit 1; }
note() { echo "  - $1"; }

echo "==== ICASSP handoff preflight ===="
echo "repo   : $ROOT"
echo "user   : $(whoami)"
echo "host   : $(hostname)"
echo "date   : $(date -Iseconds)"

[ -f AGENTS.md ] && [ -d .git ] || fail "not the ICASSP repo root (need AGENTS.md + .git)"
[ -f HANDOFF.md ] || fail "HANDOFF.md missing — pull the portable remote"
[ -f code/configs/paths.local.yaml ] || fail "code/configs/paths.local.yaml missing. Copy docs/handoff/PATHS_LOCAL.template.yaml and fill it."
grep -q '/CHANGE/ME' code/configs/paths.local.yaml && fail "paths.local.yaml still contains /CHANGE/ME"
grep -Eq '/share/workspace2/|/share/home/yuan/' code/configs/paths.local.yaml \
  && fail "paths.local.yaml still points at the 4090 cluster (/share/workspace2 or /share/home/yuan). Rewrite it."

command -v conda >/dev/null || fail "conda not on PATH. Install miniconda or export CONDA_ROOT and source conda.sh"
# shellcheck disable=SC1091
if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${MI_ENV:-mi_torch_cu118}" || fail "cannot activate env ${MI_ENV:-mi_torch_cu118}. Run HANDOFF Phase 4."

python - <<'PY' || fail "python import / version gate failed (see above)"
import sys
print("python", sys.version.split()[0])
if not sys.version.startswith("3.10"):
    raise SystemExit("need Python 3.10.x")
import torch, numpy, scipy, sklearn, pandas, yaml, einops, timm
print("torch", torch.__version__, "cuda_build", torch.version.cuda, "avail", torch.cuda.is_available())
if not str(torch.__version__).startswith("2.7.1"):
    print("WARN: torch is not 2.7.1 — results will not be bit-comparable; code needs >= 2.4")
if tuple(int(x) for x in numpy.__version__.split(".")[:2]) < (2, 0):
    raise SystemExit(f"need numpy 2.x, got {numpy.__version__}")
print("numpy", numpy.__version__, "timm", timm.__version__, "einops", einops.__version__)
try:
    import torchaudio
    print("torchaudio", torchaudio.__version__)
except Exception as e:
    raise SystemExit(f"torchaudio missing (needed for LaBraM/EEGPT/CBraMod): {e}")
PY

# CUDA is required on a GPU node; on a login node this may fail — then we only warn.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "WARN: torch.cuda.is_available() is False here (login node?). GPU jobs will fail-fast. Recheck on a GPU node."
else
  python - <<'PY'
import torch
print("gpu", torch.cuda.get_device_name(0), "sm", ".".join(map(str, torch.cuda.get_device_capability(0))))
PY
fi

# Resolve manifests. PREFLIGHT_COHORTS=2c,shu,3c (default all). Comma list, lower-case.
python - <<'PY' || fail "manifest resolve / repath verify failed"
from pathlib import Path
import os, sys, subprocess
sys.path.insert(0, ".")
from code.utils.paths import load_paths
P = load_paths(require_raw=False)
wanted = {x.strip().lower() for x in os.environ.get("PREFLIGHT_COHORTS", "2c,shu,3c").split(",") if x.strip()}
all_specs = {
    "2c": ("2C", P.processed_manifest, 148, 51, 153),
    "shu": ("SHU", P.shu_processed_manifest, 125, 25, 125),
    "3c": ("3C", P.wbci_3c_processed_manifest, 33, 11, 33),
}
unknown = wanted - set(all_specs)
if unknown:
    raise SystemExit(f"PREFLIGHT_COHORTS unknown: {sorted(unknown)}")
specs = [all_specs[k] for k in ("2c", "shu", "3c") if k in wanted]
print("PREFLIGHT_COHORTS", ",".join(k for k in ("2c", "shu", "3c") if k in wanted))
print("resolved manifests:")
for name, path, ok, sub, rows in specs:
    print(f"  {name}: {path} exists={path.is_file()}")
    if not path.is_file():
        raise SystemExit(f"{name} manifest missing: {path}")
script = Path(".").resolve() / "scripts" / "repath_manifests.py"
for name, path, ok, sub, rows in specs:
    cmd = [sys.executable, str(script), "--csv", str(path), "--verify",
           "--expect-ok", str(ok), "--expect-subjects", str(sub), "--expect-rows", str(rows)]
    print("RUN", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"{name} verify failed")
PY

# Weights: required only if those files are present OR user intends FM. We require
# the three files if any of labram/eegpt/cbramod cells are unfinished.
python - <<'PY'
from pathlib import Path
import os, sys
sys.path.insert(0, ".")
from code.models.external._common import weights_dir
w = weights_dir()
print("weights_dir", w)
need = ["labram-base.pth", "eegpt.ckpt", "CBraMod.pth"]
missing = [n for n in need if not (w / n).is_file()]
if missing:
    print("WARN: pretrained weights missing:", ", ".join(missing))
    print("      EEGNeX / SNN / EEGNet / ATCNet / FBCNet / EEG-Deformer / TCFormer / AdaHGNN-MI do not need them.")
    print("      LaBraM / EEGPT / CBraMod WILL fail until you copy the three files (see HANDOFF Phase 8).")
else:
    print("weights: three FM checkpoints present")
PY

# Roster: every live alias must resolve (typos surface here, not on the GPU).
python - <<'PY' || fail "model roster failed"
from code.models.model_specs import resolve_model_params
from code.models.registry import MODEL_NAMES
need = [
    "eegnex", "eegnet_official", "seegnex@hplif", "seegnex@s_hplif", "seegnex@s_plif",
    "seegnex@s_lif", "seegnet@plif", "tcformer", "labram", "eegpt", "cbramod",
    "atcnet", "eeg_deformer", "fbcnet", "adahgnn@mi", "adahgnn@mi_mix",
]
for n in need:
    resolve_model_params(n)
print(f"roster check: {len(need)} entries OK (registry has {len(MODEL_NAMES)} names)")
PY

mkdir -p outputs/experiments logs/slurm checkpoints
echo "PREFLIGHT OK"
