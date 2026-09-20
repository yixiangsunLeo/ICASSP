"""External published baselines (2024–2025), each vendored from its authors' official code.

This package is **lazily imported**: ``code/models/registry.py`` only pulls in
``code.models.external.<name>.adapter`` when that model is actually requested, so a missing
optional dependency or a bug in one adapter can never break the import of the registry for
the other models (the SNN main run imports the registry at job start).

Keep this module free of heavy imports: it must stay importable with nothing but the
standard library, because ``registry.MODEL_NAMES`` (used by the Slurm roster check) needs
the names below at import time.

Entries (protocol: ``docs/PROTOCOL_external_baselines.md``, decision D-034):

  * ``tcformer``  — TCFormer, Altaheri et al., Sci Rep 2025 (github.com/altaheri/TCFormer)
  * ``labram``    — LaBraM-base, Jiang et al., ICLR 2024 (official ``modeling_finetune.py``)
  * ``eegpt``     — EEGPT (large, 58 ch / 4 s / 256 Hz), Wang et al., NeurIPS 2024
  * ``cbramod``   — CBraMod, Wang et al., ICLR 2025
  * ``adahgnn``   — AdaHGNN / EEG-AdaMSHyper, DeepHypergraph project_V4 (lab, 2026-01-16)

Every adapter exposes ``build(*, n_channels, n_times, n_classes, sfreq, params) -> nn.Module``
returning a module with the project contract ``forward(x[B,C,T]) -> {"logits", "features",
"confidence"}`` plus ``describe()``.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional

EXTERNAL_MODEL_NAMES: List[str] = ["tcformer", "labram", "eegpt", "cbramod", "adahgnn"]
#: D-034 published-FM roster. ``adahgnn`` (D-037) is extra and has its own ``bl_*_adahgnn``
#: configs so the live D-034 ``bl_*_zs_lr1e4.yaml`` files are not rewritten while jobs run.
D034_EXTERNAL_MODEL_NAMES: List[str] = ["tcformer", "labram", "eegpt", "cbramod"]


def build_external(name: str, *, n_channels: int, n_times: int, n_classes: int,
                   sfreq: float = 250.0, params: Optional[Dict] = None):
    """Import ``code.models.external.<name>.adapter`` on demand and build the model."""
    key = (name or "").lower().strip()
    if key not in EXTERNAL_MODEL_NAMES:
        raise ValueError(f"unknown external model '{name}'. Known: {EXTERNAL_MODEL_NAMES}")
    module = importlib.import_module(f"{__name__}.{key}.adapter")
    return module.build(n_channels=int(n_channels), n_times=int(n_times),
                        n_classes=int(n_classes), sfreq=float(sfreq),
                        params=dict(params or {}))


__all__ = ["EXTERNAL_MODEL_NAMES", "D034_EXTERNAL_MODEL_NAMES", "build_external"]
