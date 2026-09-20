"""Shared helpers for the external-baseline adapters (format conversion only, never tuning).

Protocol (``docs/PROTOCOL_external_baselines.md`` P4/P5): every model receives the *same*
normalised trials ``[B, C, 1000] @ 250 Hz`` as the 18 ANN entries and the SNNs. Adapters may
only re-format that tensor into what the upstream network expects — resampling to the
model's native rate, cutting fixed-length patches, looking up channel ids in the model's
own electrode vocabulary — and load the official pretrained weights. No extra filtering,
no amplitude rescaling, no alignment.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ...utils.logging_utils import get_logger
from ...utils.paths import LOCAL_PATHS_CONFIG, PROJECT_ROOT

logger = get_logger("models.external")

#: Where the official checkpoints live. Resolution order (first hit wins):
#:   1. env ``ICASSP_WEIGHTS_DIR``
#:   2. ``code/configs/paths.local.yaml`` key ``pretrained_weights_dir`` (optional)
#:   3. ``<project root>/weights`` (on this cluster a symlink to ../Benchmark/weights)
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"

#: Old 10-20 names that some datasets (SHU 2022) still use, mapped to the modern names
#: found in the electrode vocabularies of LaBraM / EEGPT. Positions are identical
#: (Jurcak et al. 2007); this is a rename, not an interpolation.
LEGACY_1020_ALIASES: Dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def weights_dir() -> Path:
    env = os.environ.get("ICASSP_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser()
    try:
        if LOCAL_PATHS_CONFIG.exists():
            with open(LOCAL_PATHS_CONFIG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            p = cfg.get("pretrained_weights_dir")
            if p:
                p = Path(str(p)).expanduser()
                return p if p.is_absolute() else PROJECT_ROOT / p
    except Exception as exc:  # pragma: no cover - defensive: a broken local yaml must not kill a run
        logger.warning("could not read %s (%s); falling back to %s",
                       LOCAL_PATHS_CONFIG, exc, DEFAULT_WEIGHTS_DIR)
    return DEFAULT_WEIGHTS_DIR


def resolve_weights_path(filename: str, *, required: bool = True) -> Optional[Path]:
    """Absolute path of an official checkpoint, or ``None`` when absent and not required."""
    p = weights_dir() / filename
    if p.exists():
        return p
    if required:
        raise FileNotFoundError(
            f"pretrained checkpoint '{filename}' not found in {weights_dir()}. See "
            "code/models/external/README.md for the download instructions, or set "
            "ICASSP_WEIGHTS_DIR / paths.local.yaml:pretrained_weights_dir."
        )
    return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# channels
# --------------------------------------------------------------------------- #
def dataset_channel_names(channel_set: str) -> List[str]:
    """Electrode names of one of our cohorts, from ``code/configs/datasets/<set>.yaml``.

    ``channel_set`` is ``"wbci_shu"`` (58 ch, used by the 2C and 3C cohorts) or ``"shu"``
    (32 ch). Injected into a model via its yaml ``per_dataset`` block so no adapter hard-codes
    a dataset.
    """
    path = PROJECT_ROOT / "code" / "configs" / "datasets" / f"{channel_set}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown channel_set '{channel_set}': {path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("channels")
    if not names:
        raise ValueError(f"{path} has no 'channels' list")
    return [str(c) for c in names]


def canonical_channel_name(name: str) -> str:
    """Upper-case, strip dots/spaces, and map legacy 10-20 names (T3 -> T7, ...)."""
    n = str(name).strip().strip(".").upper()
    return LEGACY_1020_ALIASES.get(n, n)


def match_channels(names: Sequence[str], vocabulary: Iterable[str]
                   ) -> Tuple[List[int], List[str], List[str]]:
    """Return ``(keep_idx, matched_names, dropped_names)`` for ``names`` against a vocabulary.

    ``keep_idx`` indexes the *input* channel axis (channels present in the vocabulary, in
    input order); ``matched_names`` are their canonical names; ``dropped_names`` are the
    inputs the vocabulary cannot represent (e.g. earlobe A1/A2 for EEGPT).
    """
    vocab = {canonical_channel_name(v) for v in vocabulary}
    keep, matched, dropped = [], [], []
    for i, n in enumerate(names):
        c = canonical_channel_name(n)
        if c in vocab:
            keep.append(i)
            matched.append(c)
        else:
            dropped.append(str(n))
    return keep, matched, dropped


# --------------------------------------------------------------------------- #
# time axis
# --------------------------------------------------------------------------- #
def resample_time(x: torch.Tensor, sfreq_in: float, sfreq_out: float,
                  n_out: Optional[int] = None) -> torch.Tensor:
    """Resample the last axis of ``x`` from ``sfreq_in`` to ``sfreq_out`` Hz.

    Uses torchaudio's windowed-sinc polyphase resampler (exact for the rational ratios we
    need: 250->200 gives 1000->800 samples, 250->256 gives 1000->1024). Our cohorts are
    band-limited to 0.5-40 Hz, far below either Nyquist, so no anti-alias margin is lost.
    ``n_out`` crops or zero-pads the result to an exact length when the ratio is inexact.
    """
    if float(sfreq_in) == float(sfreq_out):
        y = x
    else:
        try:
            import torchaudio.functional as AF  # noqa: WPS433 - optional at import time
            y = AF.resample(x, int(round(sfreq_in)), int(round(sfreq_out)))
        except ImportError:  # pragma: no cover - torchaudio is installed in mi_torch_cu118
            logger.warning("torchaudio missing: falling back to linear interpolation")
            n = int(round(x.shape[-1] * float(sfreq_out) / float(sfreq_in)))
            flat = x.reshape(-1, 1, x.shape[-1])
            y = F.interpolate(flat, size=n, mode="linear", align_corners=False)
            y = y.reshape(*x.shape[:-1], n)
    if n_out is not None and y.shape[-1] != n_out:
        if y.shape[-1] > n_out:
            y = y[..., :n_out]
        else:
            y = F.pad(y, (0, n_out - y.shape[-1]))
    return y


# --------------------------------------------------------------------------- #
# contract helpers
# --------------------------------------------------------------------------- #
def squeeze_input(x: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, C, T]`` or ``[B, 1, C, T]`` and return ``[B, C, T]``."""
    if x.dim() == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.dim() != 3:
        raise ValueError(f"expected [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}")
    return x


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def warn_unknown_params(known: Iterable[str], params: Dict, who: str) -> Dict:
    """Drop and log keys the adapter does not understand (mirrors ``registry._select``)."""
    known = set(known)
    unknown = sorted(k for k in params if k not in known)
    if unknown:
        logger.warning("%s: ignoring unknown params %s", who, unknown)
    return {k: v for k, v in params.items() if k in known}


__all__ = [
    "DEFAULT_WEIGHTS_DIR", "LEGACY_1020_ALIASES", "weights_dir", "resolve_weights_path",
    "sha256_file", "dataset_channel_names", "canonical_channel_name", "match_channels",
    "resample_time", "squeeze_input", "count_params", "warn_unknown_params",
]
