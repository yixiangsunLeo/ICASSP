"""AdaHGNN wrapper: original ``EEG_AdaMSHyper`` from project_V4, unedited.

The network in ``_official/models.py`` is the lab file byte-for-byte. This adapter only:

* builds a Config-like object (32/64 from source; 58-ch is the one extra scale the
  source Config refused) and dummy ``num_domain_classes=2`` so the DANN head is
  constructed exactly as in the source;
* calls ``model(x, alpha=0)`` and maps the first return (task logits) into the
  project contract ``{logits, features, confidence}``.

No grouped-conv rewrite, no homemade hypergraph, no deleted GRL. Training uses the
shared zs recipe (plain CE on task logits). Domain logits and the NHC constraint
are computed by the original forward but not added to our loss — same rule as
TCFormer (their optimiser is not used either).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .._common import count_params, squeeze_input, warn_unknown_params
from ._official.models import EEG_AdaMSHyper

SOURCE_PATH = "/share/home/yuan/LY/DeepHypergraph/project_V4"
SOURCE_DATE = "2026-01-16"
PAPER_NOTE = (
    "AdaHGNN / EEG-AdaMSHyper (DeepHypergraph project_V4, 2026-01-16). "
    "Runtime class is the vendored EEG_AdaMSHyper, unmodified."
)

_DIM_FIELDS = ("n_channels", "n_times", "n_classes", "sfreq")
STRUCTURAL_FIELDS = ("d_model", "patch_size", "k")

# Source Config._generate_model_architecture. 58 is the only add-on (2C/3C have 58 ch;
# source only shipped 32/64). Pooling rule is still consecutive-index average.
SCALE_TABLE: Dict[int, Tuple[List[int], List[int]]] = {
    32: ([32, 8, 2], [30, 15, 8]),
    64: ([64, 16, 4], [40, 20, 10]),
    58: ([58, 29, 1], [40, 20, 10]),
}


def scale_spec(n_channels: int) -> Tuple[List[int], List[int]]:
    if int(n_channels) not in SCALE_TABLE:
        raise ValueError(
            f"adahgnn: n_channels={n_channels} not in {sorted(SCALE_TABLE)} "
            "(source Config supports 32/64; 58 is the WBCIC extension)"
        )
    a, b = SCALE_TABLE[int(n_channels)]
    return list(a), list(b)


def consecutive_mappings(scale_node_nums: Sequence[int]) -> List[List[List[int]]]:
    """Source ``Config._generate_mappings``."""
    mappings: List[List[List[int]]] = []
    for src, dst in zip(scale_node_nums, scale_node_nums[1:]):
        if src % dst != 0:
            raise ValueError(f"adahgnn: cannot pool {src} -> {dst}")
        ratio = src // dst
        mappings.append([list(range(j, j + ratio)) for j in range(0, src, ratio)])
    return mappings


@dataclass
class AdaHGNNConfig:
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0
    d_model: int = 128
    patch_size: int = 8
    k: int = 5


def _source_config(cfg: AdaHGNNConfig) -> SimpleNamespace:
    scales, hyper_num = scale_spec(cfg.n_channels)
    if int(cfg.n_times) % int(cfg.patch_size) != 0:
        raise ValueError(
            f"adahgnn: n_times={cfg.n_times} not divisible by patch_size={cfg.patch_size}"
        )
    return SimpleNamespace(
        num_channels=int(cfg.n_channels),
        time_sample_num=int(cfg.n_times),
        patch_size=int(cfg.patch_size),
        d_model=int(cfg.d_model),
        k=int(cfg.k),
        num_classes=int(cfg.n_classes),
        scale_node_nums=scales,
        hyper_num=hyper_num,
    )


class AdaHGNNClassifier(nn.Module):
    def __init__(self, cfg: AdaHGNNConfig):
        super().__init__()
        src = _source_config(cfg)
        self.config = cfg
        self.scale_node_nums = list(src.scale_node_nums)
        self.hyper_num = list(src.hyper_num)
        # Domain head is part of the source class; 2 is a dummy (our trainer has no
        # domain labels). alpha=0 at call time, and domain logits are not in the loss.
        self.model = EEG_AdaMSHyper(
            src, consecutive_mappings(src.scale_node_nums), num_domain_classes=2)
        self.feature_dim = int(src.d_model) * len(src.scale_node_nums)
        self._features: Optional[torch.Tensor] = None
        self.model.classifier.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, _module, inputs):
        self._features = inputs[0]

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        x = squeeze_input(x)
        if x.shape[1] != self.config.n_channels:
            raise ValueError(
                f"adahgnn was built for {self.config.n_channels} channels, got {x.shape[1]}"
            )
        if x.shape[-1] != self.config.n_times:
            raise ValueError(
                f"adahgnn was built for n_times={self.config.n_times}, got {x.shape[-1]}"
            )
        logits, _, _ = self.model(x, alpha=0.0)
        feats = self._features
        if feats is None:
            raise RuntimeError("adahgnn: classifier hook did not capture features")
        return {"logits": logits, "features": feats, "confidence": None}

    def describe(self) -> Dict[str, object]:
        cfg = self.config
        return {
            "name": "adahgnn",
            "source": f"{SOURCE_PATH} @ {SOURCE_DATE} (_official/models.py::EEG_AdaMSHyper unmodified)",
            "paper": PAPER_NOTE,
            "n_params": count_params(self),
            "n_trainable_params": count_params(self, trainable_only=True),
            "feature_dim": self.feature_dim,
            "d_model": int(cfg.d_model),
            "n_patches": int(cfg.n_times) // int(cfg.patch_size),
            "patch_size": int(cfg.patch_size),
            "scale_node_nums": list(self.scale_node_nums),
            "hyper_num": list(self.hyper_num),
            "k": int(cfg.k),
            "pretrained_weights": None,
            "official_class": "EEG_AdaMSHyper",
            "dann_head_present": True,
            "dann_in_loss": False,
        }


def build(*, n_channels: int, n_times: int, n_classes: int, sfreq: float = 250.0,
          params: Optional[Dict] = None) -> nn.Module:
    """``arch: official`` (default; the unmodified lab class) or ``arch: mi`` (AdaHGNN-MI,
    ``mi_model.py``; selected by the ``adahgnn@mi`` / ``adahgnn@mi_mix`` variants)."""
    p = dict(params or {})
    for k in _DIM_FIELDS:
        p.pop(k, None)
    arch = str(p.pop("arch", "official") or "official").lower()
    if arch == "mi":
        from .mi_model import MI_FIELDS, build_mi

        p.pop("d_model", None)                       # official-only default from the yaml
        p = warn_unknown_params(MI_FIELDS, p, "adahgnn@mi")
        return build_mi(n_channels=int(n_channels), n_times=int(n_times),
                        n_classes=int(n_classes), sfreq=float(sfreq), params=p)
    if arch != "official":
        raise ValueError(f"adahgnn: unknown arch '{arch}' (official | mi)")
    p.pop("channel_set", None)                       # per_dataset key used only by the MI variant
    p = warn_unknown_params(STRUCTURAL_FIELDS, p, "adahgnn")
    cfg = AdaHGNNConfig(n_channels=int(n_channels), n_times=int(n_times),
                        n_classes=int(n_classes), sfreq=float(sfreq), **p)
    return AdaHGNNClassifier(cfg)


__all__ = ["AdaHGNNConfig", "AdaHGNNClassifier", "build", "STRUCTURAL_FIELDS", "SCALE_TABLE"]
