"""TCFormer (Altaheri, Karray & Karimi, *Scientific Reports* 15:32959, 2025) behind the project contract.

Official code, run unmodified: ``_official/models/tcformer.py::TCFormerModule`` from
https://github.com/altaheri/TCFormer @ 701a361640708ef1cc07cf22660d47edc6d9bf3c (MIT). This file
only (i) fixes the hyperparameters to the paper configuration recorded in the upstream
``configs/tcformer.yaml`` and (ii) wraps the network so that it returns
``{"logits", "features", "confidence"}`` — no architecture, initialisation or forward maths is
touched. Provenance, hashes and the three edited import lines: ``PROVENANCE.md``.

Input contract: ``x`` float32 ``[B, C, T]`` (or ``[B, 1, C, T]``) with C = ``n_channels`` and
T = ``n_times`` (1000 samples @ 250 Hz, per-trial z-scored by the shared data pipeline). TCFormer
consumes raw multichannel EEG at any sampling rate/length (its front-end is convolutional and
pooled), so **no resampling, cropping or channel selection is performed**: the tensor goes into the
upstream module exactly as every other entry receives it. With T = 1000 the two average-poolings
(8, then 7) leave ``(1000 // 8) // 7 = 17`` tokens for the Transformer/TCN (upstream BCI IV-2a uses
T = 1125 -> 20 tokens); nothing in the network depends on T being 1125.

``features`` is the tensor entering the final classification layer
(``TCNHead.classifier.linear``, a grouped 1x1 ``Conv1dWithConstraint``): the last TCN time step,
``[B, d_group * (n_groups + 1)] = [B, 64]``. It is captured with a forward pre-hook, as
``code/models/paper_baselines/adapter.py`` does for EEG-Deformer, so the upstream forward is not
re-implemented.

Training: plain cross-entropy through the shared recipe (Adam 1e-4, batch 128, no scheduler, no
augmentation). Upstream's own optimiser settings (Adam ``lr=9e-4``, ``beta_1=0.5``,
``weight_decay=1e-3``, warm-up + cosine, inter-trial augmentation) are deliberately **not**
reproduced — the protocol applies one recipe to every model — and this class defines neither
``default_lr`` nor the ``uses_custom_loss`` / ``training_step`` hooks. Weight initialisation is
upstream's own: ``MultiKernelConvBlock.__init__`` calls ``glorot_weight_zero_bias(self)`` and
``_GQAttention.__init__`` calls ``_xavier_zero_bias(self)``; the Lightning module and the training
script apply no further initialisation, so nothing is added here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .._common import count_params, squeeze_input, warn_unknown_params
from ._official.models.tcformer import TCFormerModule

UPSTREAM_URL = "https://github.com/altaheri/TCFormer"
UPSTREAM_COMMIT = "701a361640708ef1cc07cf22660d47edc6d9bf3c"
UPSTREAM_COMMIT_DATE = "2026-08-15"
PAPER = ("H. Altaheri, F. Karray, A.-H. Karimi, 'Temporal convolutional transformer for EEG based "
         "motor imagery decoding', Scientific Reports 15:32959, 2025, doi:10.1038/s41598-025-16219-7")

#: Data-dimension fields that the experiment layer injects; never read from ``params``.
_DIM_FIELDS = ("n_channels", "n_times", "n_classes", "sfreq")


@dataclass
class TCFormerConfig:
    """Structural hyperparameters of ``TCFormerModule``.

    Defaults = the paper configuration shipped as ``model_kwargs`` in the upstream
    ``configs/tcformer.yaml`` (vendored: ``_official/configs/tcformer.yaml``). Note that the
    Python class defaults of ``TCFormerModule`` differ (F1=16, kernels (16, 32, 64), q_heads=8,
    kv_heads=4, dropout_conv=0.3); the yaml — what ``train_pipeline.py`` actually passes — wins.
    ``drop_path_max`` is not a yaml key: it keeps the class default 0.25, which the yaml comment
    ("0.4 and path drop = 0.25") confirms.
    """

    # data dims (injected by the experiment layer, never hard-coded)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    # --- multi-kernel convolutional front-end (MK-CNN) ---
    F1: int = 32
    temp_kernel_lengths: Tuple[int, ...] = (20, 32, 64)
    d_group: int = 16
    D: int = 2
    pool_length_1: int = 8
    pool_length_2: int = 7
    dropout_conv: float = 0.4
    use_group_attn: bool = True

    # --- Transformer encoder (grouped-query attention + RoPE, pre-LN, GEGLU-free MLP x2) ---
    q_heads: int = 4
    kv_heads: int = 2
    trans_depth: int = 5
    trans_dropout: float = 0.4
    drop_path_max: float = 0.25

    # --- TCN head ---
    tcn_depth: int = 2
    kernel_length_tcn: int = 4
    dropout_tcn: float = 0.3

    def __post_init__(self) -> None:
        # yaml delivers a list; the upstream code only needs a sequence, but a tuple keeps the
        # config hashable/JSON-stable and identical across construction paths.
        if not isinstance(self.temp_kernel_lengths, (list, tuple)) or not self.temp_kernel_lengths:
            raise ValueError("temp_kernel_lengths must be a non-empty list/tuple of ints")
        self.temp_kernel_lengths = tuple(int(k) for k in self.temp_kernel_lengths)
        if self.q_heads % self.kv_heads != 0:
            raise ValueError("q_heads must be a multiple of kv_heads (upstream assertion)")
        d_model = self.d_group * len(self.temp_kernel_lengths)
        if d_model % self.q_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by q_heads={self.q_heads}")

    @property
    def n_groups(self) -> int:
        return len(self.temp_kernel_lengths)

    @property
    def d_model(self) -> int:
        return self.d_group * self.n_groups

    @property
    def feature_dim(self) -> int:
        """Width of the tensor entering the classifier: TCN filters = d_group * (n_groups + 1)."""
        return self.d_group * (self.n_groups + 1)

    def n_tokens(self, n_times: Optional[int] = None) -> int:
        """Transformer sequence length after the two average-poolings (floor, as AvgPool2d)."""
        t = int(self.n_times if n_times is None else n_times)
        return (t // self.pool_length_1) // self.pool_length_2

    def structural_kwargs(self) -> Dict[str, object]:
        """Exactly the keyword arguments handed to ``TCFormerModule`` (data dims excluded)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if f.name not in _DIM_FIELDS}


STRUCTURAL_FIELDS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(TCFormerConfig) if f.name not in _DIM_FIELDS)


class TCFormerClassifier(nn.Module):
    """``TCFormerModule`` with the project's dict contract; see the module docstring."""

    def __init__(self, config: Optional[TCFormerConfig] = None):
        super().__init__()
        cfg = config or TCFormerConfig()
        if cfg.n_tokens() < 1:
            raise ValueError(
                f"n_times={cfg.n_times} is too short: pooling {cfg.pool_length_1} then "
                f"{cfg.pool_length_2} leaves no token")
        self.config = cfg
        self.model = TCFormerModule(n_channels=int(cfg.n_channels), n_classes=int(cfg.n_classes),
                                    **cfg.structural_kwargs())

        head: nn.Conv1d = self.model.tcn_head.classifier.linear
        self.feature_dim = int(head.in_channels)
        assert self.feature_dim == cfg.feature_dim, (self.feature_dim, cfg.feature_dim)
        self._features: Optional[torch.Tensor] = None
        # Upstream returns logits only; capture the head's input without touching the maths.
        head.register_forward_pre_hook(self._capture_features)

    # ------------------------------------------------------------------ hooks / contract
    def _capture_features(self, module: nn.Module, inputs: Sequence[torch.Tensor]) -> None:
        self._features = inputs[0]

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        x = squeeze_input(x)
        if x.shape[1] != self.config.n_channels:
            raise ValueError(
                f"tcformer was built for {self.config.n_channels} channels, got input with "
                f"{x.shape[1]} (shape {tuple(x.shape)})")
        self._features = None
        logits = self.model(x)
        feats = self._features
        self._features = None
        if feats is None:
            raise RuntimeError(
                "tcformer: classifier hook captured nothing; the upstream head structure "
                "changed and the feature hook must be revisited.")
        # [B, d, 1] (last TCN step, kept as a length-1 sequence upstream) -> [B, d]
        features = feats.flatten(1)
        return {"logits": logits, "features": features, "confidence": None}

    # ------------------------------------------------------------------ diagnostics
    def describe(self) -> Dict[str, object]:
        cfg = self.config
        return {
            "name": "tcformer",
            "paper": PAPER,
            "source": f"{UPSTREAM_URL} @ {UPSTREAM_COMMIT} ({UPSTREAM_COMMIT_DATE}), MIT; "
                      "models/tcformer.py::TCFormerModule run unmodified (3 import paths edited, "
                      "see code/models/external/tcformer/PROVENANCE.md)",
            "n_params": count_params(self),
            "n_trainable_params": count_params(self, trainable_only=True),
            "feature_dim": self.feature_dim,
            "pretrained_weights": None,   # upstream releases none; every cell trains from scratch
            "n_channels": int(cfg.n_channels), "n_times": int(cfg.n_times),
            "n_classes": int(cfg.n_classes), "sfreq": float(cfg.sfreq),
            "input_format": "raw [B, C, T] @ 250 Hz, no resampling / cropping / channel selection",
            "n_tokens": cfg.n_tokens(),
            "d_model": cfg.d_model, "n_groups": cfg.n_groups,
            "head_dim": cfg.d_model // cfg.q_heads,
            "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in cfg.structural_kwargs().items()},
            "hyperparameter_source": "upstream configs/tcformer.yaml::model_kwargs "
                                     "(drop_path_max = class default 0.25)",
            "init": "upstream: glorot_weight_zero_bias on the conv block, xavier/zero-bias on "
                    "GQA projections, PyTorch defaults elsewhere; nothing added",
            "training": "shared recipe (plain CE); upstream Adam lr 9e-4 / beta1 0.5 / wd 1e-3 / "
                        "warm-up+cosine / interaug NOT used",
        }


def build(*, n_channels: int, n_times: int, n_classes: int, sfreq: float = 250.0,
          params: Optional[Dict] = None) -> TCFormerClassifier:
    """Entry point used by ``code.models.external.build_external``."""
    p = dict(params or {})
    for k in _DIM_FIELDS + ("variant",):
        p.pop(k, None)
    p = warn_unknown_params(STRUCTURAL_FIELDS, p, "tcformer")
    cfg = TCFormerConfig(n_channels=int(n_channels), n_times=int(n_times),
                         n_classes=int(n_classes), sfreq=float(sfreq), **p)
    return TCFormerClassifier(cfg)


__all__ = ["TCFormerConfig", "TCFormerClassifier", "STRUCTURAL_FIELDS", "build",
           "UPSTREAM_URL", "UPSTREAM_COMMIT", "UPSTREAM_COMMIT_DATE", "PAPER"]
