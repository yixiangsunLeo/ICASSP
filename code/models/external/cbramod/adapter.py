"""CBraMod (Wang et al., ICLR 2025) behind this project's ``{logits, features, confidence}`` contract.

Upstream: https://github.com/wjq-learning/CBraMod, commit b9e96100 (files vendored verbatim under
``_official/``, provenance in ``PROVENANCE.md``). Pretrained weights: ``weights/CBraMod.pth``
(HF ``weighting666/CBraMod`` ``pretrained_weights.pth``), loaded with ``strict=True``.

What this adapter does — and nothing else (protocol ``docs/PROTOCOL_external_baselines.md`` P4/P5):

  1. ``[B, C, 1000] @ 250 Hz`` -> sinc resample to CBraMod's native 200 Hz -> ``[B, C, 800]``
     -> cut into 4 contiguous 1 s patches -> ``[B, C, 4, 200]`` (the backbone's input layout,
     identical to the upstream downstream models ``models/model_for_*.py`` and to the
     EEG-FM-Compass wrapper ``Loader_CBraMod.forward``).
  2. Backbone = upstream ``CBraMod()`` with its default (= pretrained) configuration; the
     pre-training reconstruction head ``proj_out`` is replaced by ``nn.Identity()`` *after* the
     strict load, as every upstream fine-tuning model does.
  3. Read-out = flatten all ``C x 4`` token embeddings (``feature_dim = C * 4 * 200``) and apply
     ``Dropout(0.1) -> Linear(feature_dim, n_classes)`` — the EEG-FM-Compass ``LinearLayers``
     head (``utils/model_layers.py``), i.e. the linear-head convention shared by all foundation
     models in this comparison. (Upstream's own default head for its downstream tasks is a
     3-layer ELU MLP on the same flattened tokens; see PROVENANCE.md.)
  4. Full fine-tuning: every parameter is trainable (upstream ``finetune_trainer.py`` with
     ``frozen=False`` does the same).

CBraMod is channel-agnostic (its positional encoding is a depthwise convolution over the
(channel, patch) grid, and spatial attention runs over whatever channels are present), so no
electrode vocabulary, no channel selection and no ``per_dataset`` block are needed; ``n_channels``
only sizes the read-out. The adapter verifies that with a dry forward at construction.

No filtering, no amplitude scaling, no alignment: the tensor that arrives here is the same
per-trial z-scored trial every other entry of the benchmark receives.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch
import torch.nn as nn

from .._common import (count_params, resample_time, resolve_weights_path, sha256_file,
                       squeeze_input, warn_unknown_params)
from ._official.cbramod import CBraMod

MODEL_NAME = "cbramod"
UPSTREAM_URL = "https://github.com/wjq-learning/CBraMod"
UPSTREAM_COMMIT = "b9e961003214326972c567eff390e75b0287e32a"
PAPER = "J. Wang et al., CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding, ICLR 2025"

#: ``weights/CBraMod.pth`` = HF ``weighting666/CBraMod`` ``pretrained_weights.pth`` (19,775,842 bytes).
PRETRAINED_FILE = "CBraMod.pth"
PRETRAINED_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"

#: Upstream fine-tuning learning rate for the backbone (``finetune_main.py --lr 1e-4``). Only
#: consulted when an experiment config leaves ``train.lr`` empty; the benchmark recipe sets it.
DEFAULT_LR = 1e-4


@dataclass
class CBraModConfig:
    """Structural hyperparameters. Backbone defaults are upstream ``CBraMod.__init__`` defaults,
    which are also the configuration of the released pretrained checkpoint (a different value
    would make ``strict=True`` loading fail, by design)."""

    # data dims (injected by the experiment layer, never hard-coded)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    # --- backbone: upstream CBraMod() defaults ---
    in_dim: int = 200            # samples per patch fed to the patch embedding
    out_dim: int = 200           # width of the (discarded) reconstruction head proj_out
    d_model: int = 200           # token width; criss-cross attention splits it 100 + 100
    dim_feedforward: int = 800
    seq_len: int = 30            # accepted by upstream PatchEmbedding but unused (no learned table)
    n_layer: int = 12
    nhead: int = 8               # 4 heads spatial + 4 heads temporal

    # --- format conversion (protocol P4) ---
    native_sfreq: float = 200.0  # CBraMod is pretrained on 200 Hz, 1 s patches
    patch_len: int = 200         # samples per patch at native_sfreq (= in_dim)

    # --- read-out head (EEG-FM-Compass LinearLayers: Dropout -> Linear) ---
    head_dropout: float = 0.1

    # --- pretrained weights (protocol P5: official checkpoint, strict, full fine-tuning) ---
    weights_file: Optional[str] = PRETRAINED_FILE      # None/"" -> random init (debugging only)
    weights_sha256: Optional[str] = PRETRAINED_SHA256  # None -> skip the integrity check


_DIM_KEYS = ("n_channels", "n_times", "n_classes", "sfreq")
KNOWN_PARAMS = tuple(f.name for f in fields(CBraModConfig) if f.name not in _DIM_KEYS)


class CBraModClassifier(nn.Module):
    """Upstream CBraMod backbone + flatten read-out, project contract."""

    def __init__(self, config: Optional[CBraModConfig] = None):
        super().__init__()
        cfg = config or CBraModConfig()
        self.config = cfg

        # ---- geometry of the format conversion -------------------------------------------
        if cfg.patch_len != cfg.in_dim:
            raise ValueError(
                f"cbramod: patch_len ({cfg.patch_len}) must equal the backbone in_dim "
                f"({cfg.in_dim}); the upstream patch embedding hard-codes a 200-sample patch "
                "(rfft -> 101 bins -> Linear(101, d_model))."
            )
        self.native_n_times = int(round(cfg.n_times * float(cfg.native_sfreq) / float(cfg.sfreq)))
        if self.native_n_times <= 0 or self.native_n_times % cfg.patch_len != 0:
            raise ValueError(
                f"cbramod: {cfg.n_times} samples @ {cfg.sfreq} Hz resample to "
                f"{self.native_n_times} samples @ {cfg.native_sfreq} Hz, which is not a multiple "
                f"of patch_len={cfg.patch_len}."
            )
        self.n_patches = self.native_n_times // cfg.patch_len

        # ---- backbone: upstream code, upstream defaults, official weights ----------------
        self.backbone = CBraMod(
            in_dim=cfg.in_dim, out_dim=cfg.out_dim, d_model=cfg.d_model,
            dim_feedforward=cfg.dim_feedforward, seq_len=cfg.seq_len,
            n_layer=cfg.n_layer, nhead=cfg.nhead,
        )
        self.n_backbone_params_incl_proj_out = count_params(self.backbone)
        self.pretrained: Dict[str, object] = self._load_pretrained()
        # The reconstruction head is part of the checkpoint (hence loaded strictly above) but
        # not of any downstream model: upstream `model_for_*.py` and the Compass loader both
        # replace it with Identity after loading.
        self.backbone.proj_out = nn.Identity()
        # Full fine-tuning, as upstream `finetune_trainer.py` (frozen=False) does for every
        # backbone parameter. This also flips `patch_embedding.mask_encoding` (200 zeros that
        # upstream defines with requires_grad=False); it is only read when a pre-training mask
        # is given, so in fine-tuning it never receives a gradient and stays at zero.
        self.backbone.requires_grad_(True)

        # ---- read-out: verify the flattened token size with a dry forward ---------------
        expected_dim = int(cfg.n_channels) * self.n_patches * int(cfg.d_model)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probe = torch.zeros(1, int(cfg.n_channels), int(cfg.n_times))
            feats = self._backbone_features(probe)
        self.train(was_training)
        if tuple(feats.shape) != (1, expected_dim):
            raise RuntimeError(
                f"cbramod: dry forward produced features {tuple(feats.shape)}, expected "
                f"(1, {expected_dim}) = C({cfg.n_channels}) x patches({self.n_patches}) x "
                f"d_model({cfg.d_model}). Upstream layout changed; revisit the read-out."
            )
        self.feature_dim = expected_dim
        # EEG-FM-Compass `LinearLayers(feature_dim, n_classes, flatten=1, dropout_rate=0.1)`.
        self.head = nn.Sequential(
            nn.Dropout(float(cfg.head_dropout)),
            nn.Linear(self.feature_dim, int(cfg.n_classes)),
        )

    # ------------------------------------------------------------------------------ weights
    def _load_pretrained(self) -> Dict[str, object]:
        cfg = self.config
        if not cfg.weights_file:
            return {"loaded": False, "file": None, "path": None, "sha256": None,
                    "n_tensors": 0, "n_values": 0, "missing_keys": None, "unexpected_keys": None}
        path = resolve_weights_path(str(cfg.weights_file))
        digest = sha256_file(path)
        if cfg.weights_sha256 and digest.lower() != str(cfg.weights_sha256).lower():
            raise ValueError(
                f"cbramod: {path} has sha256 {digest}, expected {cfg.weights_sha256} "
                "(weights/MANIFEST.yaml). Refusing to fine-tune from an unverified checkpoint; "
                "set weights_sha256: null to bypass deliberately."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"cbramod: {path} is not a state_dict (got {type(state).__name__})")
        # strict=True raises on any mismatch; the returned lists are therefore empty, but we
        # record them so describe() carries the fact explicitly.
        result = self.backbone.load_state_dict(state, strict=True)
        return {
            "loaded": True, "file": path.name, "path": str(path), "sha256": digest,
            "n_tensors": len(state), "n_values": int(sum(v.numel() for v in state.values())),
            "missing_keys": len(result.missing_keys),
            "unexpected_keys": len(result.unexpected_keys),
        }

    # ----------------------------------------------------------------------- conversion
    def to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T] @ sfreq`` (or ``[B, 1, C, T]``) -> ``[B, C, n_patches, patch_len] @ native_sfreq``.

        Patch ``k`` holds native samples ``[k * patch_len, (k + 1) * patch_len)``: contiguous,
        non-overlapping 1 s windows, exactly ``x.reshape(b, c, -1, 200)`` of the upstream and
        Compass wrappers, applied after the 250 -> 200 Hz sinc resampling.
        """
        cfg = self.config
        x = squeeze_input(x)
        if x.shape[1] != cfg.n_channels:
            raise ValueError(
                f"cbramod: built for n_channels={cfg.n_channels}, got input with {x.shape[1]} channels"
            )
        x = resample_time(x, float(cfg.sfreq), float(cfg.native_sfreq), n_out=self.native_n_times)
        b, c, _ = x.shape
        return x.reshape(b, c, self.n_patches, cfg.patch_len)

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(self.to_patches(x))          # [B, C, n_patches, d_model]
        return tokens.flatten(1)                            # [B, C * n_patches * d_model]

    # ------------------------------------------------------------------------- contract
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        features = self._backbone_features(x)
        logits = self.head(features)
        return {"logits": logits, "features": features, "confidence": None}

    def default_lr(self) -> float:
        return DEFAULT_LR

    def describe(self) -> Dict[str, object]:
        cfg = self.config
        return {
            "name": MODEL_NAME,
            "paper": PAPER,
            "source": f"{UPSTREAM_URL} @ {UPSTREAM_COMMIT}",
            "n_params": count_params(self),
            "n_trainable_params": count_params(self, trainable_only=True),
            "n_backbone_params": count_params(self.backbone),
            "n_backbone_params_incl_proj_out": int(self.n_backbone_params_incl_proj_out),
            "n_head_params": count_params(self.head),
            "feature_dim": int(self.feature_dim),
            "readout": f"flatten(C x {self.n_patches} tokens x {cfg.d_model}) -> "
                       f"Dropout({cfg.head_dropout}) -> Linear({self.feature_dim}, {cfg.n_classes})",
            "fine_tuning": "full",
            # format conversion
            "input_sfreq": float(cfg.sfreq),
            "input_n_times": int(cfg.n_times),
            "native_sfreq": float(cfg.native_sfreq),
            "native_n_times": int(self.native_n_times),
            "patch_len": int(cfg.patch_len),
            "n_patches": int(self.n_patches),
            "n_tokens": int(cfg.n_channels) * int(self.n_patches),
            # channels: nothing is selected or renamed
            "channel_agnostic": True,
            "channels_kept": int(cfg.n_channels),
            "channels_dropped": 0,
            # backbone configuration actually instantiated
            "backbone": {
                "in_dim": cfg.in_dim, "out_dim": cfg.out_dim, "d_model": cfg.d_model,
                "dim_feedforward": cfg.dim_feedforward, "seq_len": cfg.seq_len,
                "n_layer": cfg.n_layer, "nhead": cfg.nhead,
            },
            # pretrained checkpoint facts
            "pretrained_file": self.pretrained.get("file"),
            "pretrained_sha256": self.pretrained.get("sha256"),
            "pretrained": dict(self.pretrained),
        }


def build(*, n_channels: int, n_times: int, n_classes: int, sfreq: float = 250.0,
          params: Optional[Dict] = None) -> CBraModClassifier:
    """Entry point used by ``code.models.external.build_external`` (and thus the registry)."""
    p = warn_unknown_params(KNOWN_PARAMS, dict(params or {}), MODEL_NAME)
    cfg = CBraModConfig(n_channels=int(n_channels), n_times=int(n_times),
                        n_classes=int(n_classes), sfreq=float(sfreq), **p)
    return CBraModClassifier(cfg)


__all__ = [
    "MODEL_NAME", "UPSTREAM_URL", "UPSTREAM_COMMIT", "PRETRAINED_FILE", "PRETRAINED_SHA256",
    "DEFAULT_LR", "KNOWN_PARAMS", "CBraModConfig", "CBraModClassifier", "build",
]
