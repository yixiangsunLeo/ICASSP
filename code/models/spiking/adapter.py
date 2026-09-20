"""Project-contract wrapper for the spiking models (``seegnex`` / ``seegnet``).

``forward(x)`` takes ``[B, C, T]`` (or ``[B, 1, C, T]``) and returns
``{"logits", "features", "confidence"}`` like every other model in the registry, so the
cross-subject runner, trainer and summarizer need no changes.

Extra, spiking-only surface (all optional for the framework):

* ``describe()`` — parameters, ``T``, neuron kind, encoding, read-out, current time constants
  and the energy table (dense upper bound; measured numbers come with ``pop_spike_stats``).
* ``pop_spike_stats()`` — mean firing rate per spiking layer over every eval-mode forward
  since the last call + the resulting energy estimate; resets the counters. The
  cross-subject runner drains it right before and reads it right after each held-out
  subject's prediction pass, so per-subject firing statistics land in ``result.json``.
* ``training_step`` (``uses_custom_loss=True`` when ``rate_reg_lambda > 0``) — cross-entropy
  plus a homeostatic penalty ``lambda * sum_l (r_l - r*)^2`` on the per-layer mean firing rate
  of the batch (Zenke & Vogels 2021 use activity bounds the same way). It only acts when a
  layer drifts toward silence or saturation; at ``r_l ~ r*`` it is ~0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (ENCODERS, ENCODINGS, NEURON_KINDS, READOUTS, RESETS, SpikingEEGNeX,
                     SpikingEEGNet)

SPIKING_MODEL_NAMES: List[str] = ["seegnex", "seegnet"]

#: Direct-trained SNNs are conventionally optimised with Adam at 1e-3 (Fang et al. 2021).
#: Used only when the experiment config leaves ``train.lr`` empty; our benchmark recipe
#: pins the value for every model and that setting always wins.
DEFAULT_LR = {"seegnex": 1e-3, "seegnet": 1e-3}


@dataclass
class SpikingConfig:
    # data dims (injected by the experiment layer)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    variant: str = "seegnex"

    # --- spiking dynamics ---
    T: int = 4                      # simulation steps (repeat encoding only)
    neuron: str = "lif"             # lif | plif | hplif
    tau: float = 2.0                # initial membrane time constant (steps) for lif / plif
    tau_min: Optional[float] = None # hplif: override of the per-layer init range (steps)
    tau_max: Optional[float] = None
    v_threshold: float = 1.0
    hard_reset: bool = True         # encoder neuron
    deep_reset: str = "soft"        # deeper spiking neurons: soft | hard
    detach_reset: bool = True
    surrogate_alpha: float = 2.0
    encoding: str = "repeat"        # repeat (direct) | stream (temporal-native)
    causal: bool = False            # stream: left-padded convs (zero look-ahead)
    readout: str = "membrane"       # membrane (LI trace -> FC) | spike (binary -> FC)
    neuron_bn_init: bool = True     # BN feeding a neuron starts at beta=0.5 (gamma=2 if pooled)
    dropout: float = 0.5            # mask-only on spikes, standard on the analog read-out

    # --- homeostatic rate regulariser (training only) ---
    rate_reg_lambda: float = 1e-3   # 0 disables
    rate_target: float = 0.10

    # --- S-EEGNeX only ---
    encoder: str = "conv1"          # conv1 (spike after the filter bank) | spatial (ablation)
    spike_all: bool = False         # neurons also where EEGNeX has no activation

    # --- S-EEGNet only (arl-eegmodels defaults) ---
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernLength: int = 64


class SpikingClassifier(nn.Module):
    def __init__(self, config: Optional[SpikingConfig] = None):
        super().__init__()
        cfg = config or SpikingConfig()
        self.config = cfg
        variant = (cfg.variant or "").lower()
        if variant not in SPIKING_MODEL_NAMES:
            raise ValueError(f"unknown spiking model '{cfg.variant}' (use {SPIKING_MODEL_NAMES})")
        for val, allowed, key in ((cfg.neuron, NEURON_KINDS, "neuron"),
                                  (cfg.encoding, ENCODINGS, "encoding"),
                                  (cfg.readout, READOUTS, "readout"),
                                  (cfg.deep_reset, RESETS, "deep_reset"),
                                  (cfg.encoder, ENCODERS, "encoder")):
            if str(val).lower() not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, got {val!r}")
        self.variant = variant
        common = dict(T=cfg.T, neuron=cfg.neuron, tau=cfg.tau, v_threshold=cfg.v_threshold,
                      hard_reset=cfg.hard_reset, deep_reset=cfg.deep_reset,
                      detach_reset=cfg.detach_reset, alpha=cfg.surrogate_alpha,
                      encoding=cfg.encoding, causal=cfg.causal, readout=cfg.readout,
                      neuron_bn_init=cfg.neuron_bn_init, dropout=cfg.dropout,
                      tau_min=cfg.tau_min, tau_max=cfg.tau_max)
        if variant == "seegnex":
            self.model = SpikingEEGNeX(cfg.n_classes, cfg.n_channels, cfg.n_times,
                                       encoder=cfg.encoder, spike_all=cfg.spike_all, **common)
        else:
            self.model = SpikingEEGNet(cfg.n_classes, cfg.n_channels, cfg.n_times,
                                       F1=cfg.F1, D=cfg.D, F2=cfg.F2,
                                       kernLength=cfg.kernLength, **common)
        self.feature_dim = int(self.model.feature_dim)
        self.rate_reg_lambda = float(cfg.rate_reg_lambda)
        self.rate_target = float(cfg.rate_target)
        self.uses_custom_loss = self.rate_reg_lambda > 0.0

    # ------------------------------------------------------------------ #
    def default_lr(self) -> float:
        return DEFAULT_LR[self.variant]

    def describe(self) -> Dict[str, object]:
        m = self.model
        return {
            "name": self.variant,
            "source": "spiking port of the official ANN (code/models/spiking/models.py)",
            "n_params": sum(p.numel() for p in self.parameters()),
            "feature_dim": self.feature_dim,
            "T": m.T, "neuron": m.neuron_kind, "encoding": m.encoding, "causal": m.causal,
            "readout": m.readout, "encoder": getattr(m, "encoder", None),
            "v_threshold": m.v_threshold, "tau_init": m.tau_init,
            "hard_reset_encoder": m.hard_reset, "deep_reset": m.deep_reset,
            "neuron_bn_init": m.neuron_bn_init,
            "rate_reg": {"lambda": self.rate_reg_lambda, "target": self.rate_target},
            "spike_all": bool(getattr(m, "spike_all", False)),
            "spiking_layers": sorted(m.spiking_neurons()),
            "time_constants": m.time_constants(),
            "op_table": [{"layer": o.name, "macs_per_pass": o.macs, "passes": o.steps,
                          "spike_input": o.spike_input, "source": o.source}
                         for o in m.op_table],
            "energy_dense_upper_bound": m.energy_report(dense=True),
        }

    def pop_spike_stats(self) -> Dict[str, object]:
        rates = self.model.firing_rates()
        out = {"firing_rates": rates, "energy": self.model.energy_report(rates),
               "time_constants": self.model.time_constants()}
        self.model.reset_rate_stats()
        return out

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        feats = self.model.forward_features(x)              # [T, B, D]
        logits = self.model.classifier(feats).mean(dim=0)   # rate read-out
        # time-averaged features are the penultimate representation
        return {"logits": logits, "features": feats.mean(dim=0), "confidence": None}

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0,
                      **_ignored) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CE + homeostatic firing-rate penalty (see module docstring)."""
        ce = F.cross_entropy(self.forward(x)["logits"], y)
        rates = self.model.batch_rates()
        if self.rate_reg_lambda > 0.0 and rates:
            reg = sum((r - self.rate_target) ** 2 for r in rates.values())
            loss = ce + self.rate_reg_lambda * reg
            parts = {"ce": float(ce.detach().item()), "rate_reg": float(reg.detach().item())}
            parts.update({f"r_{k}": float(v.detach().item()) for k, v in rates.items()})
            return loss, parts
        return ce, {"ce": float(ce.detach().item())}


def build_spiking(variant: str, *, n_channels: int, n_times: int, n_classes: int,
                  sfreq: float = 250.0, params: Optional[Dict] = None) -> SpikingClassifier:
    cfg = SpikingConfig(n_channels=n_channels, n_times=n_times, n_classes=n_classes,
                        sfreq=float(sfreq), variant=variant, **(params or {}))
    return SpikingClassifier(cfg)


__all__ = ["SPIKING_MODEL_NAMES", "SpikingConfig", "SpikingClassifier", "build_spiking",
           "DEFAULT_LR"]
