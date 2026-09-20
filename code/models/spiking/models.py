r"""Spiking counterparts of EEGNeX and EEGNet (direct training, multi-step).

Design rule: **keep the linear maps of the published ANN, replace every non-linearity by a
spiking neuron, and put the neuron *after* the (linear) average pooling** so that every
layer downstream of the input stem receives binary spikes. Nothing is added where the
published model has no activation (EEGNeX deliberately has none between ``conv2`` and the
depthwise spatial conv, nor between the two dilated convs), unless ``spike_all=True``
(the "fully spiking" ablation).

Layer map, S-EEGNeX (``EEGNeX_8_32`` of Chen et al. 2024 -> spiking)::

    ANN:   conv1 BN ELU | conv2 BN | dw BN ELU pool drop | conv3 BN | conv4 BN ELU pool drop | fc
    SNN:   conv1 BN LIF | conv2 BN | dw BN pool LIF drop | conv3 BN | conv4 BN pool  LI  drop | fc
           \_ stem _/    (encoder)              (envelope)                      (read-out)

Layer map, S-EEGNet (``EEGNet`` of Lawhern et al. 2018, arl-eegmodels defaults)::

    ANN:   conv1 BN | dw BN ELU pool4 drop | sep BN ELU pool8 drop | fc
    SNN:   conv1 BN | dw BN pool4 LIF drop | sep BN pool8  LI  drop | fc

Reviewer-driven design decisions (2026-09-18, see docs/PLAN_spiking_eegnex_icassp2027.md §2):

* **Read-out** (``readout="membrane"``, default): the last neuron is a non-spiking leaky
  integrator whose membrane trace feeds the linear classifier (SpyTorch / snnTorch / Yin et
  al. 2021 convention). A binary read-out of 8x63 bits at ~4 % rate would leave ~20 active
  features and, with the ANN's ``max_norm(0.25)`` on the classifier, a logit ceiling of ~1.1.
  ``readout="spike"`` keeps the fully-binary read-out as an ablation.
* **Reset**: hard reset at the encoder neuron (one phase-locked burst per oscillation cycle,
  lowest rate), **soft** (reset-by-subtraction) at deeper spiking layers so their rate stays
  ~linear in the input current (Rueckauer et al. 2017; Han et al. 2020). ``deep_reset``.
* **Heterogeneous tau** (``hplif``) is initialised log-uniformly per channel over a
  *layer-specific* range: a leaky integrator with tau >~ 8 steps averages a zero-mean 10 Hz
  oscillation to ~0 at 250 Hz and never crosses threshold, so the encoder gets ``[1.5, 4]``
  steps (6-16 ms) and the deeper layers ``[1.5, 8]`` steps (their inputs are non-negative
  rates, where longer integration is meaningful).
* **BatchNorm feeding a neuron** starts at ``beta = +0.5`` (and ``gamma = 2`` where an
  average pool sits between the BN and the neuron, undoing the pooling shrinkage), so the
  pre-threshold distribution has a healthy fraction above ``v_th = 1`` at initialisation;
  otherwise deep layers start 2-3.5 sigma below threshold and the fixed Adam-1e-4 /
  patience-25 recipe stops them at chance. ``neuron_bn_init=False`` disables (ablation).
* **Dropout on spikes** is mask-only (:class:`SpikeDropout`); the usual ``1/(1-p)`` rescale
  would give the following BatchNorm a training variance ~2x its test variance.
* **Encoder position** (``encoder``): ``"conv1"`` (default) spikes right after the temporal
  filter bank so ``conv2`` — 95.7 % of EEGNeX's MACs — runs on spikes; ``"spatial"`` keeps
  conv1 -> conv2 -> depthwise analog and spikes only after the first pooling (the ANN-like
  front end, i.e. the hypothesis test for "does 1-bit encoding before spatial filtering cost
  accuracy?", at the price of the energy story).

Input encodings (same weights, same parameter count):

* ``repeat`` (direct encoding, the SNN-literature default): the analog stem is evaluated
  **once** and injected as a constant current for ``T`` steps; logits are the mean of the
  per-step read-outs. Compute after the stem scales with ``T``.
* ``stream`` (temporal-native, this paper's main model): **the EEG sample axis is the spiking
  time axis.** Each spiking layer iterates over its own sample rate (1000 steps @ 250 Hz
  after ``conv1``, 250 after the first pooling, 63 after the second); no step is repeated, so
  the synaptic-operation count is that of a single pass. ``causal=True`` left-pads the convs
  (streamable with zero look-ahead); the default keeps the ANN's centred 'same' padding so
  the only difference to EEGNeX is the neuron (look-ahead <= 32 samples = 128 ms).

Energy accounting (``op_table`` / ``energy_report``): analog-input layers cost one MAC per
weight-input product per pass; binary-input layers cost one AC per **non-zero** input, i.e.
``dense MACs x firing rate`` per pass. ``E_MAC = 4.6 pJ``, ``E_AC = 0.9 pJ`` (45 nm, Horowitz
ISSCC 2014). The models are hybrids: conv1 (and, without ``spike_all``, the depthwise and
conv4 layers) stay analog; ``energy_report`` counts them at MAC cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..paper_baselines.keras_compat import (avgpool_same, constrain_max_norm, glorot_uniform_,
                                            pad_same_time)
from .neurons import MultiStepLIF, SpikeDropout

E_MAC_PJ = 4.6
E_AC_PJ = 0.9

NEURON_KINDS = ("lif", "plif", "hplif")
ENCODINGS = ("repeat", "stream")
READOUTS = ("membrane", "spike")
ENCODERS = ("conv1", "spatial")
RESETS = ("hard", "soft")

#: heterogeneous-tau init ranges (steps), see module docstring
TAU_RANGE_ENCODER = (1.5, 4.0)
TAU_RANGE_DEEP = (1.5, 8.0)


def _neuron_kind_to_learnable(kind: str) -> str:
    kind = (kind or "lif").lower()
    if kind not in NEURON_KINDS:
        raise ValueError(f"neuron must be one of {NEURON_KINDS}, got {kind!r}")
    return {"lif": "none", "plif": "layer", "hplif": "channel"}[kind]


def pad_causal_time(x: torch.Tensor, kernel: int, dilation: int = 1) -> torch.Tensor:
    """Left-pad the time axis so a conv output at ``t`` sees only ``x[..., :t+1]``."""
    return F.pad(x, ((kernel - 1) * dilation, 0, 0, 0))


@dataclass
class OpEntry:
    """One weighted layer in the energy table."""

    name: str
    macs: int                       # per sample, per pass, dense (as if the input were analog)
    steps: int                      # how many passes per trial (T for repeat, 1 for stream)
    spike_input: bool               # binary input -> ACs weighted by the firing rate
    source: Optional[str] = None    # neuron whose output feeds this layer


def _seq_apply(fn: Callable[[torch.Tensor], torch.Tensor], x_seq: torch.Tensor) -> torch.Tensor:
    """Apply a stateless map to ``[T, B, ...]`` by folding ``T`` into the batch."""
    T, B = x_seq.shape[:2]
    y = fn(x_seq.reshape(T * B, *x_seq.shape[2:]))
    return y.reshape(T, B, *y.shape[1:])


def _time_front(x: torch.Tensor) -> torch.Tensor:
    """``[B, F, C, L]`` -> ``[L, B, F, C]`` (sample axis becomes the spiking time axis)."""
    return x.permute(3, 0, 1, 2).contiguous()


def _time_back(s: torch.Tensor) -> torch.Tensor:
    """``[L, B, F, C]`` -> ``[B, F, C, L]``."""
    return s.permute(1, 2, 3, 0).contiguous()


def _init_bn_for_neuron(bn: nn.BatchNorm2d, *, pooled: bool) -> None:
    """BN feeding a neuron: shift the operating point toward the threshold (see docstring)."""
    with torch.no_grad():
        bn.bias.fill_(0.5)
        if pooled:
            bn.weight.fill_(2.0)


class _SpikingCNN(nn.Module):
    """Common machinery for the two spiking models."""

    def __init__(self, *, T: int, neuron: str, tau: float, v_threshold: float,
                 hard_reset: bool, deep_reset: str, detach_reset: bool, alpha: float,
                 encoding: str, causal: bool, readout: str, neuron_bn_init: bool,
                 tau_min: Optional[float], tau_max: Optional[float]):
        super().__init__()
        if int(T) < 1:
            raise ValueError("T must be >= 1")
        encoding = (encoding or "repeat").lower()
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        readout = (readout or "membrane").lower()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        deep_reset = (deep_reset or "soft").lower()
        if deep_reset not in RESETS:
            raise ValueError(f"deep_reset must be one of {RESETS}")
        self.T = int(T)
        self.neuron_kind = (neuron or "lif").lower()
        self._learnable = _neuron_kind_to_learnable(self.neuron_kind)
        self.tau_init = float(tau)
        self.v_threshold = float(v_threshold)
        self.hard_reset = bool(hard_reset)          # encoder neuron
        self.deep_reset = deep_reset                # deeper spiking neurons
        self.detach_reset = bool(detach_reset)
        self.alpha = float(alpha)
        self.encoding = encoding
        self.causal = bool(causal)
        self.readout = readout
        self.neuron_bn_init = bool(neuron_bn_init)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.op_table: List[OpEntry] = []
        self._neurons: Dict[str, MultiStepLIF] = {}

    # ------------------------------------------------------------------ #
    @property
    def is_stream(self) -> bool:
        return self.encoding == "stream"

    def _tau_range(self, n_steps: int, *, encoder: bool) -> Optional[Tuple[float, float]]:
        """Heterogeneous-tau init range (steps) for a layer that runs ``n_steps`` per pass."""
        if self._learnable != "channel":
            return None
        lo, hi = TAU_RANGE_ENCODER if encoder else TAU_RANGE_DEEP
        if self.tau_min is not None:
            lo = float(self.tau_min)
        if self.tau_max is not None:
            hi = float(self.tau_max)
        hi = min(hi, max(float(n_steps), lo))       # never integrate beyond the pass length
        return (lo, max(hi, lo))

    def make_neuron(self, name: str, n_channels: int, n_steps: int, *,
                    role: str) -> MultiStepLIF:
        """``role``: ``encoder`` (hard reset) | ``deep`` (``deep_reset``) | ``readout`` (LI)."""
        spiking = not (role == "readout" and self.readout == "membrane")
        if role == "encoder":
            hard = self.hard_reset
        else:
            hard = self.deep_reset == "hard"
        n = MultiStepLIF(self.tau_init, self.v_threshold, 0.0, learnable=self._learnable,
                         tau_range=self._tau_range(n_steps, encoder=(role == "encoder")),
                         n_channels=n_channels, channel_dim=1, hard_reset=hard,
                         detach_reset=self.detach_reset, alpha=self.alpha, spiking=spiking)
        n.record_rates = True
        self._neurons[name] = n
        return n

    def pad_t(self, x: torch.Tensor, kernel: int, dilation: int = 1) -> torch.Tensor:
        return (pad_causal_time(x, kernel, dilation) if self.causal
                else pad_same_time(x, kernel, dilation))

    def make_dropout(self, p: float, *, on_spikes: bool) -> nn.Module:
        return SpikeDropout(p) if on_spikes else nn.Dropout(p)

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, C, L]`` -> ``[T, B, 1, C, L]`` (repeat) or ``[1, B, 1, C, L]`` (stream)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.encoding == "repeat":
            return x.unsqueeze(0).expand(self.T, *x.shape)
        return x.unsqueeze(0)

    def spike(self, neuron: MultiStepLIF, z: torch.Tensor) -> torch.Tensor:
        """Run a neuron on ``[T, B, F, C, L]``: over ``T`` (repeat) or over ``L`` (stream)."""
        if self.is_stream:
            return _time_back(neuron(_time_front(z[0]))).unsqueeze(0)
        return neuron(z)

    # ------------------------------------------------------------------ #
    # firing statistics / energy / regulariser
    # ------------------------------------------------------------------ #
    def spiking_neurons(self) -> Dict[str, MultiStepLIF]:
        return {k: n for k, n in self._neurons.items() if n.spiking}

    def firing_rates(self) -> Dict[str, float]:
        return {k: n.mean_rate for k, n in self.spiking_neurons().items()}

    def batch_rates(self) -> Dict[str, torch.Tensor]:
        """Differentiable mean rates of the last training forward (for the regulariser)."""
        return {k: n.batch_rate for k, n in self.spiking_neurons().items()
                if n.batch_rate is not None}

    def reset_rate_stats(self) -> None:
        for n in self._neurons.values():
            n.reset_rate_stats()

    def time_constants(self) -> Dict[str, List[float]]:
        return {k: [round(float(v), 3) for v in torch.atleast_1d(n.tau()).tolist()]
                for k, n in self._neurons.items()}

    def energy_report(self, rates: Optional[Dict[str, float]] = None, *,
                      dense: bool = False) -> Dict[str, float]:
        """MAC/AC counts per trial and the 45 nm energy estimate, ANN reference included.

        ``dense=True`` assumes every spiking input fires every step (upper bound, no
        statistics needed). Otherwise ``rates`` (or the recorded eval-time rates) are used;
        a layer without a recorded rate falls back to the dense assumption.
        """
        rates = {} if dense else (rates if rates is not None else self.firing_rates())
        macs = acs = 0.0
        ann_macs = 0.0
        for op in self.op_table:
            ann_macs += op.macs
            if op.spike_input:
                r = rates.get(op.source or "", float("nan"))
                if r != r:
                    r = 1.0
                acs += op.macs * op.steps * r
            else:
                macs += op.macs * op.steps
        e_snn = macs * E_MAC_PJ + acs * E_AC_PJ
        e_ann = ann_macs * E_MAC_PJ
        return {
            "snn_macs": float(macs), "snn_acs": float(acs), "snn_sops": float(macs + acs),
            "snn_energy_uJ": float(e_snn * 1e-6), "ann_macs": float(ann_macs),
            "ann_energy_uJ": float(e_ann * 1e-6),
            "energy_ratio_ann_over_snn": float(e_ann / e_snn) if e_snn > 0 else float("nan"),
            "dense_assumption": bool(dense),
        }

    @staticmethod
    def _conv_macs(conv: nn.Conv2d, out_shape) -> int:
        """Dense MACs of one conv call for a **single** sample given its output shape."""
        cout, hout, wout = out_shape[-3], out_shape[-2], out_shape[-1]
        kh, kw = conv.kernel_size
        return int(cout * hout * wout * (conv.in_channels // conv.groups) * kh * kw)

    def _infer_feature_dim(self, n_features: int, n_timesteps: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feat = self.forward_features(torch.zeros(1, 1, n_features, n_timesteps))
        self.reset_rate_stats()
        if was_training:
            self.train()
        return int(feat.shape[-1])

    def _passes(self) -> Tuple[int, int]:
        """(passes of the analog stem, passes of every later layer) per trial."""
        return (1, self.T) if self.encoding == "repeat" else (1, 1)


# --------------------------------------------------------------------------- #
# S-EEGNeX
# --------------------------------------------------------------------------- #
class SpikingEEGNeX(_SpikingCNN):
    """Spiking ``EEGNeX_8_32``. Same weights/shapes as ``EEGNeXOfficial`` (+ neuron params)."""

    def __init__(self, n_outputs: int, n_features: int, n_timesteps: int, *,
                 T: int = 4, neuron: str = "lif", tau: float = 2.0, v_threshold: float = 1.0,
                 hard_reset: bool = True, deep_reset: str = "soft", detach_reset: bool = True,
                 alpha: float = 2.0, encoding: str = "repeat", causal: bool = False,
                 readout: str = "membrane", encoder: str = "conv1", neuron_bn_init: bool = True,
                 dropout: float = 0.5, spike_all: bool = False,
                 tau_min: Optional[float] = None, tau_max: Optional[float] = None):
        super().__init__(T=T, neuron=neuron, tau=tau, v_threshold=v_threshold,
                         hard_reset=hard_reset, deep_reset=deep_reset, detach_reset=detach_reset,
                         alpha=alpha, encoding=encoding, causal=causal, readout=readout,
                         neuron_bn_init=neuron_bn_init, tau_min=tau_min, tau_max=tau_max)
        encoder = (encoder or "conv1").lower()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}")
        self.encoder = encoder
        self.spike_all = bool(spike_all)
        if self.spike_all and self.encoder == "spatial":
            raise ValueError("spike_all is incompatible with encoder='spatial'")
        self.n_features, self.n_timesteps = int(n_features), int(n_timesteps)
        L = self.n_timesteps
        p1, p2 = -(-L // 4), -(-(-(-L // 4)) // 4)
        s1, s2, s3 = (L, p1, p2) if self.is_stream else (self.T, self.T, self.T)

        self.conv1 = glorot_uniform_(nn.Conv2d(1, 8, (1, 64), bias=False))
        self.bn1 = nn.BatchNorm2d(8)
        self.lif1 = (self.make_neuron("lif1", 8, s1, role="encoder")
                     if self.encoder == "conv1" else None)

        self.conv2 = glorot_uniform_(nn.Conv2d(8, 32, (1, 64), bias=False))
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = self.make_neuron("lif2", 32, s1, role="deep") if self.spike_all else None

        self.depthwise = constrain_max_norm(
            glorot_uniform_(nn.Conv2d(32, 64, (n_features, 1), groups=32, bias=False)), 1.0)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = self.make_neuron("lif3", 64, s2,
                                     role=("encoder" if self.encoder == "spatial" else "deep"))
        self.drop1 = self.make_dropout(dropout, on_spikes=True)

        self.conv3 = glorot_uniform_(nn.Conv2d(64, 32, (1, 16), dilation=(1, 2), bias=False))
        self.bn4 = nn.BatchNorm2d(32)
        self.lif4 = self.make_neuron("lif4", 32, s2, role="deep") if self.spike_all else None
        self.conv4 = glorot_uniform_(nn.Conv2d(32, 8, (1, 16), dilation=(1, 4), bias=False))
        self.bn5 = nn.BatchNorm2d(8)
        self.lif5 = self.make_neuron("lif5", 8, s3, role="readout")
        self.drop2 = self.make_dropout(dropout, on_spikes=self.lif5.spiking)

        if self.neuron_bn_init:
            if self.lif1 is not None:
                _init_bn_for_neuron(self.bn1, pooled=False)
            if self.lif2 is not None:
                _init_bn_for_neuron(self.bn2, pooled=False)
            _init_bn_for_neuron(self.bn3, pooled=True)
            if self.lif4 is not None:
                _init_bn_for_neuron(self.bn4, pooled=False)
            if self.lif5.spiking:
                _init_bn_for_neuron(self.bn5, pooled=True)

        self.feature_dim = self._infer_feature_dim(n_features, n_timesteps)
        self.classifier = constrain_max_norm(
            glorot_uniform_(nn.Linear(self.feature_dim, n_outputs)), 0.25)
        self._build_op_table(n_features, n_timesteps)

    # ------------------------------------------------------------------ #
    def _stem(self, x_seq: torch.Tensor) -> torch.Tensor:
        """conv1+BN1 on ``[T,B,1,C,L]``; evaluated once under ``repeat`` (constant current)."""
        f = lambda z: self.bn1(self.conv1(self.pad_t(z, 64)))  # noqa: E731
        if self.encoding == "repeat":
            cur = f(x_seq[0])
            return cur.unsqueeze(0).expand(self.T, *cur.shape)
        return _seq_apply(f, x_seq)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``[T, B, feature_dim]`` per-pass features (``T = 1`` for stream)."""
        x_seq = self.to_sequence(x)
        z = self._stem(x_seq)
        if self.lif1 is not None:
            z = self.spike(self.lif1, z)                                      # 1-bit encoder
        z = _seq_apply(lambda a: self.bn2(self.conv2(self.pad_t(a, 64))), z)
        if self.lif2 is not None:
            z = self.spike(self.lif2, z)
        z = _seq_apply(lambda a: avgpool_same(self.bn3(self.depthwise(a)), 4), z)
        s = self.drop1(self.spike(self.lif3, z))
        z = _seq_apply(lambda a: self.bn4(self.conv3(self.pad_t(a, 16, dilation=2))), s)
        if self.lif4 is not None:
            z = self.spike(self.lif4, z)
        z = _seq_apply(lambda a: avgpool_same(self.bn5(self.conv4(self.pad_t(a, 16, dilation=4))), 4), z)
        s = self.drop2(self.spike(self.lif5, z))                              # spikes or LI trace
        return s.flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)                # [T, B, D]
        return self.classifier(feats).mean(dim=0)       # rate read-out (identity for stream)

    # ------------------------------------------------------------------ #
    def _build_op_table(self, C: int, L: int) -> None:
        stem_passes, passes = self._passes()
        pooled1 = -(-L // 4)
        pooled2 = -(-pooled1 // 4)
        m = self._conv_macs
        enc_spikes = self.lif1 is not None
        self.op_table = [
            OpEntry("conv1", m(self.conv1, (8, C, L)), stem_passes, False, None),
            OpEntry("conv2", m(self.conv2, (32, C, L)), passes if enc_spikes else stem_passes,
                    enc_spikes, "lif1" if enc_spikes else None),
            OpEntry("depthwise", m(self.depthwise, (64, 1, L)),
                    passes if self.spike_all else (passes if enc_spikes else stem_passes),
                    self.spike_all, "lif2" if self.spike_all else None),
            OpEntry("conv3", m(self.conv3, (32, 1, pooled1)), passes, True, "lif3"),
            OpEntry("conv4", m(self.conv4, (8, 1, pooled1)), passes, self.spike_all,
                    "lif4" if self.spike_all else None),
            OpEntry("fc", int(self.classifier.in_features * self.classifier.out_features),
                    passes, self.lif5.spiking, "lif5" if self.lif5.spiking else None),
        ]
        assert 8 * pooled2 == self.feature_dim, (pooled2, self.feature_dim)


# --------------------------------------------------------------------------- #
# S-EEGNet
# --------------------------------------------------------------------------- #
class SpikingEEGNet(_SpikingCNN):
    """Spiking EEGNet-8,2 (arl-eegmodels defaults: F1=8, D=2, F2=16, kernLength=64).

    EEGNet has no non-linearity before its depthwise spatial conv, so the faithful port
    spikes only after the first pooling: the stem (conv1 + depthwise, ~99 % of EEGNet's MACs)
    stays analog and the model saves essentially no energy — which is the point of
    reporting it next to S-EEGNeX.
    """

    def __init__(self, n_outputs: int, n_features: int, n_timesteps: int, *,
                 T: int = 4, neuron: str = "lif", tau: float = 2.0, v_threshold: float = 1.0,
                 hard_reset: bool = True, deep_reset: str = "soft", detach_reset: bool = True,
                 alpha: float = 2.0, encoding: str = "repeat", causal: bool = False,
                 readout: str = "membrane", neuron_bn_init: bool = True, dropout: float = 0.5,
                 F1: int = 8, D: int = 2, F2: int = 16, kernLength: int = 64,
                 tau_min: Optional[float] = None, tau_max: Optional[float] = None):
        super().__init__(T=T, neuron=neuron, tau=tau, v_threshold=v_threshold,
                         hard_reset=hard_reset, deep_reset=deep_reset, detach_reset=detach_reset,
                         alpha=alpha, encoding=encoding, causal=causal, readout=readout,
                         neuron_bn_init=neuron_bn_init, tau_min=tau_min, tau_max=tau_max)
        self.F1, self.D, self.F2, self.kernLength = int(F1), int(D), int(F2), int(kernLength)
        L = int(n_timesteps)
        p1, p2 = L // 4, (L // 4) // 8                    # floor pooling (EEGNet-official)
        s1, s2 = (p1, p2) if self.is_stream else (self.T, self.T)

        self.conv1 = glorot_uniform_(nn.Conv2d(1, F1, (1, kernLength), bias=False))
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = constrain_max_norm(
            glorot_uniform_(nn.Conv2d(F1, F1 * D, (n_features, 1), groups=F1, bias=False)), 1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.lif1 = self.make_neuron("lif1", F1 * D, s1, role="encoder")
        self.drop1 = self.make_dropout(dropout, on_spikes=True)
        self.sep_dw = glorot_uniform_(nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False))
        self.sep_pw = glorot_uniform_(nn.Conv2d(F1 * D, F2, (1, 1), bias=False))
        self.bn3 = nn.BatchNorm2d(F2)
        self.lif2 = self.make_neuron("lif2", F2, s2, role="readout")
        self.drop2 = self.make_dropout(dropout, on_spikes=self.lif2.spiking)
        if self.neuron_bn_init:
            _init_bn_for_neuron(self.bn2, pooled=True)
            if self.lif2.spiking:
                _init_bn_for_neuron(self.bn3, pooled=True)
        self.feature_dim = self._infer_feature_dim(n_features, n_timesteps)
        self.classifier = constrain_max_norm(
            glorot_uniform_(nn.Linear(self.feature_dim, n_outputs)), 0.25)
        self._build_op_table(n_features, n_timesteps)

    def _stem(self, x_seq: torch.Tensor) -> torch.Tensor:
        def f(z):
            z = self.bn1(self.conv1(self.pad_t(z, self.kernLength)))
            return F.avg_pool2d(self.bn2(self.depthwise(z)), (1, 4))
        if self.encoding == "repeat":
            cur = f(x_seq[0])
            return cur.unsqueeze(0).expand(self.T, *cur.shape)
        return _seq_apply(f, x_seq)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = self.to_sequence(x)
        s = self.drop1(self.spike(self.lif1, self._stem(x_seq)))
        z = _seq_apply(lambda a: F.avg_pool2d(self.bn3(self.sep_pw(self.sep_dw(self.pad_t(a, 16)))), (1, 8)), s)
        s = self.drop2(self.spike(self.lif2, z))
        return s.flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)).mean(dim=0)

    def _build_op_table(self, C: int, L: int) -> None:
        stem_passes, passes = self._passes()
        pooled1 = L // 4
        m = self._conv_macs
        self.op_table = [
            OpEntry("conv1", m(self.conv1, (self.F1, C, L)), stem_passes, False, None),
            OpEntry("depthwise", m(self.depthwise, (self.F1 * self.D, 1, L)), stem_passes, False, None),
            OpEntry("sep_dw", m(self.sep_dw, (self.F1 * self.D, 1, pooled1)), passes, True, "lif1"),
            # pointwise conv consumes the analog output of sep_dw (no neuron in between)
            OpEntry("sep_pw", m(self.sep_pw, (self.F2, 1, pooled1)), passes, False, None),
            OpEntry("fc", int(self.classifier.in_features * self.classifier.out_features),
                    passes, self.lif2.spiking, "lif2" if self.lif2.spiking else None),
        ]


__all__ = ["SpikingEEGNeX", "SpikingEEGNet", "OpEntry", "E_MAC_PJ", "E_AC_PJ", "NEURON_KINDS",
           "ENCODINGS", "READOUTS", "ENCODERS", "RESETS", "pad_causal_time",
           "TAU_RANGE_ENCODER", "TAU_RANGE_DEEP"]
