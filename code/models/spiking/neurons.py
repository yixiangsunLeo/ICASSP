"""Multi-step spiking neurons with surrogate gradients (self-contained, no spikingjelly).

Why not spikingjelly: the cluster env pins torch 2.7.1+cu118 and every extra dependency is
one more thing that can drift; a LIF neuron is ~40 lines. The maths follows SpikingJelly's
``MultiStepLIFNode`` / ``MultiStepParametricLIFNode`` (Fang et al., ICCV 2021) exactly, so
numbers are comparable to the literature that uses that library.

Neuron dynamics (``decay_input=True`` form, per time step ``t``)::

    H[t] = V[t-1] + (X[t] - (V[t-1] - v_reset)) / tau          # charge
    S[t] = Theta(H[t] - v_threshold)                             # fire (Heaviside)
    V[t] = H[t] * (1 - S[t]) + v_reset * S[t]                    # hard reset
         = H[t] - S[t] * v_threshold                             # soft reset

``1/tau`` is a fixed constant for ``LIF`` and ``sigmoid(w)`` with a learnable ``w`` for
``PLIF`` (one ``w`` per layer) / ``HPLIF`` (one ``w`` per channel: *heterogeneous* time
constants, initialised log-uniformly over ``[tau_min, tau_max]`` so the layer starts as a
bank of integration windows). The Heaviside step is replaced in the backward pass by the
ATan surrogate ``g'(x) = alpha / 2 / (1 + (pi/2 * alpha * x)^2)``.

``detach_reset=True`` stops gradients from flowing through the reset path, which is the
recommended setting for direct training (Zenke & Vogels 2021).

Time axis
---------
The neuron iterates over ``dim 0`` of its input. Two layouts are supported by the callers:

* ``[T, B, ...]``  — simulation steps (direct / repeat encoding, ``T`` small).
* ``[L, B, ...]``  — the **EEG sample axis** moved to the front (temporal-native /
  streaming encoding, ``L`` = 1000 / 250 / 63 depending on the layer). Same code path.

Speed
-----
A step is ~10 elementwise kernels; for ``L = 1000`` the Python loop is launch-bound. The
loop is therefore also provided as a TorchScript function (``_lif_loop_script``) in which the
fuser collapses the step body into 1-2 kernels. The surrogate gradient is expressed with the
straight-through identity ``spike + (g(h) - g(h).detach())`` where ``g`` is the *integral* of
the ATan surrogate, so autograd (also inside TorchScript) yields exactly the ATan gradient
while the forward value stays the Heaviside spike. Set ``MultiStepLIF.use_script = False`` to
fall back to the plain Python loop (identical numerics).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Surrogate spike function
# --------------------------------------------------------------------------- #
class ATanSurrogate(torch.autograd.Function):
    """Heaviside forward, ATan-derivative backward (SpikingJelly ``surrogate.ATan``)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sg = alpha / 2.0 / (1.0 + (math.pi / 2.0 * alpha * x).pow(2))
        return grad_output * sg, None


def heaviside_atan(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return ATanSurrogate.apply(x, alpha)


@torch.jit.script
def _lif_loop_script(x_seq: torch.Tensor, k: torch.Tensor, v_th: float, v_reset: float,
                     hard_reset: bool, detach_reset: bool, alpha: float) -> torch.Tensor:
    """LIF over ``dim 0``; ``k`` = 1/tau broadcastable to one step ``x_seq[0]``.

    Straight-through ATan surrogate: forward = Heaviside, backward = ATan derivative.
    """
    T = x_seq.shape[0]
    v = torch.full_like(x_seq[0], v_reset)
    out = torch.empty_like(x_seq)
    half_pi_alpha = 1.5707963267948966 * alpha       # pi/2 * alpha
    inv_pi = 0.3183098861837907                      # 1/pi
    for t in range(T):
        h = v + (x_seq[t] - (v - v_reset)) * k
        u = h - v_th
        hard = (u >= 0.0).to(h.dtype)
        # integral of the ATan surrogate derivative: g(u) = atan(pi/2 alpha u)/pi + 1/2
        soft = torch.atan(half_pi_alpha * u) * inv_pi + 0.5
        s = hard + (soft - soft.detach())
        s_reset = s.detach() if detach_reset else s
        if hard_reset:
            v = h * (1.0 - s_reset) + v_reset * s_reset
        else:
            v = h - s_reset * v_th
        out[t] = s
    return out


def _lif_loop_python(x_seq: torch.Tensor, k: torch.Tensor, v_th: float, v_reset: float,
                     hard_reset: bool, detach_reset: bool, alpha: float) -> torch.Tensor:
    T = x_seq.shape[0]
    v = torch.full_like(x_seq[0], v_reset)
    out = []
    for t in range(T):
        h = v + (x_seq[t] - (v - v_reset)) * k
        s = heaviside_atan(h - v_th, alpha)
        s_reset = s.detach() if detach_reset else s
        if hard_reset:
            v = h * (1.0 - s_reset) + v_reset * s_reset
        else:
            v = h - s_reset * v_th
        out.append(s)
    return torch.stack(out, dim=0)


@torch.jit.script
def _li_loop_script(x_seq: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Non-spiking leaky integrator over ``dim 0``: ``V[t] = V[t-1] + (X[t] - V[t-1]) k``.

    Used as the read-out layer ("membrane readout", cf. SpyTorch / snnTorch / Yin et al.
    2021): the final layer integrates the last spike train and its analog membrane trace is
    what the linear classifier reads. No threshold, no reset.
    """
    T = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])
    out = torch.empty_like(x_seq)
    for t in range(T):
        v = v + (x_seq[t] - v) * k
        out[t] = v
    return out


class SpikeDropout(nn.Module):
    """Dropout for binary spike trains **without** the ``1/(1-p)`` rescale.

    Standard dropout scales the kept units by ``1/(1-p)``, so a downstream layer sees
    ``{0, 2}`` in training and ``{0, 1}`` at test. Means match, but the variance does not
    (``2r - r^2`` vs ``r - r^2``), so the following BatchNorm's running variance is ~2x the
    test-time variance and every neuron downstream sees its input current shrink by ~0.7x at
    test — which a threshold turns into a several-fold firing-rate change. Masking only
    (Bernoulli keep, no rescale) keeps the test-time statistics of a spike train intact.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = torch.rand_like(x) >= self.p
        return x * keep.to(x.dtype)

    def extra_repr(self) -> str:
        return f"p={self.p}, rescale=False"


# --------------------------------------------------------------------------- #
# Neuron module
# --------------------------------------------------------------------------- #
class MultiStepLIF(nn.Module):
    """LIF / PLIF / heterogeneous-PLIF neuron operating on a ``[T, B, ...]`` sequence.

    Parameters
    ----------
    tau : float
        Initial membrane time constant (in time steps). ``tau=2`` is the SpikingJelly default.
    learnable : {"none", "layer", "channel"}
        ``none`` -> plain LIF. ``layer`` -> PLIF (one learnable ``tau``). ``channel`` -> one
        learnable ``tau`` per feature channel (``channel_dim`` of a step ``x[t]``), i.e. a bank
        of heterogeneous integration windows, the spiking analogue of EEGNeX's dilated
        multi-scale temporal convolutions.
    tau_range : (float, float) | None
        For ``learnable="channel"``: initial taus are log-uniformly spaced over this range
        (deterministic, so runs are reproducible). ``None`` -> all channels start at ``tau``.
    n_channels : int
        Required when ``learnable="channel"``.
    channel_dim : int
        Which axis of one step ``x[t]`` (i.e. of ``[B, ...]``) is the channel axis; default 1.
    hard_reset : bool
        ``True`` -> reset to ``v_reset``; ``False`` -> subtract the threshold (soft reset).
    detach_reset : bool
        Detach the spike in the reset term (recommended).
    alpha : float
        ATan surrogate width.
    """

    #: class-level switch so tests can compare both loop implementations
    use_script: bool = True

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        *,
        learnable: str = "none",
        tau_range: Optional[tuple] = None,
        n_channels: Optional[int] = None,
        channel_dim: int = 1,
        hard_reset: bool = True,
        detach_reset: bool = True,
        alpha: float = 2.0,
        spiking: bool = True,
    ):
        super().__init__()
        if tau <= 1.0:
            raise ValueError(f"tau must be > 1 (got {tau})")
        learnable = (learnable or "none").lower()
        if learnable not in ("none", "layer", "channel"):
            raise ValueError("learnable must be one of none|layer|channel")
        self.learnable = learnable
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.hard_reset = bool(hard_reset)
        self.detach_reset = bool(detach_reset)
        self.alpha = float(alpha)
        self.channel_dim = int(channel_dim)
        #: ``False`` -> non-spiking leaky integrator (LI read-out): emits its membrane trace
        self.spiking = bool(spiking)
        #: mean spike rate of the most recent *training* forward, differentiable through the
        #: surrogate; consumed by the homeostatic rate regulariser (None for LI / eval)
        self.batch_rate: Optional[torch.Tensor] = None
        # PLIF parameterisation: 1/tau = sigmoid(w), w0 = -log(tau - 1)  (Fang et al. 2021)
        init_w = -math.log(tau - 1.0)
        if learnable == "none":
            self.register_buffer("w", torch.tensor(init_w))
        elif learnable == "layer":
            self.w = nn.Parameter(torch.tensor(init_w))
        else:
            if not n_channels:
                raise ValueError("learnable='channel' requires n_channels")
            n = int(n_channels)
            if tau_range is not None:
                lo, hi = float(tau_range[0]), float(tau_range[1])
                if not (1.0 < lo <= hi):
                    raise ValueError(f"tau_range must satisfy 1 < lo <= hi (got {tau_range})")
                taus = torch.logspace(math.log10(lo), math.log10(hi), n) if n > 1 else torch.tensor([lo])
                self.w = nn.Parameter(-torch.log(taus - 1.0))
            else:
                self.w = nn.Parameter(torch.full((n,), init_w))
        # firing-rate bookkeeping (eval-time statistics for the energy estimate)
        self.record_rates = False
        self.register_buffer("_rate_sum", torch.zeros(()), persistent=False)
        self.register_buffer("_rate_n", torch.zeros(()), persistent=False)

    # ------------------------------------------------------------------ #
    def decay(self, x_t: torch.Tensor) -> torch.Tensor:
        """``1/tau`` broadcast to one step's shape (``[B, ...]``)."""
        k = torch.sigmoid(self.w)
        if self.learnable == "channel":
            shape = [1] * x_t.dim()
            shape[self.channel_dim] = -1
            k = k.view(*shape)
        return k

    def tau(self) -> torch.Tensor:
        """Current time constants (``1/sigmoid(w)``), for ``describe()``."""
        return 1.0 / torch.sigmoid(self.w.detach())

    def reset_rate_stats(self) -> None:
        self._rate_sum.zero_()
        self._rate_n.zero_()

    @property
    def mean_rate(self) -> float:
        n = float(self._rate_n)
        return float(self._rate_sum) / n if n > 0 else float("nan")

    # ------------------------------------------------------------------ #
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """``x_seq``: ``[T, B, ...]`` input currents -> ``[T, B, ...]`` spikes in {0, 1}
        (or the analog membrane trace when ``spiking=False``)."""
        k = self.decay(x_seq[0]).to(x_seq.dtype)
        # expand k to a full step shape so TorchScript sees a plain tensor broadcast
        if k.dim() == 0:
            k = k.expand_as(x_seq[0])
        x_seq = x_seq.contiguous()
        k = k.contiguous()
        if not self.spiking:
            if x_seq.is_cuda and self.use_script:
                return _li_loop_script(x_seq, k)
            v = torch.zeros_like(x_seq[0])
            out = []
            for t in range(x_seq.shape[0]):
                v = v + (x_seq[t] - v) * k
                out.append(v)
            return torch.stack(out, dim=0)
        loop = _lif_loop_script if (self.use_script and x_seq.is_cuda) else _lif_loop_python
        spikes = loop(x_seq, k, self.v_threshold, self.v_reset,
                      self.hard_reset, self.detach_reset, self.alpha)
        if self.training:
            self.batch_rate = spikes.mean()            # differentiable (surrogate path)
        elif self.record_rates:
            with torch.no_grad():
                self._rate_sum += spikes.mean() * spikes.shape[1]
                self._rate_n += spikes.shape[1]
        return spikes

    def extra_repr(self) -> str:
        kind = "LIF" if self.spiking else "LI(readout)"
        return (f"{kind}, learnable={self.learnable}, v_th={self.v_threshold}, "
                f"hard_reset={self.hard_reset}, detach_reset={self.detach_reset}, "
                f"alpha={self.alpha}")


__all__ = ["ATanSurrogate", "heaviside_atan", "MultiStepLIF", "SpikeDropout",
           "_lif_loop_script", "_lif_loop_python", "_li_loop_script"]



