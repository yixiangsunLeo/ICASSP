"""Unit tests for the spiking models (CPU, small tensors).

Run:  python -m pytest tests/models/test_spiking.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from code.experiments.cross_subject_protocols import euclidean_align, split_normalization
from code.models.model_specs import resolve_model_params, variant_names
from code.models.registry import build_model
from code.models.spiking.neurons import MultiStepLIF, _lif_loop_python, _lif_loop_script

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# neurons
# --------------------------------------------------------------------------- #
def test_script_and_python_loops_agree_forward_and_backward():
    x = torch.randn(50, 3, 4, requires_grad=True)
    k = torch.full((3, 4), 0.5)
    args = (1.0, 0.0, True, True, 2.0)
    s1 = _lif_loop_python(x, k, *args)
    s2 = _lif_loop_script(x, k, *args)
    assert torch.equal(s1, s2)
    g1 = torch.autograd.grad(s1.sum(), x)[0]
    x2 = x.detach().clone().requires_grad_(True)
    g2 = torch.autograd.grad(_lif_loop_script(x2, k, *args).sum(), x2)[0]
    assert torch.allclose(g1, g2, atol=1e-6)


def test_lif_constant_current_rate_is_monotone():
    n = MultiStepLIF(tau=2.0)
    for lo, hi in [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]:
        r = [n(torch.full((200, 1, 1), c)).mean().item() for c in (lo, hi)]
        assert r[1] >= r[0]
    # sub-threshold constant input (steady state h -> x < v_th) never fires
    assert n(torch.full((200, 1, 1), 0.9)).sum().item() == 0.0


def test_lif_as_level_crossing_encoder_on_oscillation():
    """A 10 Hz oscillation @250 Hz should yield ~1 spike per cycle for tau=2, v_th=1."""
    t = torch.arange(1000) / 250.0
    x = 2.0 * torch.sin(2 * math.pi * 10 * t)                # amplitude 2 (BN units)
    n = MultiStepLIF(tau=2.0)
    s = n(x.view(1000, 1, 1))
    cycles = 10 * 4.0
    spikes = s.sum().item()
    assert 0.6 * cycles <= spikes <= 3.0 * cycles, spikes    # rate encodes amplitude, per cycle
    # doubling the amplitude increases the count
    s2 = n((2 * x).view(1000, 1, 1))
    assert s2.sum().item() > spikes


def test_heterogeneous_tau_init_range():
    n = MultiStepLIF(tau=2.0, learnable="channel", n_channels=8, tau_range=(1.5, 16.0))
    taus = n.tau()
    assert taus.shape == (8,)
    assert abs(taus[0].item() - 1.5) < 1e-4 and abs(taus[-1].item() - 16.0) < 1e-3
    assert torch.all(taus[1:] > taus[:-1])


# --------------------------------------------------------------------------- #
# normalization / alignment
# --------------------------------------------------------------------------- #
def test_split_normalization_grammar():
    assert split_normalization("per_sample_zscore") == ("none", "per_sample_zscore", None)
    assert split_normalization("ea_session") == ("session", "none", None)
    assert split_normalization("ea_session+per_sample_zscore") == ("session", "per_sample_zscore", None)
    assert split_normalization("ea_subject+per_sample_global") == ("subject", "per_sample_global", None)
    assert split_normalization("ea_session_mu+per_sample_global") == ("session", "per_sample_global", (8.0, 30.0))
    with pytest.raises(ValueError):
        split_normalization("foo+bar")
    with pytest.raises(ValueError):
        split_normalization("per_sample_zscore_mu")


def test_euclidean_align_band_limited_reference():
    rng = np.random.default_rng(1)
    t = np.arange(500) / 250.0
    # 6 sources: a strong 2 Hz drift + a 12 Hz mu rhythm on two of them, mixed by A
    src = rng.normal(size=(30, 6, 500)) * 0.3
    src[:, 0] += 5.0 * np.sin(2 * np.pi * 2 * t)          # big low-frequency power
    src[:, 1] += 1.0 * np.sin(2 * np.pi * 12 * t)         # mu-band source
    A = np.eye(6) + 0.3 * rng.normal(size=(6, 6))          # well-conditioned mixing
    X = np.einsum("cd,ndt->nct", A, src).astype(np.float32)
    Xb = euclidean_align(X, shrinkage=0.0)                # broadband reference
    Xm = euclidean_align(X, shrinkage=0.0, ref_band=(8.0, 30.0), sfreq=250.0)
    assert Xb.shape == Xm.shape == X.shape and np.isfinite(Xm).all()
    # the mu-referenced whitening equalises the 8-30 Hz band, not the broadband signal
    from scipy.signal import filtfilt, firwin
    taps = firwin(101, [8.0, 30.0], pass_zero=False, fs=250.0)
    Rm = _mean_cov(filtfilt(taps, [1.0], Xm.astype(np.float64), axis=-1))
    assert np.allclose(Rm, np.eye(6), atol=0.08)
    assert not np.allclose(_mean_cov(Xm), np.eye(6), atol=0.08)   # broadband deliberately not I


def _mean_cov(X):
    Xc = X - X.mean(2, keepdims=True)
    return np.einsum("nct,ndt->cd", Xc, Xc) / (X.shape[0] * (X.shape[2] - 1))


def test_euclidean_align_whitens_mean_covariance():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    X = np.einsum("cd,ndt->nct", A, rng.normal(size=(40, 6, 300))).astype(np.float32)
    # no shrinkage: exact whitening, mean covariance == identity
    R0 = _mean_cov(euclidean_align(X, shrinkage=0.0))
    assert np.allclose(R0, np.eye(6), atol=0.02)
    # 10% shrinkage: eigenvalues lambda/(0.9 lambda + 0.1 mu) in (0, 1/0.9]; strong directions ~1,
    # weak (noise) directions are damped instead of amplified
    R1 = _mean_cov(euclidean_align(X, shrinkage=0.1))
    w = np.linalg.eigvalsh(R1)
    assert w.max() <= 1.0 / 0.9 + 0.02 and w.min() > 0
    assert abs(w.max() - 1.0) < 0.12
    # alignment removes subject-specific mixing: two differently mixed sources align to ~same cov
    B = rng.normal(size=(6, 6))
    Y = np.einsum("cd,ndt->nct", B, rng.normal(size=(40, 6, 300))).astype(np.float32)
    RX, RY = _mean_cov(euclidean_align(X, 0.0)), _mean_cov(euclidean_align(Y, 0.0))
    assert np.abs(RX - RY).max() < 0.05


# --------------------------------------------------------------------------- #
# models via the registry (every declared variant builds, runs, has gradients)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", ["seegnex", "seegnet"])
def test_all_variants_build_forward_backward(base):
    C, L = 8, 200                                              # small: 200 samples -> 50 -> 13
    y = torch.tensor([0, 1, 0, 1])
    ann = build_model("eegnex" if base == "seegnex" else "eegnet_official",
                      n_channels=C, n_times=L, n_classes=2, params={})
    ann_params = sum(p.numel() for p in ann.parameters())
    for v in [""] + variant_names(base):
        name = base if not v else f"{base}@{v}"
        if "chunk" in v and L % 4:
            continue
        params = resolve_model_params(name, dataset="shu")
        m = build_model(name, n_channels=C, n_times=L, n_classes=2, params=params)
        d = m.describe()
        # spiking params = ANN params + neuron taus only
        n_tau = sum(p.numel() for n, p in m.named_parameters() if n.endswith(".w"))
        assert d["n_params"] - n_tau == ann_params, (name, d["n_params"], ann_params)
        x = torch.randn(4, C, L)
        m.train()
        out = m(x)
        assert out["logits"].shape == (4, 2)
        assert out["features"].shape[0] == 4
        # custom-loss hook: CE + homeostatic rate penalty (disabled -> plain CE)
        loss, parts = m.training_step(x, y)
        assert "ce" in parts and torch.isfinite(loss)
        if m.rate_reg_lambda > 0:
            assert "rate_reg" in parts and any(k.startswith("r_") for k in parts)
        loss.backward()
        gsum = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        assert gsum > 0, name
        m.eval()
        with torch.no_grad():
            m.pop_spike_stats()
            m(x)
            st = m.pop_spike_stats()
        for k, r in st["firing_rates"].items():
            assert 0.0 <= r <= 1.0, (name, k, r)
        e = st["energy"]
        assert e["ann_macs"] > 0 and e["snn_energy_uJ"] > 0
        # stream encoding: single pass -> SNN ops never exceed the ANN's dense MACs
        if d["encoding"] == "stream":
            assert e["snn_macs"] + e["snn_acs"] <= e["ann_macs"] + 1e-6, name


def test_stream_causal_variant_is_causal():
    """With causal=True, changing the tail of the input must not change earlier features."""
    C, L = 6, 200
    params = resolve_model_params("seegnex@s_hplif_causal", dataset="shu")
    m = build_model("seegnex@s_hplif_causal", n_channels=C, n_times=L, n_classes=2, params=params)
    m.eval()
    x = torch.randn(1, C, L)
    x2 = x.clone()
    x2[..., 120:] += 10.0                                       # perturb only the tail
    with torch.no_grad():
        f1 = m.model.forward_features(x)[0, 0].view(8, -1)     # [8, pooled2]
        f2 = m.model.forward_features(x2)[0, 0].view(8, -1)
    # pooled index j covers samples [16 j, 16 j + 16); indices fully before sample 120 are 0..6
    assert torch.equal(f1[:, :7], f2[:, :7])


def test_readout_and_init_choices():
    C, L = 8, 200
    p = resolve_model_params("seegnex@s_hplif", dataset="shu")
    m = build_model("seegnex@s_hplif", n_channels=C, n_times=L, n_classes=2, params=p).model
    # LI membrane read-out: last neuron does not spike, features are analog
    assert not m.lif5.spiking and m.lif3.spiking and m.lif1.spiking
    assert m.lif1.hard_reset and not m.lif3.hard_reset            # encoder hard, deep soft
    m.train()                                                       # batch-stat BN: live network
    with torch.no_grad():
        f = m.forward_features(torch.randn(16, C, L))
    assert f.abs().sum() > 0 and not torch.all((f == 0) | (f == 1))  # analog LI trace
    # per-layer heterogeneous tau ranges: encoder [1.5, 4], deep [1.5, 8]
    t1, t3 = m.lif1.tau(), m.lif3.tau()
    assert abs(t1.min().item() - 1.5) < 1e-3 and abs(t1.max().item() - 4.0) < 1e-2
    assert abs(t3.min().item() - 1.5) < 1e-3 and abs(t3.max().item() - 8.0) < 1e-2
    # BN feeding a neuron starts shifted toward threshold; pooled ones also scaled
    assert torch.allclose(m.bn1.bias, torch.full_like(m.bn1.bias, 0.5))
    assert torch.allclose(m.bn3.weight, torch.full_like(m.bn3.weight, 2.0))
    # spike read-out ablation is binary; analog-front-end ablation has no encoder neuron
    ps = resolve_model_params("seegnex@s_hplif_spk", dataset="shu")
    ms = build_model("seegnex@s_hplif_spk", n_channels=C, n_times=L, n_classes=2, params=ps).model
    ms.train()
    with torch.no_grad():
        fs = ms.forward_features(torch.randn(16, C, L))
    assert torch.all((fs == 0) | (fs == 1))
    pa = resolve_model_params("seegnex@s_hplif_afe", dataset="shu")
    ma = build_model("seegnex@s_hplif_afe", n_channels=C, n_times=L, n_classes=2, params=pa).model
    assert ma.lif1 is None and ma.lif3.hard_reset
    e = ma.energy_report(dense=True)
    assert e["snn_macs"] > 0.9 * e["ann_macs"]                     # conv2 analog -> no saving


def test_spike_dropout_mask_only():
    from code.models.spiking.neurons import SpikeDropout
    d = SpikeDropout(0.5).train()
    x = torch.ones(10000)
    y = d(x)
    assert set(y.unique().tolist()) <= {0.0, 1.0}                   # never rescaled to 2
    assert 0.4 < y.mean().item() < 0.6
    assert torch.equal(d.eval()(x), x)


def test_stream_energy_below_repeat():
    C, L = 8, 200
    outs = {}
    for v in ("hplif", "s_hplif"):
        params = resolve_model_params(f"seegnex@{v}", dataset="shu")
        m = build_model(f"seegnex@{v}", n_channels=C, n_times=L, n_classes=2, params=params)
        m.eval()
        with torch.no_grad():
            m.pop_spike_stats(); m(torch.randn(8, C, L)); outs[v] = m.pop_spike_stats()["energy"]
    assert outs["s_hplif"]["snn_acs"] <= outs["hplif"]["snn_acs"] + 1e-6
