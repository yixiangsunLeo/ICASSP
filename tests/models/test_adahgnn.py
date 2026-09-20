"""AdaHGNN: (a) the vendored EEG_AdaMSHyper is byte-identical to the lab source and wraps into the
project contract; (b) AdaHGNN-MI (D-040) builds for all three cohorts, learns its structure, matches
dhg's aggregation, partitions the montage hemisphere-aware, and its mixup arm is label-preserving;
(c) the bl_*_adahgnn experiment configs are the main-run protocol with only the roster changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from code.models.external.adahgnn.adapter import consecutive_mappings, scale_spec
from code.models.external.adahgnn.mi_model import (
    REGIONS, AdaHGNNMI, AdaptiveIncidence, aggregation_from_incidence, anatomical_pooling)
from code.models.model_specs import resolve_model_params, safe_tag
from code.models.registry import MODEL_NAMES, build_model
from code.utils.io import _json_default, _sanitize_for_json

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "code" / "models" / "external" / "adahgnn" / "_official"
LAB_SOURCE = Path("/share/home/yuan/LY/DeepHypergraph/project_V4")
EXP = ROOT / "code" / "configs" / "experiments"

COHORTS = {
    "58ch_2cls": (58, 2, "wbci_shu_2c"),
    "58ch_3cls": (58, 3, "wbci_shu_3c"),
    "32ch_2cls": (32, 2, "shu"),
}

OFFICIAL_SHA256 = {
    "models.py": "468468087dead1088d37dc73249feaed44b0f072abe50cd6db4d119919b0b7e1",
    "config.py": "0bb0decd9b0032e9d054fb82edba7a42d25486ea194af8efc734923772adfa94",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(name: str, key: str):
    n_ch, n_cls, dataset = COHORTS[key]
    torch.manual_seed(0)
    params = resolve_model_params(name, dataset=dataset)
    return build_model(name, n_channels=n_ch, n_times=1000, n_classes=n_cls, sfreq=250, params=params)


@pytest.fixture(scope="module")
def mi_models():
    return {k: _build("adahgnn@mi", k) for k in COHORTS}


# --------------------------------------------------------------------------- #
# (a) the official class
# --------------------------------------------------------------------------- #
def test_official_files_byte_identical_to_lab_source():
    assert not (ROOT / "code" / "models" / "external" / "adahgnn" / "network.py").exists()
    for name, digest in OFFICIAL_SHA256.items():
        vendored = OFFICIAL / name
        assert vendored.is_file(), name
        assert _sha256(vendored) == digest
        src = LAB_SOURCE / name
        if src.is_file():                      # lab tree is present on this cluster only
            assert vendored.read_bytes() == src.read_bytes()
    text = (OFFICIAL / "models.py").read_text(encoding="utf-8")
    assert "class EEG_AdaMSHyper" in text and "import dhg" in text and "class GradientReversalLayer" in text


def test_official_wrapper_32ch():
    pytest.importorskip("dhg")
    from code.models.external.adahgnn._official.models import EEG_AdaMSHyper
    m = _build("adahgnn", "32ch_2cls")
    assert type(m.model) is EEG_AdaMSHyper
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 32, 1000))
    assert out["logits"].shape == (2, 2) and torch.isfinite(out["logits"]).all()
    assert not getattr(m, "uses_custom_loss", False)
    assert scale_spec(32) == ([32, 8, 2], [30, 15, 8]) and scale_spec(58) == ([58, 29, 1], [40, 20, 10])
    assert consecutive_mappings([32, 8, 2])[0][0] == list(range(4))


def test_registered_lazy_and_aliases():
    assert "adahgnn" in MODEL_NAMES
    src = (ROOT / "code" / "models" / "registry.py").read_text(encoding="utf-8")
    assert "external.adahgnn" not in src
    assert safe_tag("adahgnn@mi") == "adahgnn-mi" and safe_tag("adahgnn@mi_mix") == "adahgnn-mi_mix"
    with pytest.raises(ValueError):
        resolve_model_params("adahgnn@nope", dataset="shu")


# --------------------------------------------------------------------------- #
# (b) AdaHGNN-MI
# --------------------------------------------------------------------------- #
def test_mi_variants_resolve():
    p = resolve_model_params("adahgnn@mi", dataset="wbci_shu_2c")
    assert p["arch"] == "mi" and p["channel_set"] == "wbci_shu" and p["mixup"] is False
    assert (p["n_filters"], list(p["kernel_sizes"]), p["patch_size"], p["k"]) == (32, [125, 63], 16, 5)
    q = resolve_model_params("adahgnn@mi_mix", dataset="shu")
    assert q["channel_set"] == "shu" and q["mixup"] is True and (q["mixup_prob"], q["mixup_alpha"], q["mixup_k"]) == (0.5, 2.0, 3)
    base = {k: v for k, v in p.items() if not k.startswith("mixup")}
    assert base == {k: v for k, v in resolve_model_params("adahgnn@mi_mix", dataset="wbci_shu_2c").items()
                    if not k.startswith("mixup")}


@pytest.mark.parametrize("key", list(COHORTS))
def test_mi_build_forward_backward_shapes(mi_models, key):
    n_ch, n_cls, _ = COHORTS[key]
    m = mi_models[key]
    assert isinstance(m, AdaHGNNMI)
    R2, R3 = m.pool_12.shape[0], m.pool_23.shape[0]
    assert (R2, R3) == ((13, 4) if n_ch == 58 else (14, 4))
    assert m.feature_dim == 32 + R2 * 32 + R3 * 32
    x = torch.randn(4, n_ch, 1000)
    m.train()
    out = m(x)
    assert set(out) == {"logits", "features", "confidence"}
    assert out["logits"].shape == (4, n_cls) and out["features"].shape == (4, m.feature_dim)
    assert out["confidence"] is None and torch.isfinite(out["logits"]).all()
    m.zero_grad()
    out["logits"].sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert dead == [], f"parameters without gradient: {dead}"
    m.zero_grad()
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x.unsqueeze(1))
    assert torch.equal(a["logits"], b["logits"])
    with pytest.raises(ValueError):
        m(torch.randn(2, n_ch + 1, 1000))


def test_mi_structure_is_learned(mi_models):
    """The source's incidence is thresholded (no gradient); ours must carry gradient to the
    embeddings and satisfy the top-k / row-stochastic invariants."""
    m = mi_models["58ch_2cls"]
    m.train(); m.zero_grad()
    m(torch.randn(3, 58, 1000))["logits"].sum().backward()
    for hg in (m.hg1, m.hg2, m.hg3):
        assert hg.node_emb.grad is not None and float(hg.node_emb.grad.abs().sum()) > 0
        assert hg.edge_emb.grad is not None and float(hg.edge_emb.grad.abs().sum()) > 0
    assert m.lap_logit.grad is not None and float(m.lap_logit.grad.abs().sum()) > 0
    assert m.time_logits.grad is not None
    m.zero_grad()
    with torch.no_grad():
        for hg, n_nodes in ((m.hg1, 58), (m.hg2, 13), (m.hg3, 4)):
            A, H = hg()
            assert H.shape[0] == n_nodes
            assert torch.allclose(H.sum(1), torch.ones(n_nodes), atol=1e-5)
            assert int((H > 0).sum(1).max()) <= hg.k
            assert torch.allclose(A.sum(1), torch.ones(n_nodes), atol=1e-5)
            assert (A >= 0).all()
        g = m.gamma()
        assert torch.allclose(g, torch.full_like(g, 0.5))            # init before training
        assert torch.allclose(torch.softmax(m.time_logits, 0), torch.full((m.n_patches,), 1 / m.n_patches))


def test_mi_aggregation_matches_dhg_mean_v2v():
    dhg = pytest.importorskip("dhg")
    torch.manual_seed(1)
    N, E, k, D = 12, 7, 3, 5
    H = torch.zeros(N, E)
    for i in range(N):
        H[i, torch.randperm(E)[:k]] = 1.0
    e_list = [[int(v) for v in torch.nonzero(H[:, e], as_tuple=True)[0]] for e in range(E)]
    e_list = [e for e in e_list if e]
    hg = dhg.Hypergraph(N, e_list)
    X = torch.randn(N, D)
    ours = aggregation_from_incidence(H) @ X
    theirs = hg.v2v(X, aggr="mean")
    assert torch.allclose(ours, theirs, atol=1e-5), (ours - theirs).abs().max()


def test_mi_incidence_init_not_saturated():
    torch.manual_seed(0)
    hg = AdaptiveIncidence(58, 40, 32, 5, 3.0)
    with torch.no_grad():
        H = hg.incidence()
    # every node keeps k soft entries; the source's N(0,1)@128 init collapses to one-hot rows
    assert int((H > 1e-3).sum(1).min()) >= 2
    assert float(H.max()) < 0.999


@pytest.mark.parametrize("channel_set", sorted(REGIONS))
def test_mi_anatomy_is_hemisphere_aware_partition(channel_set):
    from code.models.external._common import canonical_channel_name, dataset_channel_names
    names = dataset_channel_names(channel_set)
    P12, P23, regions, supers = anatomical_pooling(channel_set, names)
    C = len(names)
    assert P12.shape == (len(regions), C) and P23.shape == (4, len(regions))
    assert torch.allclose(P12.sum(1), torch.ones(len(regions)))
    assert torch.equal((P12 > 0).sum(0), torch.ones(C, dtype=torch.long))          # partition
    assert torch.allclose(P23.sum(1), torch.ones(4))
    assert torch.equal((P23 > 0).sum(0), torch.ones(len(regions), dtype=torch.long))
    canon = [canonical_channel_name(n) for n in names]
    region_of = {}
    for r_idx, r in enumerate(regions):
        for c_idx in torch.nonzero(P12[r_idx], as_tuple=True)[0].tolist():
            region_of[canon[c_idx]] = r
    super_of = {regions[r]: supers[s] for s in range(4) for r in torch.nonzero(P23[s], as_tuple=True)[0].tolist()}
    # left/right homologues never share a region (the source's consecutive pooling averaged C3+C4)
    for left, right in (("C3", "C4"), ("FC1", "FC2"), ("CP1", "CP2"), ("F3", "F4"), ("P3", "P4")):
        assert region_of[left] != region_of[right]
        assert region_of[left].endswith("_L") and region_of[right].endswith("_R")
    assert super_of[region_of["C3"]] == "SM_L" and super_of[region_of["C4"]] == "SM_R"
    assert region_of["CZ"] == "MID" and super_of["MID"] == "MID"
    assert supers == ["SM_L", "SM_R", "MID", "REST"]


def test_mi_anatomy_rejects_wrong_montage():
    with pytest.raises(ValueError):
        anatomical_pooling("shu", ["C3", "C4", "CZ"])
    with pytest.raises(ValueError):
        anatomical_pooling("nope", ["C3"])


def test_mi_single_trial_train_batch_does_not_crash(mi_models):
    m = mi_models["32ch_2cls"]
    m.train()
    out = m(torch.randn(1, 32, 1000))
    assert out["logits"].shape == (1, 2) and torch.isfinite(out["logits"]).all()


def test_mi_mixup_arm_is_label_preserving():
    torch.manual_seed(0)
    m = _build("adahgnn@mi_mix", "32ch_2cls")
    assert m.uses_custom_loss is True and m.cfg.mixup is True
    m.train()
    x = torch.randn(16, 32, 1000)
    x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
    y = torch.tensor([0, 1] * 8)
    partners = m._same_class_partners(y, 2)
    assert partners.shape == (2, 16)
    assert torch.equal(y[partners[0]], y) and torch.equal(y[partners[1]], y)
    assert not torch.equal(partners[0], torch.arange(16))          # not self
    with torch.no_grad():
        mixed, frac = m.mix_batch(x, y)
    assert mixed.shape == x.shape and 0.0 <= frac <= 1.0
    assert torch.allclose(mixed.mean(-1), torch.zeros(16, 32), atol=1e-4)
    assert torch.allclose(mixed.std(-1), torch.ones(16, 32), atol=1e-3)
    loss, parts = m.training_step(x, y, epoch=0, subject_ids=None)
    assert torch.isfinite(loss) and loss.requires_grad
    assert set(parts) == {"ce", "mix_frac"} and all(isinstance(v, float) for v in parts.values())
    # prob 0 -> identity
    m.cfg.mixup_prob = 0.0
    with torch.no_grad():
        same, frac0 = m.mix_batch(x, y)
    assert torch.equal(same, x) and frac0 == 0.0
    # the plain variant does not use the hook
    plain = _build("adahgnn@mi", "32ch_2cls")
    assert plain.uses_custom_loss is False


def test_mi_three_optimizer_steps_cpu():
    torch.manual_seed(0)
    m = _build("adahgnn@mi_mix", "32ch_2cls")
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    m.train()
    losses = []
    for _ in range(3):
        x = torch.randn(6, 32, 1000)
        y = torch.randint(0, 2, (6,))
        loss, _ = m.training_step(x, y, epoch=0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    assert all(map(lambda v: v == v, losses))                        # no NaN
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(torch.randn(2, 32, 1000))["logits"]).all()


@pytest.mark.parametrize("key", list(COHORTS))
def test_mi_describe_json_and_size(mi_models, key):
    m = mi_models[key]
    d = m.describe()
    assert d["name"] == "adahgnn-mi" and d["arch"] == "mi" and d["dann"] is False and d["nhc_constraint"] is False
    json.dumps(_sanitize_for_json(d), default=_json_default, allow_nan=False)
    assert 50_000 < d["n_params"] < 200_000, d["n_params"]
    assert d["scale_node_nums"][0] == COHORTS[key][0] and d["scale_node_nums"][2] == 4
    assert not hasattr(m, "default_lr")


# --------------------------------------------------------------------------- #
# (c) experiment configs and plan
# --------------------------------------------------------------------------- #
def test_experiment_configs_match_snn_protocol():
    pairs = {
        "wbci2c": ("bl_wbci2c_adahgnn_zs_lr1e4.yaml", "snn_wbci2c_zs_lr1e4.yaml"),
        "shu": ("bl_shu_adahgnn_zs_lr1e4.yaml", "snn_shu_zs_lr1e4.yaml"),
        "wbci3c": ("bl_wbci3c_adahgnn_zs_lr1e4.yaml", "snn_wbci3c_zs_lr1e4.yaml"),
    }
    protected = ("run_id", "data", "cross_subject", "eval", "checkpoints", "output", "model_params")
    for ds, (bl_name, snn_name) in pairs.items():
        bl = yaml.safe_load((EXP / bl_name).read_text())
        snn = yaml.safe_load((EXP / snn_name).read_text())
        for key in protected:
            assert bl[key] == snn[key], (ds, key)
        tb = {k: v for k, v in bl["train"].items() if k != "micro_batch_per_model"}
        ts = {k: v for k, v in snn["train"].items() if k != "micro_batch_per_model"}
        assert tb == ts, ds
        assert bl["models"] == ["adahgnn@mi", "adahgnn@mi_mix"]
        assert bl["train"]["micro_batch_per_model"] == {}
        t = bl["train"]
        assert (t["optimizer"], t["lr"], t["batch_size"], t["early_stopping_patience"],
                t["monitor"], t["normalization"], t["seeds"]) == (
            "adam", 1e-4, 128, 25, "macro_f1", "per_sample_zscore", [0])
        for m in bl["models"]:
            resolve_model_params(m, dataset=bl["data"]["name"])


def test_plan_file_consistent():
    plan = (ROOT / "scripts" / "slurm" / "baselines_plan_adahgnn.txt").read_text().splitlines()
    lines = [l.split("#")[0].strip() for l in plan]
    lines = [l for l in lines if l]
    assert len(lines) == 3
    seen = set()
    for line in lines:
        specs = [s.strip() for s in line.split(";")[1:]]
        assert len(specs) == 2
        for spec in specs:
            cfg, models, tag = spec.split("|")
            assert (ROOT / cfg).exists(), cfg
            roster = yaml.safe_load((ROOT / cfg).read_text())["models"]
            assert models in roster and tag
            assert (cfg, models) not in seen
            seen.add((cfg, models))
