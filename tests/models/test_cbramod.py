"""CBraMod external baseline: contract, format conversion, strict weight loading, provenance.

CPU only, small batches. Building the model loads the 19.8 MB checkpoint and runs one dry
forward, so the module-scoped fixtures below build each configuration once.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest
import torch

from code.models.external._common import resample_time, resolve_weights_path, sha256_file
from code.models.external.cbramod import adapter as A
from code.models.model_specs import resolve_model_params
from code.models.registry import build_model

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "code" / "models" / "external" / "cbramod" / "_official"

CONFIGS = {
    # (n_channels, n_classes, data.name) -> expected total parameter count
    "58ch_2cls": (58, 2, "wbci_shu_2c", 4_976_602),
    "58ch_3cls": (58, 3, "wbci_shu_3c", 5_023_003),
    "32ch_2cls": (32, 2, "shu", 4_935_002),
}
BACKBONE_PARAMS = 4_883_800           # upstream CBraMod() with proj_out -> Identity
CHECKPOINT_VALUES = 4_924_000         # backbone + proj_out(200x200 + 200), as released
D_MODEL, N_PATCHES = 200, 4

# sha256 of the UPSTREAM files at commit b9e96100 (see PROVENANCE.md §1)
UPSTREAM_SHA256 = {
    "cbramod.py": "caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14",
    "criss_cross_transformer.py": "90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c",
    "LICENSE": "eb839b706e4d7219842e96d7dc33285844abc794292602deb3498be7b699537c",
}
VENDORED_IMPORT = ("from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder"
                   "  # vendored: was `from models.criss_cross_transformer import ...`")
UPSTREAM_IMPORT = "from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder"


def _weights_present() -> bool:
    return resolve_weights_path(A.PRETRAINED_FILE, required=False) is not None


def _build(key: str):
    n_ch, n_cls, dataset, _ = CONFIGS[key]
    torch.manual_seed(0)
    params = resolve_model_params("cbramod", dataset=dataset)
    return build_model("cbramod", n_channels=n_ch, n_times=1000, n_classes=n_cls, sfreq=250,
                       params=params)


@pytest.fixture(scope="module")
def models():
    if not _weights_present():
        pytest.skip(f"{A.PRETRAINED_FILE} not found in weights/ (see code/models/external/_common.py)")
    return {k: _build(k) for k in CONFIGS}


# --------------------------------------------------------------------------- #
# 1. contract: build through the registry, forward, determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(CONFIGS))
def test_registry_build_forward_and_eval_determinism(models, key):
    n_ch, n_cls, _, _ = CONFIGS[key]
    m = models[key]
    x = torch.randn(2, n_ch, 1000)
    m.train()
    out = m(x)
    assert set(out) == {"logits", "features", "confidence"}
    assert out["logits"].shape == (2, n_cls)
    assert out["features"].shape == (2, n_ch * N_PATCHES * D_MODEL)
    assert out["confidence"] is None
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["features"]).all()

    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a["logits"], b["logits"]), "eval() output must be deterministic"
    assert torch.equal(a["features"], b["features"])
    # the head consumes exactly `features`
    assert torch.allclose(m.head(a["features"]), a["logits"])


def test_accepts_4d_input(models):
    m = models["32ch_2cls"].eval()
    x = torch.randn(2, 32, 1000)
    with torch.no_grad():
        assert torch.equal(m(x)["logits"], m(x.unsqueeze(1))["logits"])


def test_wrong_channel_count_raises(models):
    with pytest.raises(ValueError, match="n_channels"):
        models["32ch_2cls"](torch.randn(2, 58, 1000))


def test_construction_is_deterministic_given_the_seed():
    if not _weights_present():
        pytest.skip("weights absent")
    a, b = _build("32ch_2cls"), _build("32ch_2cls")
    for (ka, pa), (kb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb
        assert torch.equal(pa, pb), ka


# --------------------------------------------------------------------------- #
# 2. parameter counts (exact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(CONFIGS))
def test_parameter_counts(models, key):
    n_ch, n_cls, _, expected_total = CONFIGS[key]
    d = models[key].describe()
    feature_dim = n_ch * N_PATCHES * D_MODEL
    assert d["feature_dim"] == feature_dim
    assert d["n_backbone_params"] == BACKBONE_PARAMS
    assert d["n_backbone_params_incl_proj_out"] == CHECKPOINT_VALUES
    assert d["n_head_params"] == feature_dim * n_cls + n_cls
    assert d["n_params"] == BACKBONE_PARAMS + feature_dim * n_cls + n_cls == expected_total
    assert d["n_trainable_params"] == d["n_params"], "full fine-tuning: everything trainable"


def test_backbone_is_channel_agnostic(models):
    """Same backbone tensors for 58 and 32 channels: only the read-out depends on C."""
    sd58 = models["58ch_2cls"].backbone.state_dict()
    sd32 = models["32ch_2cls"].backbone.state_dict()
    assert sd58.keys() == sd32.keys()
    for k in sd58:
        assert torch.equal(sd58[k], sd32[k]), k
    assert models["58ch_2cls"].head[1].in_features == 58 * N_PATCHES * D_MODEL
    assert models["32ch_2cls"].head[1].in_features == 32 * N_PATCHES * D_MODEL


# --------------------------------------------------------------------------- #
# 3. pretrained weights: sha256 and strict loading
# --------------------------------------------------------------------------- #
def test_pretrained_weights_load_strictly(models):
    path = resolve_weights_path(A.PRETRAINED_FILE, required=False)
    if path is None:
        pytest.skip("weights absent")
    assert path.stat().st_size == 19_775_842
    assert sha256_file(path) == A.PRETRAINED_SHA256

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert len(state) == 211
    assert sum(v.numel() for v in state.values()) == CHECKPOINT_VALUES

    # strict load into a fresh upstream backbone: zero missing / unexpected keys
    torch.manual_seed(0)
    fresh = A.CBraMod()
    result = fresh.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []

    # ... and the adapter actually holds those tensors (not a re-initialised backbone)
    m = models["58ch_2cls"]
    for k, v in state.items():
        if k.startswith("proj_out"):
            continue                       # replaced by Identity after loading, by design
        assert torch.equal(m.backbone.state_dict()[k], v), k
    assert isinstance(m.backbone.proj_out, torch.nn.Identity)

    d = m.describe()
    assert d["pretrained"]["loaded"] is True
    assert d["pretrained"]["missing_keys"] == 0 and d["pretrained"]["unexpected_keys"] == 0
    assert d["pretrained_sha256"] == A.PRETRAINED_SHA256
    assert d["pretrained_file"] == A.PRETRAINED_FILE


def test_sha256_mismatch_is_refused():
    if not _weights_present():
        pytest.skip("weights absent")
    with pytest.raises(ValueError, match="sha256"):
        A.build(n_channels=32, n_times=1000, n_classes=2, sfreq=250.0,
                params={"weights_sha256": "00" * 32})


def test_mask_token_never_receives_a_gradient(models):
    """`mask_encoding` is trainable only nominally (upstream fine-tuning does the same); with
    mask=None it is not in the graph, so Adam never touches it and it stays at zero."""
    m = models["32ch_2cls"]
    m.train()
    m.zero_grad()
    m(torch.randn(2, 32, 1000))["logits"].sum().backward()
    tok = m.backbone.patch_embedding.mask_encoding
    assert tok.requires_grad and tok.grad is None
    assert torch.count_nonzero(tok) == 0
    assert m.head[1].weight.grad is not None
    assert m.backbone.encoder.layers[0].linear1.weight.grad is not None
    m.zero_grad()


# --------------------------------------------------------------------------- #
# 4. format conversion written by the adapter
# --------------------------------------------------------------------------- #
def test_resample_1000_at_250_gives_800_at_200():
    x = torch.randn(2, 32, 1000)
    y = resample_time(x, 250.0, 200.0, n_out=800)
    assert y.shape == (2, 32, 800)
    # a pure tone survives: 10 Hz at 250 Hz -> 10 Hz at 200 Hz (compare away from the edges)
    t250 = torch.arange(1000) / 250.0
    t200 = torch.arange(800) / 200.0
    tone = torch.sin(2 * torch.pi * 10 * t250)[None, None]
    ref = torch.sin(2 * torch.pi * 10 * t200)
    got = resample_time(tone, 250.0, 200.0, n_out=800)[0, 0]
    assert torch.allclose(got[40:-40], ref[40:-40], atol=2e-2)


def test_to_patches_layout(models):
    """Patch k = native samples [200k, 200k+200), per channel, i.e. reshape(b, c, -1, 200)."""
    m = models["32ch_2cls"]
    x = torch.randn(3, 32, 1000)
    p = m.to_patches(x)
    assert p.shape == (3, 32, N_PATCHES, 200)
    native = resample_time(x, 250.0, 200.0, n_out=800)
    assert torch.equal(p, native.reshape(3, 32, N_PATCHES, 200))
    for k in range(N_PATCHES):
        assert torch.equal(p[:, :, k, :], native[:, :, 200 * k:200 * (k + 1)])
    assert m.native_n_times == 800 and m.n_patches == N_PATCHES


def test_features_are_the_flattened_tokens(models):
    m = models["32ch_2cls"].eval()
    x = torch.randn(2, 32, 1000)
    with torch.no_grad():
        tokens = m.backbone(m.to_patches(x))
        out = m(x)
    assert tokens.shape == (2, 32, N_PATCHES, D_MODEL)
    assert torch.equal(out["features"], tokens.flatten(1))


def test_incompatible_geometry_raises():
    """A trial length that does not cut into whole 1 s patches must fail loudly at build."""
    if not _weights_present():
        pytest.skip("weights absent")
    with pytest.raises(ValueError, match="multiple of patch_len"):
        A.build(n_channels=32, n_times=900, n_classes=2, sfreq=250.0, params={})
    with pytest.raises(ValueError, match="patch_len"):
        A.build(n_channels=32, n_times=1000, n_classes=2, sfreq=250.0, params={"patch_len": 100})


# --------------------------------------------------------------------------- #
# 5. describe() and yaml
# --------------------------------------------------------------------------- #
def test_describe_has_required_keys(models):
    d = models["58ch_3cls"].describe()
    for k in ("name", "source", "n_params", "n_trainable_params", "feature_dim",
              "pretrained_file", "pretrained_sha256", "channels_kept", "channels_dropped",
              "native_sfreq", "native_n_times", "patch_len", "n_patches", "n_tokens"):
        assert k in d, k
    assert d["name"] == "cbramod"
    assert A.UPSTREAM_URL in d["source"] and A.UPSTREAM_COMMIT in d["source"]
    assert d["native_sfreq"] == 200.0 and d["native_n_times"] == 800
    assert d["n_tokens"] == 58 * N_PATCHES
    assert d["channels_kept"] == 58 and d["channels_dropped"] == 0 and d["channel_agnostic"]
    assert d["fine_tuning"] == "full"
    assert d["backbone"] == {"in_dim": 200, "out_dim": 200, "d_model": 200,
                             "dim_feedforward": 800, "seq_len": 30, "n_layer": 12, "nhead": 8}


def test_yaml_defaults_are_upstream_defaults():
    p = resolve_model_params("cbramod", dataset="wbci_shu_2c")
    assert p == resolve_model_params("cbramod", dataset="shu"), "channel-agnostic: no per_dataset"
    up = A.CBraModConfig()
    for k in ("in_dim", "out_dim", "d_model", "dim_feedforward", "seq_len", "n_layer", "nhead"):
        assert p[k] == getattr(up, k), k
    assert p["native_sfreq"] == 200.0 and p["patch_len"] == 200 and p["head_dropout"] == 0.1
    assert p["weights_file"] == A.PRETRAINED_FILE
    assert p["weights_sha256"] == A.PRETRAINED_SHA256
    assert set(p) <= set(A.KNOWN_PARAMS), set(p) - set(A.KNOWN_PARAMS)


def test_unknown_param_warns_but_builds(caplog):
    if not _weights_present():
        pytest.skip("weights absent")
    with caplog.at_level(logging.WARNING, logger="models.external"):
        m = A.build(n_channels=32, n_times=1000, n_classes=2, sfreq=250.0,
                    params={"definitely_not_a_param": 1})
    assert m.describe()["n_params"] == CONFIGS["32ch_2cls"][3]
    assert any("definitely_not_a_param" in r.getMessage() for r in caplog.records)


def test_no_custom_training_hooks(models):
    """Plain cross-entropy through the shared trainer; nothing model-specific."""
    m = models["32ch_2cls"]
    assert not getattr(m, "uses_custom_loss", False)
    assert not hasattr(m, "training_step")
    assert m.default_lr() == 1e-4


# --------------------------------------------------------------------------- #
# 6. provenance: vendored files are upstream, modulo the one documented import line
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fname", ["criss_cross_transformer.py", "LICENSE"])
def test_vendored_files_match_upstream(fname):
    assert sha256_file(OFFICIAL / fname) == UPSTREAM_SHA256[fname], fname


def test_vendored_cbramod_py_differs_from_upstream_only_in_the_import_line():
    text = (OFFICIAL / "cbramod.py").read_text(encoding="utf-8")
    assert text.count(VENDORED_IMPORT) == 1
    restored = text.replace(VENDORED_IMPORT, UPSTREAM_IMPORT)
    assert hashlib.sha256(restored.encode("utf-8")).hexdigest() == UPSTREAM_SHA256["cbramod.py"]
    assert (OFFICIAL / "__init__.py").read_bytes() == b""     # upstream models/__init__.py is empty
