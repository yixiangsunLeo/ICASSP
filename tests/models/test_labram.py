"""LaBraM-base external baseline: contract, format conversion, checkpoint keys, channels."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from code.models.external._common import (
    dataset_channel_names,
    resample_time,
    resolve_weights_path,
    sha256_file,
)
from code.models.external.labram import adapter as A
from code.models.model_specs import resolve_model_params
from code.models.registry import build_model

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "code" / "models" / "external" / "labram" / "_official"
EMBED, N_SEG = 200, 4
BACKBONE_PARAMS = 5_824_736   # NeuralTransformer after head->Identity; independent of C

CONFIGS = {
    "58ch_2cls": (58, 2, "wbci_shu_2c"),
    "58ch_3cls": (58, 3, "wbci_shu_3c"),
    "32ch_2cls": (32, 2, "shu"),
}


def _weights_present() -> bool:
    return resolve_weights_path(A.DEFAULT_WEIGHTS_FILE, required=False) is not None


def _build(key: str):
    n_ch, n_cls, dataset = CONFIGS[key]
    torch.manual_seed(0)
    params = resolve_model_params("labram", dataset=dataset)
    return build_model("labram", n_channels=n_ch, n_times=1000, n_classes=n_cls,
                       sfreq=250, params=params)


@pytest.fixture(scope="module")
def models():
    if not _weights_present():
        pytest.skip(f"{A.DEFAULT_WEIGHTS_FILE} not found in weights/")
    return {k: _build(k) for k in CONFIGS}


@pytest.mark.parametrize("key", list(CONFIGS))
def test_registry_build_forward_and_eval_determinism(models, key):
    n_ch, n_cls, _ = CONFIGS[key]
    m = models[key]
    x = torch.randn(2, n_ch, 1000)
    m.train()
    out = m(x)
    n_kept = m.describe()["n_channels_kept"]
    feat = (1 + n_kept * N_SEG) * EMBED
    assert set(out) == {"logits", "features", "confidence"}
    assert out["logits"].shape == (2, n_cls)
    assert out["features"].shape == (2, feat)
    assert out["confidence"] is None
    assert torch.isfinite(out["logits"]).all()
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(a["features"], b["features"])


def test_accepts_4d_input(models):
    m = models["32ch_2cls"].eval()
    x = torch.randn(2, 32, 1000)
    with torch.no_grad():
        assert torch.equal(m(x)["logits"], m(x.unsqueeze(1))["logits"])


def test_wrong_channel_count_raises(models):
    with pytest.raises(ValueError, match="channels"):
        models["32ch_2cls"](torch.randn(2, 58, 1000))


@pytest.mark.parametrize("key", list(CONFIGS))
def test_parameter_and_geometry(models, key):
    n_ch, n_cls, _ = CONFIGS[key]
    d = models[key].describe()
    n_kept = d["n_channels_kept"]
    feat = (1 + n_kept * N_SEG) * EMBED
    assert d["feature_dim"] == feat
    assert d["n_tokens"] == 1 + n_kept * N_SEG
    assert d["embed_dim"] == EMBED and d["n_segments"] == N_SEG
    assert d["native_sfreq"] == 200.0 and d["native_length"] == 800
    assert d["n_backbone_params"] == BACKBONE_PARAMS
    assert d["n_head_params"] == feat * n_cls + n_cls
    assert d["n_params"] == BACKBONE_PARAMS + feat * n_cls + n_cls
    assert d["n_trainable_params"] == d["n_params"]
    assert d["n_channels_in"] == n_ch
    assert d["channels_dropped"] == []


def test_backbone_params_independent_of_channel_count(models):
    assert models["58ch_2cls"].describe()["n_backbone_params"] == \
        models["32ch_2cls"].describe()["n_backbone_params"]


def test_all_cohort_channels_are_in_standard_1020():
    for set_name, n in (("wbci_shu", 58), ("shu", 32)):
        names = dataset_channel_names(set_name)
        assert len(names) == n
        keep, chans, kept, dropped = A.labram_channel_plan(names)
        assert dropped == []
        assert len(keep) == n and chans[0] == 0 and len(chans) == n + 1
        assert max(chans) < 129          # pos_embed rows: cls + 128 vocab slots used by index+1


def test_shu_keeps_a1_a2_and_legacy_t3t4t5t6():
    names = dataset_channel_names("shu")
    assert {"A1", "A2", "T3", "T4", "T5", "T6"} <= set(names)
    _, chans, kept, dropped = A.labram_channel_plan(names)
    assert dropped == []
    assert {"A1", "A2", "T3", "T4", "T5", "T6"} <= set(kept)
    # vocabulary lists the legacy names themselves, so they are used as-is (not T7/P7)
    assert "T3" in kept and "T7" not in kept


def test_resample_and_segment():
    x = torch.randn(2, 32, 1000)
    y = resample_time(x, 250.0, 200.0, n_out=800)
    assert y.shape == (2, 32, 800)
    z = A.segment_trials(y, 200)
    assert z.shape == (2, 32, 4, 200)
    assert torch.equal(z[:, :, 1, :], y[:, :, 200:400])


def test_pretrained_weights_load_with_only_expected_gaps(models):
    path = resolve_weights_path(A.DEFAULT_WEIGHTS_FILE, required=False)
    if path is None:
        pytest.skip("weights absent")
    assert path.stat().st_size == A.DEFAULT_WEIGHTS_SIZE
    assert sha256_file(path) == A.DEFAULT_WEIGHTS_SHA256
    d = models["58ch_2cls"].describe()
    assert d["pretrained"] is True
    assert d["pretrained_sha256"] == A.DEFAULT_WEIGHTS_SHA256
    assert int(d["n_tensors_loaded"]) > 0
    missing = d["missing_keys"]
    unexpected = d["unexpected_keys"]
    assert all(any(s in k for s in A.EXPECTED_MISSING_SUBSTRINGS) for k in missing)
    assert set(unexpected) <= A.EXPECTED_UNEXPECTED_KEYS
    assert "lm_head.weight" in unexpected and "lm_head.bias" in unexpected


def test_sha256_mismatch_is_refused():
    if not _weights_present():
        pytest.skip("weights absent")
    with pytest.raises(RuntimeError, match="sha256"):
        A.build(n_channels=32, n_times=1000, n_classes=2, sfreq=250.0,
                params={"channel_set": "shu", "weights_sha256": "00" * 32})


def test_needs_channel_set():
    if not _weights_present():
        pytest.skip("weights absent")
    with pytest.raises(ValueError, match="channel_set"):
        A.build(n_channels=32, n_times=1000, n_classes=2, sfreq=250.0, params={})


def test_yaml_and_hooks(models):
    p = resolve_model_params("labram", dataset="wbci_shu_2c")
    assert p["arch"] == "labram_base_patch200_200"
    assert p["channel_set"] == "wbci_shu"
    assert resolve_model_params("labram", dataset="shu")["channel_set"] == "shu"
    assert p["pretrained"] is True and p["weights_file"] == A.DEFAULT_WEIGHTS_FILE
    m = models["32ch_2cls"]
    assert not getattr(m, "uses_custom_loss", False)
    assert not hasattr(m, "training_step")


def test_unknown_param_warns(caplog):
    if not _weights_present():
        pytest.skip("weights absent")
    with caplog.at_level(logging.WARNING, logger="models.external"):
        A.build(n_channels=32, n_times=1000, n_classes=2, sfreq=250.0,
                params={"channel_set": "shu", "definitely_not_a_param": 1})
    assert any("definitely_not_a_param" in r.getMessage() for r in caplog.records)


def test_official_modeling_finetune_is_verbatim():
    text = (OFFICIAL / "modeling_finetune.py").read_text(encoding="utf-8")
    assert "class NeuralTransformer" in text
    assert "def labram_base_patch200_200" in text
    assert "from timm.models.layers import" in text
