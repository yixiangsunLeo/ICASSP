"""EEGPT large external baseline: contract, SHU channel plan, 256 Hz resample, strict load."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from code.models.external._common import (
    dataset_channel_names,
    match_channels,
    resample_time,
    resolve_weights_path,
    sha256_file,
)
from code.models.external.eegpt import adapter as A
from code.models.model_specs import resolve_model_params
from code.models.registry import build_model

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "code" / "models" / "external" / "eegpt" / "_official"

# Encoder + chan_conv + head; chan_conv is n_kept^2 + n_kept, head is 32768*n_cls + n_cls.
CONFIGS = {
    "58ch_2cls": (58, 2, "wbci_shu_2c", 25_356_128, 58, []),
    "58ch_3cls": (58, 3, "wbci_shu_3c", 25_388_897, 58, []),
    "32ch_2cls": (32, 2, "shu", 25_353_636, 30, ["A1", "A2"]),
}
FEATURE_DIM = 16 * 4 * 512  # 32768


def _weights_present() -> bool:
    return resolve_weights_path("eegpt.ckpt", required=False) is not None


def _build(key: str):
    n_ch, n_cls, dataset, *_ = CONFIGS[key]
    torch.manual_seed(0)
    params = resolve_model_params("eegpt", dataset=dataset)
    return build_model("eegpt", n_channels=n_ch, n_times=1000, n_classes=n_cls,
                       sfreq=250, params=params)


@pytest.fixture(scope="module")
def models():
    if not _weights_present():
        pytest.skip("eegpt.ckpt not found in weights/")
    return {k: _build(k) for k in CONFIGS}


@pytest.mark.parametrize("key", list(CONFIGS))
def test_registry_build_forward_and_eval_determinism(models, key):
    n_ch, n_cls, *_ = CONFIGS[key]
    m = models[key]
    x = torch.randn(2, n_ch, 1000)
    m.train()
    out = m(x)
    assert set(out) == {"logits", "features", "confidence"}
    assert out["logits"].shape == (2, n_cls)
    assert out["features"].shape == (2, FEATURE_DIM)
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
def test_parameter_counts_and_channel_plan(models, key):
    n_ch, n_cls, _, expected, n_kept, dropped = CONFIGS[key]
    d = models[key].describe()
    assert d["n_params"] == expected
    assert d["n_trainable_params"] == expected
    assert d["feature_dim"] == FEATURE_DIM
    assert d["n_channels_in"] == n_ch
    assert d["n_channels_kept"] == n_kept
    assert d["channels_dropped"] == dropped
    assert d["native_sfreq"] == 256.0 and d["n_native_times"] == 1024
    assert d["n_patches"] == 16 and d["embed_dim"] == 512 and d["embed_num"] == 4
    assert d["alignment"] == "none (protocol P4: no model-specific EA)"


def test_shu_drops_earlobes_and_renames_legacy_1020():
    names = dataset_channel_names("shu")
    keep, matched, dropped = match_channels(names, A.CHANNEL_DICT.keys())
    assert dropped == ["A1", "A2"]
    assert len(keep) == 30
    # T3/T4/T5/T6 are not in EEGPT's vocab; canonical aliases are
    assert "T7" in matched and "T8" in matched and "P7" in matched and "P8" in matched
    assert "T3" not in matched


def test_wbci_keeps_all_58():
    names = dataset_channel_names("wbci_shu")
    keep, matched, dropped = match_channels(names, A.CHANNEL_DICT.keys())
    assert dropped == [] and len(keep) == 58 == len(names)


def test_resample_1000_at_250_gives_1024_at_256():
    x = torch.randn(2, 8, 1000)
    y = resample_time(x, 250.0, 256.0, n_out=1024)
    assert y.shape == (2, 8, 1024)


def test_to_native_format(models):
    m = models["32ch_2cls"]
    x = torch.randn(3, 32, 1000)
    y = m.to_native_format(x)
    assert y.shape == (3, 30, 1024)


def test_pretrained_strict_load(models):
    path = resolve_weights_path("eegpt.ckpt", required=False)
    if path is None:
        pytest.skip("weights absent")
    assert sha256_file(path) == A.WEIGHTS_SHA256
    d = models["58ch_2cls"].describe()
    assert d["pretrained"] is True
    assert d["pretrained_sha256"] == A.WEIGHTS_SHA256
    assert d["strict_load"] is True
    assert d["pretrained_prefix"] == A.TARGET_ENCODER_PREFIX


def test_sha256_mismatch_is_refused():
    if not _weights_present():
        pytest.skip("weights absent")
    with pytest.raises(RuntimeError, match="sha256"):
        A.build(n_channels=58, n_times=1000, n_classes=2, sfreq=250.0,
                params={"channel_set": "wbci_shu", "weights_sha256": "00" * 32})


def test_yaml_and_hooks(models):
    p = resolve_model_params("eegpt", dataset="shu")
    assert p["channel_set"] == "shu"
    assert p["embed_dim"] == 512 and p["depth"] == 8 and p["patch_size"] == 64
    assert p["readout"] == "flatten" and p["head_dropout"] == 0.5
    m = models["32ch_2cls"]
    assert not getattr(m, "uses_custom_loss", False)
    assert not hasattr(m, "training_step")


def test_describe_json_keys(models):
    d = models["58ch_3cls"].describe()
    for k in ("name", "source", "n_params", "pretrained_file", "pretrained_sha256",
              "channels_kept", "channels_dropped", "chans_id"):
        assert k in d, k
    assert d["name"] == "eegpt"
    assert A.UPSTREAM_URL in d["source"] and A.UPSTREAM_COMMIT in d["source"]
    from code.utils.io import _json_default, _sanitize_for_json
    import json
    json.dumps(_sanitize_for_json(d), default=_json_default, allow_nan=False)


def test_official_encoder_file_present():
    assert (OFFICIAL / "downstream" / "Modules" / "models" / "EEGPT_mcae.py").exists()
    assert (OFFICIAL / "LICENSE").exists()
