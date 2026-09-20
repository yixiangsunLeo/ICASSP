"""TCFormer external baseline: contract, paper config, no Lightning, provenance of imports."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from code.models.external.tcformer import adapter as A
from code.models.model_specs import resolve_model_params
from code.models.registry import build_model

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "code" / "models" / "external" / "tcformer" / "_official"

# Measured on this tree (paper yaml, trans_depth=5). Spatial DW-conv kernel is (C, 1),
# so the count depends on C; the classifier adds n_classes * 64 + n_classes.
CONFIGS = {
    "58ch_2cls": (58, 2, "wbci_shu_2c", 133_988),
    "58ch_3cls": (58, 3, "wbci_shu_3c", 134_056),
    "32ch_2cls": (32, 2, "shu", 128_996),
}


def _build(key: str):
    n_ch, n_cls, dataset, _ = CONFIGS[key]
    torch.manual_seed(0)
    params = resolve_model_params("tcformer", dataset=dataset)
    return build_model("tcformer", n_channels=n_ch, n_times=1000, n_classes=n_cls,
                       sfreq=250, params=params)


@pytest.fixture(scope="module")
def models():
    return {k: _build(k) for k in CONFIGS}


@pytest.mark.parametrize("key", list(CONFIGS))
def test_registry_build_forward_and_eval_determinism(models, key):
    n_ch, n_cls, _, _ = CONFIGS[key]
    m = models[key]
    x = torch.randn(2, n_ch, 1000)
    m.train()
    out = m(x)
    assert set(out) == {"logits", "features", "confidence"}
    assert out["logits"].shape == (2, n_cls)
    assert out["features"].shape == (2, 64)
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
def test_parameter_counts(models, key):
    n_ch, n_cls, _, expected = CONFIGS[key]
    d = models[key].describe()
    assert d["n_params"] == expected
    assert d["n_trainable_params"] == expected
    assert d["feature_dim"] == 64
    assert d["n_tokens"] == 17          # (1000 // 8) // 7
    assert d["pretrained_weights"] is None


def test_paper_n2_headline_count_at_bci2a_shape():
    """README 77.8 k is trans_depth=2 at BCI IV-2a (22 ch / 4 cls); yaml default is depth 5."""
    params = resolve_model_params("tcformer@n2")
    assert params["trans_depth"] == 2
    m = build_model("tcformer@n2", n_channels=22, n_times=1125, n_classes=4,
                    sfreq=250, params=params)
    n = m.describe()["n_params"]
    assert n == 77_820, n
    assert m.describe()["n_tokens"] == 20   # (1125 // 8) // 7


def test_yaml_defaults_are_upstream_yaml_not_class_defaults():
    p = resolve_model_params("tcformer", dataset="wbci_shu_2c")
    assert p["F1"] == 32 and p["temp_kernel_lengths"] == [20, 32, 64]
    assert p["q_heads"] == 4 and p["kv_heads"] == 2
    assert p["trans_depth"] == 5 and p["dropout_conv"] == 0.4
    assert p["drop_path_max"] == 0.25
    assert "channel_set" not in p


def test_describe_and_no_custom_hooks(models):
    d = models["58ch_2cls"].describe()
    for k in ("name", "source", "n_params", "paper", "n_tokens", "d_model"):
        assert k in d, k
    assert d["name"] == "tcformer"
    assert A.UPSTREAM_URL in d["source"] and A.UPSTREAM_COMMIT in d["source"]
    m = models["32ch_2cls"]
    assert not getattr(m, "uses_custom_loss", False)
    assert not hasattr(m, "training_step")
    assert not hasattr(m, "default_lr")


def test_backward_reaches_conv_and_head(models):
    m = models["32ch_2cls"]
    m.train()
    m.zero_grad()
    m(torch.randn(2, 32, 1000))["logits"].sum().backward()
    assert m.model.tcn_head.classifier.linear.weight.grad is not None
    assert m.model.conv_block.temporal_convs[0][1].weight.grad is not None
    m.zero_grad()


def test_adapter_does_not_import_lightning():
    import sys
    # classification_module.py is vendored for the record but must not be imported
    assert "pytorch_lightning" not in sys.modules
    text = (OFFICIAL / "models" / "tcformer.py").read_text(encoding="utf-8")
    assert "from ..._compat import ClassificationModule" in text
    assert "from models.classification_module" not in text
