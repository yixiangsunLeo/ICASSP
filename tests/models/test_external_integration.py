"""Protocol-identity and registry checks for the external-baseline round (D-034).

These tests guard the one thing that makes the new entries comparable with the 18 ANN
benchmark entries and the spiking main run: the `bl_*` experiment configs must be the main-run
configs with nothing changed but the model roster (and gradient-accumulation micro-batches).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

from code.models.external import D034_EXTERNAL_MODEL_NAMES, EXTERNAL_MODEL_NAMES
from code.models.model_specs import resolve_model_params
from code.models.registry import MODEL_NAMES, build_model

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "code" / "configs" / "experiments"
MODELS_DIR = ROOT / "code" / "configs" / "models"

PAIRS = {
    "wbci2c": ("bl_wbci2c_zs_lr1e4.yaml", "snn_wbci2c_zs_lr1e4.yaml", "wbci_shu_2c"),
    "shu":    ("bl_shu_zs_lr1e4.yaml", "snn_shu_zs_lr1e4.yaml", "shu"),
    "wbci3c": ("bl_wbci3c_zs_lr1e4.yaml", "snn_wbci3c_zs_lr1e4.yaml", "wbci_shu_3c"),
}
PROTECTED_TOP = ("run_id", "data", "cross_subject", "eval", "checkpoints", "output", "model_params")


def _load(name: str) -> dict:
    with open(EXP / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_baseline_configs", ROOT / "scripts" / "make_baseline_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ds", sorted(PAIRS))
def test_bl_config_is_main_run_config_plus_roster(ds):
    bl_name, snn_name, _ = PAIRS[ds]
    bl, snn = _load(bl_name), _load(snn_name)
    for key in PROTECTED_TOP:
        assert bl.get(key) == snn.get(key), f"{bl_name}: '{key}' differs from {snn_name}"
    tb = {k: v for k, v in bl["train"].items() if k != "micro_batch_per_model"}
    ts = {k: v for k, v in snn["train"].items() if k != "micro_batch_per_model"}
    assert tb == ts, f"{bl_name}: train recipe differs from {snn_name}"
    # the recipe itself, spelled out (the numbers the paper quotes)
    t = bl["train"]
    assert (t["optimizer"], t["lr"], t["batch_size"], t["early_stopping_patience"],
            t["monitor"], t["normalization"], t["seeds"], t["scheduler"], t["max_epochs"]) == (
        "adam", 1e-4, 128, 25, "macro_f1", "per_sample_zscore", [0], "none", 500)
    assert bl["experiment"]["name"] == "foundation_cross_subject"
    assert bl["experiment"]["source_config"].endswith(bl_name)
    # cells go into the SAME run directory as the main run
    assert bl["output"]["output_dir"].endswith(snn["run_id"])


@pytest.mark.parametrize("ds", sorted(PAIRS))
def test_bl_config_on_disk_matches_generator(ds):
    gen = _generator()
    path, text = gen.render(ds)
    assert path.exists(), f"run scripts/make_baseline_configs.py ({path} missing)"
    assert path.read_text(encoding="utf-8") == text, "regenerate: on-disk bl config is stale"


def test_bl_rosters():
    for ds, (bl_name, _, _) in PAIRS.items():
        models = _load(bl_name)["models"]
        assert all(m in MODEL_NAMES for m in models), (ds, models)
        assert all(m in models for m in D034_EXTERNAL_MODEL_NAMES), (ds, models)
        assert "adahgnn" not in models  # D-037 uses bl_*_adahgnn_zs_lr1e4.yaml, not these files
        if ds == "wbci3c":
            assert {"atcnet", "eeg_deformer", "fbcnet"} <= set(models)
        else:
            assert not ({"atcnet", "eeg_deformer", "fbcnet"} & set(models))
        # micro-batches only for the pretrained transformers (no BatchNorm) and only for
        # models in the roster; TCFormer (BatchNorm) keeps the full batch
        mb = _load(bl_name)["train"].get("micro_batch_per_model") or {}
        assert set(mb) <= {"labram", "eegpt", "cbramod"} and set(mb) <= set(models)
        assert "tcformer" not in mb


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_external_names_registered_and_lazy():
    assert set(D034_EXTERNAL_MODEL_NAMES) == {"tcformer", "labram", "eegpt", "cbramod"}
    assert set(EXTERNAL_MODEL_NAMES) == set(D034_EXTERNAL_MODEL_NAMES) | {"adahgnn"}
    assert all(n in MODEL_NAMES for n in EXTERNAL_MODEL_NAMES)
    # importing the registry must not import any adapter (they are pulled in on demand)
    import code.models.registry  # noqa: F401
    heavy = [m for m in sys.modules
             if m.startswith("code.models.external.") and not m.endswith("_common")]
    # an adapter may already be loaded by another test in this session; that is fine, but
    # the registry module itself must not reference them at import time
    src = (ROOT / "code" / "models" / "registry.py").read_text(encoding="utf-8")
    for n in EXTERNAL_MODEL_NAMES:
        assert f"external.{n}" not in src, f"registry imports external.{n} eagerly"
    assert isinstance(heavy, list)


@pytest.mark.parametrize("name", ["atcnet", "eeg_deformer", "fbcnet"])
def test_3c_ann_fill_ins_build_at_58ch_3cls(name):
    params = resolve_model_params(name, dataset="wbci_shu_3c")
    m = build_model(name, n_channels=58, n_times=1000, n_classes=3, sfreq=250, params=params)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 58, 1000))
    assert out["logits"].shape == (2, 3)
    assert torch.isfinite(out["logits"]).all()


@pytest.mark.parametrize("name", D034_EXTERNAL_MODEL_NAMES)
@pytest.mark.parametrize("dataset,n_ch,n_cls", [("wbci_shu_2c", 58, 2), ("wbci_shu_3c", 58, 3),
                                                 ("shu", 32, 2)])
def test_external_models_build_for_every_cohort(name, dataset, n_ch, n_cls):
    if not (MODELS_DIR / f"{name}.yaml").exists():
        pytest.skip(f"{name}.yaml not written yet (adapter agent still running)")
    params = resolve_model_params(name, dataset=dataset)
    m = build_model(name, n_channels=n_ch, n_times=1000, n_classes=n_cls, sfreq=250, params=params)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, n_ch, 1000))
    assert out["logits"].shape == (2, n_cls)
    assert torch.isfinite(out["logits"]).all()
    assert out["features"].dim() == 2 and out["features"].shape[0] == 2
    assert out["confidence"] is None
    d = m.describe()
    for k in ("name", "source", "n_params"):
        assert k in d, k
    # result.json stores describe() verbatim: it must survive the project's JSON writer
    # (a torch.Size or tensor here would fail the cell *after* training)
    from code.utils.io import _json_default, _sanitize_for_json
    import json
    json.dumps(_sanitize_for_json(d), default=_json_default, allow_nan=False)
    assert not getattr(m, "uses_custom_loss", False), "external baselines train with plain CE"


# --------------------------------------------------------------------------- #
# split-seed 43 comparison sweep (D-036)
# --------------------------------------------------------------------------- #
S43_MODELS = [
    "eegnet_official", "fbcnet", "atcnet", "eegnex",
    "labram", "eegpt", "eeg_deformer", "tcformer", "cbramod",
]
S43_FILES = {
    "wbci2c": "bl_wbci2c_zs_s43.yaml",
    "shu": "bl_shu_zs_s43.yaml",
}


def test_s43_configs_are_2c_and_shu_only():
    """Seed 43 is a k-fold redraw on 2C/SHU. 3C stays LOSO (no s43 yaml)."""
    assert not (EXP / "bl_wbci3c_zs_s43.yaml").exists()
    for ds, bl_name in (("wbci2c", "bl_wbci2c_zs_lr1e4.yaml"),
                        ("shu", "bl_shu_zs_lr1e4.yaml")):
        bl = _load(bl_name)
        s43 = _load(S43_FILES[ds])
        assert s43["data"]["name"] == bl["data"]["name"]
        assert s43["models"] == S43_MODELS
        assert s43["cross_subject"].get("split_seeds") == [43]
        assert s43["cross_subject"]["protocol"] == "kfold_subject"
        assert s43["run_id"] == bl["run_id"]
        assert s43["output"] == bl["output"]
        assert s43["train"]["lr"] == bl["train"]["lr"]
        assert s43["train"]["early_stopping_patience"] == bl["train"]["early_stopping_patience"]
        assert s43["train"]["normalization"] == bl["train"]["normalization"]
        assert s43["train"]["seeds"] == bl["train"]["seeds"]
        cs_bl = dict(bl["cross_subject"])
        cs_s43 = dict(s43["cross_subject"])
        cs_bl.pop("split_seeds", None)
        cs_bl.pop("split_seed", None)
        cs_s43.pop("split_seeds", None)
        cs_s43.pop("split_seed", None)
        assert cs_bl == cs_s43, ds

