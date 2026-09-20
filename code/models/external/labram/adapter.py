"""LaBraM-base (Jiang, Zhao & Lu, ICLR 2024) behind this project's model contract.

Upstream code: https://github.com/935963004/LaBraM @ ``c431221e6cfd23dbfa9950e0180682fb322b0548``
(``modeling_finetune.py`` vendored verbatim in ``_official/``; the ``standard_1020`` electrode
vocabulary excerpted verbatim from upstream ``utils.py``). Construction kwargs, checkpoint
loading and the classification head follow the EEG-FM-Compass benchmark wrapper (Liu et al.,
NSR 2026; ``models/FM/LaBraM/Loader_LaBraM.py``), which itself transcribes upstream
``run_class_finetuning.py`` L193-208 / L333-371. Every deviation is listed in ``PROVENANCE.md``.

Input contract: ``x[B, C, 1000] @ 250 Hz`` (per-trial z-scored by the shared data pipeline).
Inside ``forward`` only *format conversion* happens (protocol P4, ``docs/PROTOCOL_external_baselines.md``):

  1. windowed-sinc resampling 250 -> 200 Hz (LaBraM's native rate): 1000 -> 800 samples;
  2. channel lookup in LaBraM's ``standard_1020`` vocabulary -> ``input_chans`` ids that pick the
     rows of the pretrained ``pos_embed`` (id 0 = cls token, id i+1 = vocabulary entry i);
  3. cut into 4 segments x 200 samples (1 s each) -> ``[B, C, 4, 200]``, the layout the upstream
     ``forward`` documents.

No amplitude scaling (upstream's "0.1 mV unit" convention is NOT applied), no filtering, no
alignment. Full fine-tuning: every parameter trainable, plain cross-entropy (no custom hooks).
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ....utils.logging_utils import get_logger
from .._common import (
    count_params,
    dataset_channel_names,
    match_channels,
    resample_time,
    resolve_weights_path,
    sha256_file,
    squeeze_input,
    warn_unknown_params,
)

# The vendored upstream file imports ``timm.models.layers`` / ``timm.models.registry``; timm 1.0
# re-exports both with a FutureWarning. The imports work, so the upstream lines stay untouched
# and the (harmless) warnings are silenced here instead of polluting every Slurm log.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from ._official.modeling_finetune import (  # noqa: E402
        NeuralTransformer,
        labram_base_patch200_200,
        labram_huge_patch200_200,
        labram_large_patch200_200,
    )
from ._official.standard_1020 import standard_1020 as STANDARD_1020  # noqa: E402

logger = get_logger("models.external.labram")

UPSTREAM_URL = "https://github.com/935963004/LaBraM"
UPSTREAM_COMMIT = "c431221e6cfd23dbfa9950e0180682fb322b0548"
BENCHMARK_URL = "https://github.com/Dingkun0817/EEG-FM-Benchmark"  # EEG-FM-Compass wrapper we follow
PAPER = "W.-B. Jiang, L.-M. Zhao, B.-L. Lu, ICLR 2024"

#: Constructors from the vendored upstream file, called directly (no timm registry lookup, so a
#: name clash with another registered ``labram_*`` model can never pick the wrong class).
ARCHS = {
    "labram_base_patch200_200": labram_base_patch200_200,
    "labram_large_patch200_200": labram_large_patch200_200,
    "labram_huge_patch200_200": labram_huge_patch200_200,
}

#: Official LaBraM-base pretraining checkpoint (bundled in the upstream repo, ``checkpoints/``).
DEFAULT_WEIGHTS_FILE = "labram-base.pth"
DEFAULT_WEIGHTS_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
DEFAULT_WEIGHTS_SIZE = 96_612_769

#: Keys the pretraining checkpoint legitimately cannot fill / the fine-tune model cannot use.
#: Anything outside these two sets makes ``load_labram_checkpoint`` raise.
#:   * ``head.*``            -- classifier; replaced by ``nn.Identity`` + our head anyway
#:   * ``fc_norm.*``         -- ``use_mean_pooling=True`` (upstream default) discards the pretrained
#:                              final ``norm`` and adds a fresh LayerNorm (BEiT-style fine-tuning)
#:   * ``*.attn.q_bias/v_bias`` -- pretraining used ``qkv_bias=False``; the fine-tune defaults
#:                              (upstream argparse and the benchmark) use ``qkv_bias=True`` ->
#:                              zero-initialised, i.e. function-preserving at load time
#:   * ``relative_position_index`` -- buffers, never stored (and absent here: window_size=None)
EXPECTED_MISSING_SUBSTRINGS = ("head.", "fc_norm.", ".attn.q_bias", ".attn.v_bias",
                               "relative_position_index")
#: Pretraining-only tensors that the fine-tune ``NeuralTransformer`` does not have:
#:   * ``mask_token``  -- masked-token placeholder
#:   * ``norm.*``      -- pretraining final norm; ``use_mean_pooling=True`` replaces it with ``fc_norm``
#:   * ``lm_head.*``   -- pretraining reconstruction / language-model head (official
#:                        ``labram-base.pth`` student branch; Compass and upstream
#:                        ``run_class_finetuning.py`` both ignore it under non-strict load)
EXPECTED_UNEXPECTED_KEYS = frozenset({
    "mask_token", "norm.weight", "norm.bias", "lm_head.weight", "lm_head.bias",
})


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class LaBraMConfig:
    """Structural hyperparameters. Defaults = EEG-FM-Compass ``Loader_LaBraM`` construction
    (= upstream ``run_class_finetuning.py`` argparse defaults plus ``--abs_pos_emb``)."""

    # data dims (injected by the experiment layer, never hard-coded)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    #: Which electrode list to look up in ``standard_1020``; comes from the yaml ``per_dataset``
    #: block (``"wbci_shu"`` -> 58 names, ``"shu"`` -> 32 names). Required.
    channel_set: Optional[str] = None

    # --- backbone (labram_base_patch200_200: embed 200, depth 12, heads 10, mlp x4, patch 200) ---
    arch: str = "labram_base_patch200_200"
    drop_rate: float = 0.0            # dropout inside blocks / pos_drop
    drop_path_rate: float = 0.1       # stochastic depth (linear decay over the 12 blocks)
    attn_drop_rate: float = 0.0
    init_values: float = 0.1          # layer-scale gamma init (pretraining used 0.1 too)
    init_scale: float = 0.001         # upstream head init scale (head is replaced; kept for parity)
    qkv_bias: bool = True             # upstream fine-tune default; pretraining had none (-> zero-init)
    use_abs_pos_emb: bool = True      # pretrained pos_embed [1, 129, 200], rows picked by input_chans
    use_rel_pos_bias: bool = True     # no-op in the fine-tune model (blocks are built with window_size=None)
    use_mean_pooling: bool = True     # -> fresh fc_norm LayerNorm on the tokens (upstream default)

    # --- readout / head (EEG-FM-Compass ``LinearLayers(feature_dim, n_classes, flatten=1, dropout)``) ---
    readout: str = "flatten"          # "flatten": all (1 + C*4) tokens flattened; "pooling": mean token
    head_dropout: float = 0.1

    # --- input format conversion ---
    native_sfreq: float = 200.0       # LaBraM patch = 200 samples = 1 s at 200 Hz
    segment_len: int = 200

    # --- pretrained weights (``weights/`` via ``resolve_weights_path``) ---
    pretrained: bool = True
    weights_file: str = DEFAULT_WEIGHTS_FILE
    weights_sha256: Optional[str] = DEFAULT_WEIGHTS_SHA256   # verified before loading; None = skip


# --------------------------------------------------------------------------- #
# format conversion helpers (unit-tested)
# --------------------------------------------------------------------------- #
def labram_channel_plan(channel_names: Sequence[str]
                        ) -> Tuple[List[int], List[int], List[str], List[str]]:
    """Map our electrode names onto LaBraM's ``standard_1020`` vocabulary.

    Returns ``(keep_idx, input_chans, kept_names, dropped_names)``:

    * ``keep_idx`` -- positions in the *input* channel axis that the vocabulary can represent
      (input order preserved); ``dropped_names`` are the rest.
    * ``input_chans`` -- ``[0] + [standard_1020.index(NAME) + 1, ...]`` exactly as the upstream /
      benchmark ``get_input_chans`` builds it (upper-cased raw name; 0 = cls token). The
      vocabulary lists the legacy 10-20 names T3/T4/T5/T6/A1/A2 themselves, so a raw name that
      is present is used as-is; only a name that is *not* present falls back to the canonical
      alias (T3 -> T7, ...) from ``match_channels``.
    """
    keep_idx, matched, dropped = match_channels(channel_names, STANDARD_1020)
    input_chans: List[int] = [0]
    kept: List[str] = []
    for i, canonical in zip(keep_idx, matched):
        raw = str(channel_names[i]).strip().upper()
        name = raw if raw in STANDARD_1020 else canonical
        input_chans.append(STANDARD_1020.index(name) + 1)
        kept.append(name)
    return keep_idx, input_chans, kept, dropped


def segment_trials(x: torch.Tensor, segment_len: int) -> torch.Tensor:
    """``[B, C, n_seg * segment_len]`` -> ``[B, C, n_seg, segment_len]`` (contiguous 1 s windows)."""
    B, C, T = x.shape
    if T % segment_len:
        raise ValueError(f"time axis {T} is not a multiple of segment_len={segment_len}")
    return x.reshape(B, C, T // segment_len, segment_len)


# --------------------------------------------------------------------------- #
# pretrained weights
# --------------------------------------------------------------------------- #
def load_labram_checkpoint(backbone: nn.Module, path: Path) -> Dict[str, object]:
    """Load the official pretraining checkpoint the way upstream ``run_class_finetuning.py``
    (and the EEG-FM-Compass ``Loader_LaBraM``) does, then *verify* what happened.

    Steps (upstream L342-371): pick ``checkpoint['model'|'module']``; keep only the ``student.``
    branch (prefix stripped); drop ``head.*`` if its shape differs; drop
    ``relative_position_index`` buffers; load non-strictly. Upstream then merely prints the
    missing / unexpected keys. Here every missing key must be in ``EXPECTED_MISSING_SUBSTRINGS``
    and every unexpected key in ``EXPECTED_UNEXPECTED_KEYS``, otherwise we raise: a silently
    half-loaded backbone must never reach a results table.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = None
    model_key = None
    for key in ("model", "module"):                   # upstream --model_key 'model|module'
        if isinstance(checkpoint, dict) and key in checkpoint:
            state, model_key = checkpoint[key], key
            break
    if state is None:
        state = checkpoint

    student = OrderedDict((k[len("student."):], v) for k, v in state.items()
                          if k.startswith("student."))
    if not student:
        # A plain (already fine-tuned) LaBraM state_dict has no 'student.' branch. Upstream
        # would end up loading nothing; we accept the dict as-is instead (minus DDP prefix).
        student = OrderedDict((k[len("module."):] if k.startswith("module.") else k, v)
                              for k, v in state.items())
    n_student = len(student)
    dropped_non_student = sorted(k for k in state if not k.startswith("student.")) \
        if n_student != len(state) else []

    model_state = backbone.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in student and k in model_state and student[k].shape != model_state[k].shape:
            del student[k]
    for k in list(student):
        if "relative_position_index" in k:
            student.pop(k)

    result = backbone.load_state_dict(student, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)

    bad_missing = [k for k in missing if not any(s in k for s in EXPECTED_MISSING_SUBSTRINGS)]
    bad_unexpected = [k for k in unexpected if k not in EXPECTED_UNEXPECTED_KEYS]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"labram: checkpoint {path} does not match the fine-tune model. "
            f"Unexplained missing keys: {bad_missing}; unexplained unexpected keys: {bad_unexpected}"
        )
    n_loaded = len(student) - len(unexpected)
    if n_loaded != len(model_state) - len(missing):
        raise RuntimeError("labram: tensor bookkeeping mismatch while loading the checkpoint")
    if n_loaded == 0:
        raise RuntimeError(f"labram: no tensor of {path} matched the model")

    logger.info("labram: loaded %d/%d tensors from %s (key=%s, student branch %d tensors); "
                "missing (expected) %s; unexpected (expected) %s",
                n_loaded, len(model_state), path.name, model_key, n_student, missing, unexpected)
    return {
        "checkpoint_key": model_key,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "n_tensors_in_model": len(model_state),
        "n_tensors_loaded": n_loaded,
        "n_student_tensors": n_student,
        "non_student_keys_dropped": dropped_non_student,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


# --------------------------------------------------------------------------- #
# head (EEG-FM-Compass utils/model_layers.py::LinearLayers, MIT)
# --------------------------------------------------------------------------- #
class LinearLayers(nn.Module):
    """``LinearLayers(input_dim, output_dim, flatten, dropout_rate)`` of the benchmark:
    optional ``flatten(start_dim)`` -> ``Dropout`` -> ``Linear``. Same module names
    (``clshead.0`` / ``clshead.1``) so a benchmark head state_dict maps 1:1."""

    def __init__(self, input_dim: int, output_dim: int, flatten: int = 0,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.clshead = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(input_dim, output_dim))
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten:
            x = x.flatten(self.flatten)
        return self.clshead(x)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class LaBraMClassifier(nn.Module):
    """LaBraM backbone + benchmark linear head under the ``{logits, features, confidence}`` contract.

    ``features`` is the tensor the head consumes: with ``readout="flatten"`` the fc-normed
    tokens flattened to ``[B, (1 + C_kept * n_segments) * 200]`` (before the head's dropout).
    """

    def __init__(self, config: Optional[LaBraMConfig] = None):
        super().__init__()
        cfg = config or LaBraMConfig()
        self.config = cfg

        if cfg.arch not in ARCHS:
            raise ValueError(f"labram: unknown arch '{cfg.arch}' (choose from {sorted(ARCHS)})")
        if cfg.readout not in ("flatten", "pooling"):
            raise ValueError(f"labram: readout must be 'flatten' or 'pooling', got {cfg.readout!r}")

        # ---- channels -> pos_embed ids ------------------------------------------------------
        if not cfg.channel_set:
            raise ValueError(
                "labram needs params['channel_set'] (the electrode list to look up in LaBraM's "
                "standard_1020 vocabulary). It is injected by code/configs/models/labram.yaml "
                "`per_dataset` for data.name in {wbci_shu_2c, wbci_shu_3c, shu}; add an entry "
                "for a new dataset instead of hard-coding names."
            )
        names = dataset_channel_names(cfg.channel_set)      # raises for an unknown set
        if len(names) != int(cfg.n_channels):
            raise ValueError(
                f"labram: channel_set '{cfg.channel_set}' lists {len(names)} electrodes but the "
                f"data has n_channels={cfg.n_channels}; the yaml per_dataset entry is wrong."
            )
        keep_idx, input_chans, kept, dropped = labram_channel_plan(names)
        if not keep_idx:
            raise ValueError(f"labram: none of the channels of '{cfg.channel_set}' is in standard_1020")
        if dropped:
            logger.warning("labram: %d/%d channels not in standard_1020 are dropped: %s",
                           len(dropped), len(names), dropped)
        self.channel_names: List[str] = list(names)
        self.keep_idx: List[int] = list(keep_idx)
        self.input_chans: List[int] = list(input_chans)     # plain list, like the benchmark
        self.channels_kept: List[str] = kept
        self.channels_dropped: List[str] = dropped
        self._select_channels = len(keep_idx) != len(names)

        # ---- time axis -> 1 s segments at 200 Hz ---------------------------------------------
        self.native_len = int(round(int(cfg.n_times) * float(cfg.native_sfreq) / float(cfg.sfreq)))
        if self.native_len % int(cfg.segment_len):
            raise ValueError(
                f"labram: {cfg.n_times} samples @ {cfg.sfreq} Hz resample to {self.native_len} @ "
                f"{cfg.native_sfreq} Hz, not a multiple of segment_len={cfg.segment_len}"
            )
        self.n_segments = self.native_len // int(cfg.segment_len)

        # ---- backbone, built exactly like EEG-FM-Compass Loader_LaBraM (create_model kwargs) --
        backbone: NeuralTransformer = ARCHS[cfg.arch](
            pretrained=False,
            num_classes=int(cfg.n_classes),
            drop_rate=float(cfg.drop_rate),
            drop_path_rate=float(cfg.drop_path_rate),
            attn_drop_rate=float(cfg.attn_drop_rate),
            drop_block_rate=None,                       # swallowed by **kwargs upstream
            use_mean_pooling=bool(cfg.use_mean_pooling),
            init_scale=float(cfg.init_scale),
            use_rel_pos_bias=bool(cfg.use_rel_pos_bias),
            use_abs_pos_emb=bool(cfg.use_abs_pos_emb),
            init_values=float(cfg.init_values),
            qkv_bias=bool(cfg.qkv_bias),
            num_t=self.n_segments,                      # swallowed by **kwargs upstream
        )
        if int(backbone.patch_size) != int(cfg.segment_len):
            raise ValueError(f"labram: arch patch_size={backbone.patch_size} != segment_len={cfg.segment_len}")
        if backbone.pos_embed is None:
            raise ValueError("labram: use_abs_pos_emb=False leaves no pos_embed to index with input_chans")
        if max(self.input_chans) >= int(backbone.pos_embed.shape[1]):
            raise ValueError(f"labram: input_chans max {max(self.input_chans)} exceeds pos_embed rows "
                             f"{backbone.pos_embed.shape[1]}")
        if self.n_segments > int(backbone.time_embed.shape[1]):
            raise ValueError(f"labram: {self.n_segments} segments exceed the {backbone.time_embed.shape[1]} "
                             f"rows of time_embed (max {backbone.time_embed.shape[1]} s per trial)")

        # ---- official weights -----------------------------------------------------------------
        self.weights_path: Optional[str] = None
        self.weights_sha256: Optional[str] = None
        self.load_report: Dict[str, object] = {}
        if cfg.pretrained:
            path = resolve_weights_path(cfg.weights_file)   # raises with download instructions
            digest = sha256_file(path)
            if cfg.weights_sha256 and digest.lower() != str(cfg.weights_sha256).lower():
                raise RuntimeError(
                    f"labram: sha256 of {path} is {digest}, expected {cfg.weights_sha256} "
                    "(wrong / corrupted checkpoint; refusing to train on it)"
                )
            self.weights_path, self.weights_sha256 = str(path), digest
            self.load_report = load_labram_checkpoint(backbone, path)

        backbone.head = nn.Identity()                       # benchmark: main_model.head = Identity
        self.backbone = backbone

        # ---- head ------------------------------------------------------------------------------
        self.embed_dim = int(backbone.embed_dim)
        self.n_tokens = 1 + len(self.keep_idx) * self.n_segments
        self.feature_dim = self.n_tokens * self.embed_dim if cfg.readout == "flatten" else self.embed_dim
        # = benchmark: num_channels * num_time_points(800) + 200 for the flatten readout
        self.head = LinearLayers(self.feature_dim, int(cfg.n_classes), flatten=1,
                                 dropout_rate=float(cfg.head_dropout))
        self.n_backbone_params = count_params(self.backbone)
        self.n_head_params = count_params(self.head)

    # ---- contract -----------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        x = squeeze_input(x)
        if x.shape[1] != int(self.config.n_channels):
            raise ValueError(f"labram: expected {self.config.n_channels} channels, got {x.shape[1]}")
        x = resample_time(x, float(self.config.sfreq), float(self.config.native_sfreq),
                          n_out=self.native_len)                          # [B, C, 800]
        if self._select_channels:
            x = x[:, self.keep_idx]                                       # [B, C_kept, 800]
        x = segment_trials(x, int(self.config.segment_len))               # [B, C_kept, 4, 200]
        tokens = self.backbone(x, input_chans=self.input_chans,
                               return_all_tokens=(self.config.readout == "flatten"))
        features = tokens.flatten(1)                                      # [B, 233*200] or [B, 200]
        logits = self.head(features)
        return {"logits": logits, "features": features, "confidence": None}

    def describe(self) -> Dict[str, object]:
        cfg = self.config
        return {
            "name": "labram",
            "arch": cfg.arch,
            "paper": PAPER,
            "source": f"{UPSTREAM_URL} @ {UPSTREAM_COMMIT} (modeling_finetune.py, vendored verbatim); "
                      f"wrapper follows EEG-FM-Compass Loader_LaBraM.py ({BENCHMARK_URL})",
            "n_params": count_params(self),
            "n_trainable_params": count_params(self, trainable_only=True),
            "n_backbone_params": self.n_backbone_params,
            "n_head_params": self.n_head_params,
            "feature_dim": self.feature_dim,
            "readout": cfg.readout,
            "n_tokens": self.n_tokens,
            "embed_dim": self.embed_dim,
            "depth": self.backbone.get_num_layers(),
            "num_heads": int(self.backbone.blocks[0].attn.num_heads),
            "pretrained": bool(cfg.pretrained),
            "pretrained_file": self.weights_path,
            "pretrained_sha256": self.weights_sha256,
            **{k: v for k, v in self.load_report.items()},
            "native_sfreq": float(cfg.native_sfreq),
            "native_length": self.native_len,
            "n_segments": self.n_segments,
            "segment_len": int(cfg.segment_len),
            "channel_set": cfg.channel_set,
            "n_channels_in": int(cfg.n_channels),
            "n_channels_kept": len(self.keep_idx),
            "channels_kept": list(self.channels_kept),
            "channels_dropped": list(self.channels_dropped),
            "input_chans": list(self.input_chans),
            "drop_rate": float(cfg.drop_rate),
            "drop_path_rate": float(cfg.drop_path_rate),
            "attn_drop_rate": float(cfg.attn_drop_rate),
            "head_dropout": float(cfg.head_dropout),
            "qkv_bias": bool(cfg.qkv_bias),
            "use_mean_pooling": bool(cfg.use_mean_pooling),
            "init_values": float(cfg.init_values),
            "amplitude_scaling": "none (protocol P4; upstream 0.1 mV convention not applied)",
        }


# --------------------------------------------------------------------------- #
# entry point used by code/models/external/__init__.py::build_external
# --------------------------------------------------------------------------- #
_DATA_DIM_KEYS = {"n_channels", "n_times", "n_classes", "sfreq"}


def build(*, n_channels: int, n_times: int, n_classes: int, sfreq: float,
          params: Optional[Dict] = None) -> LaBraMClassifier:
    known = {f.name for f in fields(LaBraMConfig)} - _DATA_DIM_KEYS
    p = warn_unknown_params(known, dict(params or {}), "labram")
    cfg = LaBraMConfig(n_channels=int(n_channels), n_times=int(n_times), n_classes=int(n_classes),
                       sfreq=float(sfreq), **p)
    return LaBraMClassifier(cfg)


__all__ = [
    "ARCHS", "BENCHMARK_URL", "DEFAULT_WEIGHTS_FILE", "DEFAULT_WEIGHTS_SHA256", "DEFAULT_WEIGHTS_SIZE",
    "EXPECTED_MISSING_SUBSTRINGS", "EXPECTED_UNEXPECTED_KEYS", "LaBraMClassifier", "LaBraMConfig",
    "LinearLayers", "STANDARD_1020", "UPSTREAM_COMMIT", "UPSTREAM_URL", "build",
    "labram_channel_plan", "load_labram_checkpoint", "segment_trials",
]
