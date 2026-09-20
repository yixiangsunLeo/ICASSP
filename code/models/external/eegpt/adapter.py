"""EEGPT large (Wang et al., NeurIPS 2024) as the model entry ``eegpt``.

What runs here is the **official pretrained encoder** ``EEGTransformer`` from
``downstream/Modules/models/EEGPT_mcae.py`` of https://github.com/BINE022/EEGPT (vendored
verbatim under ``_official/``), fine-tuned end to end behind the same three pieces the
EEG-FM-Compass benchmark (Liu et al., NSR 2026, ``models/FM/EEGPT/Loader_EEGPT.py``) puts
around it:

  1. ``Conv1dWithConstraint(n_kept, n_kept, 1, max_norm=1)`` — a 1x1 channel mixer (the
     official downstream scripts use the same layer, ``Modules/Network/utils.py``);
  2. the encoder, told which electrodes it sees through ``chan_ids`` looked up in its own
     62-name vocabulary ``CHANNEL_DICT``;
  3. a ``flatten`` readout of the encoder output ``[B, 16 patches, 4 summary tokens, 512]``
     -> ``[B, 32768]`` followed by ``Dropout(0.5) -> Linear`` (Compass ``LinearLayers``).

Format conversion only (protocol P4, ``docs/PROTOCOL_external_baselines.md``): the shared
per-trial z-scored trial ``[B, C, 1000] @ 250 Hz`` is (a) reduced to the electrodes EEGPT can
name — on SHU that drops the earlobes A1/A2 and renames T3/T4/T5/T6 -> T7/T8/P7/P8 — and
(b) resampled 250 -> 256 Hz to 1024 samples = 16 patches of 64. No filtering, no amplitude
scaling, and **no Euclidean alignment** (the Compass EEGPT preprocessor applies EA by default;
model-specific alignment is forbidden here). Everything is trainable (full fine-tuning) and
the loss is the trainer's plain cross-entropy.

Weights: ``weights/eegpt.ckpt`` (``target_encoder.*`` loaded with ``strict=True``). The file is
a conversion of the Hugging Face mirror ``eeg-telecom-paris/eegpt-large-official`` and has not
been compared tensor-wise with the official figshare checkpoint — see ``PROVENANCE.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
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
from ._official.downstream.Modules.models.EEGPT_mcae import CHANNEL_DICT, EEGTransformer
from ._official.downstream.Modules.Network.utils import Conv1dWithConstraint

logger = get_logger("models.external.eegpt")

MODEL_NAME = "eegpt"
UPSTREAM_URL = "https://github.com/BINE022/EEGPT"
UPSTREAM_COMMIT = "a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705"
PAPER = ("G. Wang et al., EEGPT: Pretrained Transformer for Universal and Reliable "
         "Representation of EEG Signals, NeurIPS 2024")
#: sha256 of ``weights/eegpt.ckpt`` (``weights/MANIFEST.yaml``); the adapter refuses any other file
#: unless ``weights_sha256`` is set to ``None``.
WEIGHTS_SHA256 = "f406ec002fd56f618f323c170b9c9e68997ea3299b33575611cb3e26fb1be273"
WEIGHTS_PROVENANCE = (
    "conversion of the Hugging Face mirror eeg-telecom-paris/eegpt-large-official "
    "(safetensors, same module structure as the official eegpt_mcae_58chs_4s_large4E.ckpt) "
    "into the Lightning {'state_dict': ...} layout by ../Benchmark/scripts/"
    "convert_eegpt_safetensors.py; NOT compared tensor-wise with the figshare original"
)
#: Prefix of the pretrained encoder inside the Lightning checkpoint (official loader:
#: ``target_encoder_stat[k[15:]] = v for k starting with 'target_encoder.'``).
TARGET_ENCODER_PREFIX = "target_encoder."


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class EEGPTConfig:
    """Every structural hyperparameter of the entry. Defaults = the official large model
    (``eegpt_mcae_58chs_4s_large4E``) exactly as EEG-FM-Compass ``Loader_EEGPT.py`` builds it."""

    # data dims (injected by the experiment layer, never hard-coded)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    # electrode names of the cohort: ``code/configs/datasets/<channel_set>.yaml`` (injected by
    # ``per_dataset`` in ``code/configs/models/eegpt.yaml``). ``channel_names`` is an explicit
    # list that wins over ``channel_set`` (for a cohort that has no datasets yaml yet).
    channel_set: Optional[str] = None
    channel_names: Optional[List[str]] = None

    # --- encoder: official large configuration (do not tune) ---
    embed_dim: int = 512
    embed_num: int = 4                 # summary tokens per time patch
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    init_std: float = 0.02
    norm_eps: float = 1e-6             # norm_layer = partial(nn.LayerNorm, eps=1e-6)
    patch_size: int = 64
    native_sfreq: float = 256.0
    n_native_times: int = 1024         # 4 s x 256 Hz = 16 patches of 64

    # --- fine-tuning wrapper (EEG-FM-Compass defaults) ---
    use_chan_conv: bool = True
    chan_conv_max_norm: float = 1.0
    readout: str = "flatten"           # 'flatten' (Compass default for EEGPT) | 'pooling'
    head_dropout: float = 0.5

    # --- pretrained weights ---
    pretrained: bool = True
    weights_file: str = "eegpt.ckpt"
    weights_sha256: Optional[str] = WEIGHTS_SHA256


# --------------------------------------------------------------------------- #
# pretrained weights
# --------------------------------------------------------------------------- #
_STATE_CACHE: Dict[Tuple[str, int, int], Tuple[Dict[str, torch.Tensor], str]] = {}


def load_target_encoder_state(path: Path, *, expected_sha256: Optional[str] = WEIGHTS_SHA256
                              ) -> Tuple[Dict[str, torch.Tensor], str]:
    """``target_encoder.*`` tensors of the Lightning checkpoint (prefix stripped) + file sha256.

    Mirrors the official loader (``linear_probe_EEGPT_*.py`` / Compass ``Loader_EEGPT.py``):
    ``{k[15:]: v for k, v in ckpt['state_dict'].items() if k.startswith('target_encoder.')}``.
    The digest is checked against ``expected_sha256`` (``None`` disables the check) so a
    wrong or corrupted download fails loudly instead of silently training a different model.
    Cached per file (path, mtime, size): one process reads the 408 MB file once.
    """
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key in _STATE_CACHE:
        state, digest = _STATE_CACHE[key]
    else:
        digest = sha256_file(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
        if not isinstance(sd, dict):
            raise ValueError(f"{path}: expected a Lightning checkpoint {{'state_dict': ...}}")
        state = {k[len(TARGET_ENCODER_PREFIX):]: v for k, v in sd.items()
                 if k.startswith(TARGET_ENCODER_PREFIX)}
        if not state:
            raise ValueError(f"{path}: no '{TARGET_ENCODER_PREFIX}*' tensors "
                             f"(top-level prefixes: {sorted({k.split('.')[0] for k in sd})})")
        _STATE_CACHE[key] = (state, digest)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{path}: sha256 {digest} != expected {expected_sha256}. This is not the EEGPT "
            "checkpoint the results were produced with (see PROVENANCE.md); set "
            "weights_sha256: null in the model params only if you know what you are doing."
        )
    return state, digest


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class EEGPTClassifier(nn.Module):
    """Official EEGPT encoder + Compass fine-tuning wrapper behind the project contract.

    ``forward(x)``: ``x`` is ``[B, C, T]`` (or ``[B, 1, C, T]``) at the cohort's rate; returns
    ``{"logits": [B, n_classes], "features": [B, feature_dim], "confidence": None}`` where
    ``features`` is the flattened encoder output fed to the head.
    """

    def __init__(self, config: Optional[EEGPTConfig] = None):
        super().__init__()
        cfg = config or EEGPTConfig()
        self.config = cfg

        if cfg.n_native_times % cfg.patch_size != 0:
            raise ValueError(f"n_native_times={cfg.n_native_times} must be a multiple of "
                             f"patch_size={cfg.patch_size}")
        if cfg.readout not in ("flatten", "pooling"):
            raise ValueError(f"readout must be 'flatten' or 'pooling', got {cfg.readout!r}")
        n_resampled = int(round(cfg.n_times * float(cfg.native_sfreq) / float(cfg.sfreq)))
        if n_resampled != cfg.n_native_times:
            logger.warning("eegpt: %d samples @ %.1f Hz resample to %d @ %.1f Hz, which will be "
                           "cropped/zero-padded to %d", cfg.n_times, cfg.sfreq, n_resampled,
                           cfg.native_sfreq, cfg.n_native_times)
        self.n_resampled_times = n_resampled

        # ---- electrodes: which of ours does EEGPT know? -------------------------------
        names = self._resolve_channel_names(cfg)
        keep_idx, matched, dropped = match_channels(names, CHANNEL_DICT.keys())
        if not keep_idx:
            raise ValueError(f"none of the electrodes {names} is in EEGPT's CHANNEL_DICT")
        if dropped:
            logger.info("eegpt: %d/%d electrodes not in EEGPT's vocabulary, dropped: %s",
                        len(dropped), len(names), dropped)
        self.input_channel_names: List[str] = [str(n) for n in names]
        self.kept_channel_names: List[str] = list(matched)
        self.dropped_channel_names: List[str] = list(dropped)
        self.n_kept = len(keep_idx)
        self.n_patches = cfg.n_native_times // cfg.patch_size
        self.register_buffer("keep_idx", torch.as_tensor(keep_idx, dtype=torch.long),
                             persistent=False)
        self._select_channels = keep_idx != list(range(cfg.n_channels))

        # ---- official encoder (Loader_EEGPT.py / linear_probe_EEGPT_*.py construction) --
        self.encoder = EEGTransformer(
            img_size=[self.n_kept, cfg.n_native_times],
            patch_size=cfg.patch_size,
            embed_num=cfg.embed_num,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            init_std=cfg.init_std,
            qkv_bias=cfg.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=cfg.norm_eps),
        )
        self.weights_path: Optional[str] = None
        self.weights_sha256: Optional[str] = None
        if cfg.pretrained:
            path = resolve_weights_path(cfg.weights_file)
            state, digest = load_target_encoder_state(path, expected_sha256=cfg.weights_sha256)
            # strict=True: the vendored encoder and the checkpoint must agree key for key.
            self.encoder.load_state_dict(state, strict=True)
            self.weights_path, self.weights_sha256 = str(path), digest

        # channel ids in EEGPT's vocabulary, exactly `EEGTransformer.prepare_chan_ids`
        self.register_buffer("chans_id", self.encoder.prepare_chan_ids(self.kept_channel_names),
                             persistent=False)                                   # [1, n_kept]

        # ---- wrapper: 1x1 channel mixer -> encoder -> readout -> dropout + linear ------
        if cfg.use_chan_conv:
            self.chan_conv: nn.Module = Conv1dWithConstraint(
                self.n_kept, self.n_kept, 1, max_norm=cfg.chan_conv_max_norm)
        else:
            self.chan_conv = nn.Identity()

        # feature_dim by a dry forward at batch 1, like the Compass loader (zeros instead of
        # randn so construction consumes no RNG; only the shape is used).
        with torch.no_grad():
            was_training = self.training
            self.eval()
            x0 = torch.zeros(1, self.n_kept, cfg.n_native_times, device=self.chans_id.device)
            self.feature_dim = int(self._readout(self._encode(x0)).shape[1])
            self.train(was_training)

        self.head = nn.Sequential(nn.Dropout(cfg.head_dropout),
                                  nn.Linear(self.feature_dim, cfg.n_classes))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_channel_names(cfg: EEGPTConfig) -> List[str]:
        if cfg.channel_names is not None:
            names = [str(c) for c in cfg.channel_names]
            origin = "channel_names"
        elif cfg.channel_set:
            names = dataset_channel_names(cfg.channel_set)      # raises if the set is unknown
            origin = f"channel_set '{cfg.channel_set}'"
        else:
            raise ValueError(
                "eegpt needs the electrode names to look up EEGPT's channel vocabulary: set "
                "`channel_set` ('wbci_shu' or 'shu') — normally injected by "
                "per_dataset.<data.name>.channel_set in code/configs/models/eegpt.yaml (data.name "
                "must be one of wbci_shu_2c / wbci_shu_3c / shu) — or pass `channel_names`."
            )
        if len(names) != int(cfg.n_channels):
            raise ValueError(
                f"eegpt: {origin} lists {len(names)} electrodes but n_channels={cfg.n_channels}; "
                "the channel set must describe exactly the channels of the input tensor."
            )
        return names

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, n_kept, 1024]`` at 256 Hz -> encoder output ``[B, 16, embed_num, embed_dim]``."""
        x = self.chan_conv(x)
        return self.encoder(x, self.chans_id)

    def _readout(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.readout == "pooling":
            return z.mean(dim=(1, 2))                       # [B, embed_dim]
        return z.flatten(1)                                 # [B, N * embed_num * embed_dim]

    def to_native_format(self, x: torch.Tensor) -> torch.Tensor:
        """Format conversion only: ``[B, C, T] @ sfreq`` -> ``[B, n_kept, 1024] @ 256 Hz``.

        Electrodes outside EEGPT's vocabulary are removed (their signal is not mixed into
        the others) and the time axis is sinc-resampled to the pretraining rate.
        """
        x = squeeze_input(x)
        if x.shape[1] != self.config.n_channels:
            raise ValueError(f"eegpt: expected {self.config.n_channels} channels "
                             f"({self.input_channel_names[:3]}...), got {x.shape[1]}")
        if self._select_channels:
            x = x.index_select(1, self.keep_idx)
        return resample_time(x, self.config.sfreq, self.config.native_sfreq,
                             n_out=self.config.n_native_times)

    # ------------------------------------------------------------------ contract
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        x = self.to_native_format(x)
        features = self._readout(self._encode(x))
        logits = self.head(features)
        return {"logits": logits, "features": features, "confidence": None}

    def describe(self) -> Dict[str, object]:
        cfg = self.config
        tokens_per_patch = self.n_kept + cfg.embed_num
        return {
            "name": MODEL_NAME,
            "source": (f"{UPSTREAM_URL} @ {UPSTREAM_COMMIT} "
                       "(downstream/Modules/models/EEGPT_mcae.py::EEGTransformer, "
                       "downstream/Modules/Network/utils.py::Conv1dWithConstraint); "
                       "fine-tuning wrapper after EEG-FM-Compass models/FM/EEGPT/Loader_EEGPT.py"),
            "paper": PAPER,
            "n_params": count_params(self),
            "n_trainable_params": count_params(self, trainable_only=True),
            "n_params_encoder": count_params(self.encoder),
            "n_params_chan_conv": count_params(self.chan_conv),
            "n_params_head": count_params(self.head),
            "feature_dim": self.feature_dim,
            "readout": cfg.readout,
            "head_dropout": cfg.head_dropout,
            "pretrained": bool(cfg.pretrained),
            "pretrained_file": self.weights_path,
            "pretrained_sha256": self.weights_sha256,
            "pretrained_provenance": WEIGHTS_PROVENANCE if cfg.pretrained else None,
            "pretrained_prefix": TARGET_ENCODER_PREFIX,
            "strict_load": bool(cfg.pretrained),
            "n_channels_in": cfg.n_channels,
            "n_channels_kept": self.n_kept,
            "channels_kept": list(self.kept_channel_names),
            "channels_dropped": list(self.dropped_channel_names),
            "chans_id": self.chans_id.flatten().tolist(),
            "input_sfreq": float(cfg.sfreq),
            "n_times_in": cfg.n_times,
            "native_sfreq": float(cfg.native_sfreq),
            "n_native_times": cfg.n_native_times,
            "n_resampled_times": self.n_resampled_times,
            "patch_size": cfg.patch_size,
            "n_patches": self.n_patches,
            "embed_dim": cfg.embed_dim,
            "embed_num": cfg.embed_num,
            "depth": cfg.depth,
            "num_heads": cfg.num_heads,
            "tokens_per_patch": tokens_per_patch,
            "tokens_per_trial": self.n_patches * tokens_per_patch,
            "alignment": "none (protocol P4: no model-specific EA)",
        }


# --------------------------------------------------------------------------- #
# entry point used by code.models.external.build_external
# --------------------------------------------------------------------------- #
def build(*, n_channels: int, n_times: int, n_classes: int, sfreq: float,
          params: Optional[Dict] = None) -> EEGPTClassifier:
    known = [f.name for f in dataclasses.fields(EEGPTConfig)]
    p = warn_unknown_params(known, dict(params or {}), MODEL_NAME)
    for k in ("n_channels", "n_times", "n_classes", "sfreq"):
        p.pop(k, None)                       # data dims come from the experiment layer only
    cfg = EEGPTConfig(n_channels=int(n_channels), n_times=int(n_times),
                      n_classes=int(n_classes), sfreq=float(sfreq), **p)
    return EEGPTClassifier(cfg)


__all__ = [
    "MODEL_NAME", "UPSTREAM_URL", "UPSTREAM_COMMIT", "WEIGHTS_SHA256", "TARGET_ENCODER_PREFIX",
    "EEGPTConfig", "EEGPTClassifier", "load_target_encoder_state", "build", "CHANNEL_DICT",
]
