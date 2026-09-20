# PROVENANCE — CBraMod (`cbramod`)

Paper: J. Wang, S. Zhao, Z. Luo, Y. Zhou, H. Jiang, S. Li, T. Li, G. Pan,
*CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*, ICLR 2025
(<https://openreview.net/forum?id=NPNUHgHF2w>).

## 1. Upstream code

| item | value |
|:--|:--|
| repository | <https://github.com/wjq-learning/CBraMod> |
| commit | `b9e961003214326972c567eff390e75b0287e32a` (2026-08-07 16:42:58 +0800, "update") |
| license | MIT, Copyright (c) 2025 Jiquan Wang — copied verbatim to `_official/LICENSE` |
| local read-only clone used for vendoring | `../Benchmark/third_party/cbramod/` (git submodule of the sister project, same commit) |

### Vendored files (`code/models/external/cbramod/_official/`)

| vendored file | upstream path | sha256 of the **upstream** file | sha256 of the **vendored** copy | changed? |
|:--|:--|:--|:--|:--|
| `cbramod.py` | `models/cbramod.py` | `caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14` | `fad4affab288c00402fbe83d58d6a56a99b6cc80a1e70b9b9aaa836e3d963fdd` | **1 import line** (below) |
| `criss_cross_transformer.py` | `models/criss_cross_transformer.py` | `90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c` | identical | no |
| `LICENSE` | `LICENSE` | `eb839b706e4d7219842e96d7dc33285844abc794292602deb3498be7b699537c` | identical | no |
| `__init__.py` | `models/__init__.py` (empty file) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | identical | no |

The only edit to upstream code, `cbramod.py` line 5 (import path; the vendored package is
`code.models.external.cbramod._official`, not the upstream top-level `models` package):

```diff
-from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
+from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder  # vendored: was `from models.criss_cross_transformer import ...`
```

`tests/models/test_cbramod.py::test_vendored_files_match_upstream` re-hashes the vendored copies
against the upstream hashes above (for `cbramod.py` after undoing that single line), so any later
drift from upstream fails a test. The `if __name__ == '__main__':` demo blocks in both files
(which call `.cuda()`) are kept verbatim and never executed.

Not vendored (not needed for fine-tuning): `models/model_for_*.py` (upstream's 13 task-specific
heads; the SHU-MI one, `model_for_shu.py`, is a **within-dataset** classifier for the same SHU
2022 data hard-wired to 32 channels and one logit — our read-out is described in §3),
`finetune_*.py`, `pretrain_*.py`, `datasets/`, `preprocessing/`, `utils/`.

### Differences between upstream and the EEG-FM-Compass copy (reference wrapper, **not** vendored)

The reference wrapper `../Benchmark/third_party/eeg_fm_benchmark/models/FM/CBraMod/` (Liu et al.,
*NSR* 2026, MIT) was diffed against upstream `models/cbramod.py` + `models/criss_cross_transformer.py`
(`diff` on 2026-09-19); we vendor the **upstream** files. Differences found:

| file | Compass copy vs upstream | effect on numbers |
|:--|:--|:--|
| `Model_CBraMod.py` | adds `sys.path` manipulation + import `models.FM.CBraMod.criss_cross_transformer` | none (imports) |
| `Model_CBraMod.py` | `torch.abs(spectral)` (upstream) replaced by `torch.sqrt(spectral.real**2 + spectral.imag**2)` | none in fp32 forward (identical magnitude); Compass presumably did it for ONNX/complex-grad friendliness |
| `Model_CBraMod.py` | deletes two commented `print` lines and the `__main__` demo | none |
| `criss_cross_transformer.py` | adds `import pdb`, a shape comment, deletes the `__main__` demo | none |

So the Compass backbone is numerically the upstream backbone; only its **wrapper** conventions
(`Loader_CBraMod.py`) are adopted here: `CBraMod()` defaults, `load_state_dict(strict=True)`,
`proj_out = nn.Identity()`, `x.reshape(b, c, -1, 200)`, flatten read-out
`feature_dim = num_time_points(800) * num_channels`, head
`LinearLayers(feature_dim, n_classes, flatten=1, dropout_rate=0.1)` = `Dropout(0.1) -> Linear`
(`utils/model_layers.py`), full fine-tuning (`finetune_strategy='full'`).

## 2. Pretrained weights

| item | value |
|:--|:--|
| file used here | `weights/CBraMod.pth` (resolved through `resolve_weights_path`; `weights/` is git-ignored — on this cluster a symlink to `../Benchmark/weights/`) |
| origin | Hugging Face `weighting666/CBraMod`, file `pretrained_weights.pth` (the file `quick_example.py` / `finetune_main.py --foundation_dir` load); downloaded 2026-09-14 via `hf-mirror.com`, recorded in `weights/MANIFEST.yaml` |
| size | 19,775,842 bytes |
| sha256 | `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178` |
| content | plain `OrderedDict` state_dict, 211 tensors, 4,924,000 fp32 values, no `module.` prefix; loads with `torch.load(weights_only=True)` |
| how loaded | `CBraMod(**upstream defaults).load_state_dict(state, strict=True)` -> 0 missing / 0 unexpected keys (asserted in `test_pretrained_weights_load_strictly`), **then** `proj_out -> nn.Identity()` |
| integrity | the adapter recomputes the sha256 at construction and raises on mismatch (`weights_sha256: null` bypasses deliberately) |
| pre-training corpus | TUEG (Temple University Hospital EEG corpus) — contains neither SHU 2022 nor WBCIC-SHU, so no leakage into our cohorts |

The checkpoint includes `proj_out.0.{weight,bias}` (the 200x200 pre-training reconstruction head,
40,200 values) and `patch_embedding.mask_encoding` (200 zeros, the masked-patch token). Both are
loaded strictly; `proj_out` is then discarded exactly as upstream's downstream models do; the mask
token is never used in fine-tuning (`mask=None`).

## 3. Hyperparameters: upstream default vs value used here

| parameter | upstream default (`CBraMod.__init__` / fine-tuning scripts) | used here | note |
|:--|:--|:--|:--|
| `in_dim` | 200 | 200 | samples per patch |
| `out_dim` | 200 | 200 | width of `proj_out` (discarded) |
| `d_model` | 200 | 200 | token width, split 100 (spatial) + 100 (temporal) |
| `dim_feedforward` | 800 | 800 | |
| `seq_len` | 30 | 30 | accepted by `PatchEmbedding.__init__`, **unused** by upstream code (no learned position table; positions come from the 19x7 depthwise conv) |
| `n_layer` | 12 | 12 | |
| `nhead` | 8 | 8 | 4 spatial + 4 temporal heads per layer |
| patch length / rate | 1 s @ 200 Hz (pre-training) | 200 samples @ 200 Hz | our 1000 @ 250 Hz -> `resample_time` (torchaudio windowed-sinc) -> 800 @ 200 Hz -> 4 patches |
| patches per trial | 30 (pre-training, 30 s), 4 (upstream SHU-MI / BCIC-IV-2a fine-tuning, 4 s) | 4 | same 4 s trials as upstream's MI tasks |
| amplitude | upstream `shu_dataset.py` divides by 100 (µV -> 100 µV units) | **no scaling** | protocol P4: input is the shared per-trial z-scored trial |
| filtering | upstream preprocessing / Compass 4–32 Hz + notch | **none** | protocol P4 |
| channels | upstream SHU-MI head hard-wires 32 ch; Compass uses all dataset channels | all C (58 or 32) | channel-agnostic backbone; `n_channels` only sizes the read-out |
| read-out | upstream default `--classifier all_patch_reps` = flatten -> Linear(C·4·200, 800) -> ELU -> Dropout(0.1) -> Linear(800, 200) -> ELU -> Dropout(0.1) -> Linear(200, n_out) | **flatten -> Dropout(0.1) -> Linear(C·4·200, n_classes)** | EEG-FM-Compass `LinearLayers` head, the common linear-head convention of protocol P5 for all foundation-model entries |
| head dropout | 0.1 | 0.1 | |
| binary output | upstream SHU-MI uses 1 logit + `BCEWithLogitsLoss` | `n_classes` logits + cross-entropy | project contract (all entries) |
| fine-tuning | full (`frozen=False`) | full | every parameter `requires_grad=True` |
| optimizer / lr | AdamW, backbone lr 1e-4, head lr 1e-3·sqrt(bs/256), weight decay 5e-2, cosine to 1e-6, grad-clip 1, label smoothing 0.1 (multi-class), 50 epochs, batch 64 | **Adam 1e-4 (all params), wd 0, no scheduler, no clipping, batch 128, max 500 epochs, patience 25 on val macro-F1** | protocol P3: one shared benchmark recipe for every entry, deliberately not the per-paper recipe |
| seed | 3407 | training seed 0, split seeds 41/42 | protocol P1/P3 |

## 4. Parameter counts (exact, from `describe()`; asserted in tests)

| configuration | backbone (`proj_out` -> Identity) | head | total |
|:--|--:|--:|--:|
| 58 ch / 2 classes (WBCIC-SHU 2C) | 4,883,800 | 92,802 | **4,976,602** |
| 58 ch / 3 classes (WBCIC-SHU 3C) | 4,883,800 | 139,203 | **5,023,003** |
| 32 ch / 2 classes (SHU 2022) | 4,883,800 | 51,202 | **4,935,002** |
| checkpoint as released (backbone + `proj_out`) | 4,924,000 | — | — |

Per encoder layer: 402,600 (2 x MHA(100, 4 heads) = 80,800; FFN 200->800->200 = 321,000;
2 x LayerNorm = 800); x12 = 4,831,200. Patch embedding: 52,600 (depthwise 19x7 positional conv
26,800; `proj_in` conv stack 5,200; `spectral_proj` Linear(101, 200) 20,400; mask token 200).
The paper's "≈ 4.0 M" is therefore an under-statement of the released checkpoint (4.88 M without
the reconstruction head); we report the exact counts.

## 5. Adapter contract (what `adapter.py` adds, none of it upstream)

* `forward(x[B, C, 1000] or [B, 1, C, 1000])` -> `{"logits": [B, n_classes], "features": [B, C·4·200], "confidence": None}`;
  `features` is the flattened token tensor the head consumes.
* Construction runs a dry forward (`zeros(1, C, 1000)`, eval mode, no grad) and asserts the
  flattened size equals `C·4·d_model` before building the head.
* `describe()` reports counts, feature dim, conversion geometry (native 200 Hz / 800 samples /
  4 patches / C·4 tokens), `channels_kept = C`, `channels_dropped = 0`, and the checkpoint facts
  (file, sha256, 211 tensors, 0 missing / 0 unexpected keys).
* `default_lr() = 1e-4` (upstream backbone lr; only used if an experiment leaves `train.lr` empty).
