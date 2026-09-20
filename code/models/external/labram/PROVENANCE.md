# PROVENANCE — LaBraM-base (`labram`)

Paper: W.-B. Jiang, L.-M. Zhao, B.-L. Lu, *Large Brain Model for Learning Generic
Representations with Tremendous EEG Data in BCI*, ICLR 2024.

## 1. Upstream code

| item | value |
|:--|:--|
| repository | <https://github.com/935963004/LaBraM> |
| commit | `c431221e6cfd23dbfa9950e0180682fb322b0548` |
| license | MIT, Copyright (c) 2024 Weibang Jiang — `_official/LICENSE` |
| wrapper convention | EEG-FM-Compass `models/FM/LaBraM/Loader_LaBraM.py` (Liu et al., NSR 2026) = upstream `run_class_finetuning.py` construction + load |

### Vendored files (`code/models/external/labram/_official/`)

| vendored file | upstream | sha256 of vendored copy | changed? |
|:--|:--|:--|:--|
| `modeling_finetune.py` | `modeling_finetune.py` | `7b3514c9a661ebd417a2dd9834a9f2fab0a2e518c8212c73ebebaaa165c29d88` | **verbatim** |
| `standard_1020.py` | excerpt of `utils.py` L42–57 + `get_input_chans` L713–717 | `71c6b3fd5bab17df0f1ee26a6b98a1fe458eccec3f5f0728b79d86e62e472cb5` | excerpt only (header comment is ours) |
| `LICENSE` | `LICENSE` | `bc3f03e2c9071fb40507f5c575d95eeaa98d07c54c54fb2855942b2a729adc71` | no |
| `__init__.py` | — | ours (package marker) | — |

Full upstream `utils.py` is not vendored (tensorboardX / pyhealth / h5py training helpers). The electrode list and `get_input_chans` are the only pieces the fine-tune model needs.

## 2. Pretrained weights

| item | value |
|:--|:--|
| file | `weights/labram-base.pth` |
| origin | official repo `checkpoints/labram-base.pth` (same commit) |
| size | 96,612,769 bytes |
| sha256 | `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c` |
| load | student branch, non-strict; **219 / 247** tensors of the fine-tune model filled |
| expected missing | `head.*`, `fc_norm.*`, 12× `{q,v}_bias` (pretrain had `qkv_bias=False`; fine-tune zeros them) |
| expected unexpected | `lm_head.*`, `mask_token`, `norm.*` (pretraining-only) |
| corpus | TUEG + 15 public sets — **not** SHU 2022 / WBCIC-SHU |

Any unexplained missing/unexpected key raises. sha256 mismatch raises.

## 3. Format conversion (protocol P4 — adapter only)

`[B, C, 1000] @ 250 Hz` → sinc resample 200 Hz (800 samples) → 4 × 200-sample segments → `[B, C, 4, 200]`. Channel ids from `standard_1020` (id 0 = cls). All 58 WBCIC and all 32 SHU names (incl. A1/A2, T3–T6) are in the vocabulary — nothing dropped. **No** 0.1 mV scaling, **no** extra filter, **no** EA.

Read-out: Compass `LinearLayers` = flatten all tokens → Dropout(0.1) → Linear. Full fine-tuning. Shared recipe (Adam 1e-4, batch 128, patience 25).

## 4. Parameter counts (this tree)

| configuration | backbone (`head` → Identity) | head | total |
|:--|--:|--:|--:|
| 58 ch / 2 classes | 5,824,736 | 93,202 | **5,917,938** |
| 58 ch / 3 classes | 5,824,736 | 139,803 | **5,964,539** |
| 32 ch / 2 classes | 5,824,736 | 51,602 | **5,876,338** |

`feature_dim = (1 + C_kept × 4) × 200` (46600 / 25800). The paper's "5.8 M" is the backbone.
