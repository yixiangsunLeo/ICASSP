# PROVENANCE — EEGPT large (`eegpt`)

Paper: G. Wang et al., *EEGPT: Pretrained Transformer for Universal and Reliable
Representation of EEG Signals*, NeurIPS 2024.

## 1. Upstream code

| item | value |
|:--|:--|
| repository | <https://github.com/BINE022/EEGPT> |
| commit | `a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705` |
| license | Apache-2.0 — `_official/LICENSE` |
| wrapper convention | EEG-FM-Compass `models/FM/EEGPT/Loader_EEGPT.py` (1×1 `Conv1dWithConstraint`, flatten head, dropout 0.5) |

### Vendored files (`code/models/external/eegpt/_official/`)

| vendored file | upstream | sha256 of vendored copy | changed? |
|:--|:--|:--|:--|
| `downstream/Modules/models/EEGPT_mcae.py` | same path (`EEGTransformer`, `CHANNEL_DICT`) | `cfe3d786e6c351e9678f9f7183cd17dab0dfbb131590ef88e3e4a9efae728a95` | **verbatim** |
| `downstream/Modules/Network/utils.py` | same path (`Conv1dWithConstraint`) | `7d3b2289a4cd6d54eac3d33666bbd89d013eb8fc91a71b0630f2693d1e72aee2` | **verbatim** |
| `LICENSE` | `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | no |

`__init__.py` files under `downstream/` are **ours** (empty package markers). Upstream has no `__init__.py` there.

## 2. Pretrained weights

| item | value |
|:--|:--|
| file | `weights/eegpt.ckpt` |
| origin | conversion of HF `eeg-telecom-paris/eegpt-large-official` `weights.safetensors` (sha256 `c45c31cb42f68f8b9c630a7739a8084ae5dd6a6608a0d732a5165f180c755a96`) into Lightning `{"state_dict": …}` by `../Benchmark/scripts/convert_eegpt_safetensors.py` |
| ckpt sha256 | `f406ec002fd56f618f323c170b9c9e68997ea3299b33575611cb3e26fb1be273` |
| load | `target_encoder.*` with **`strict=True`** |
| figshare original | `eegpt_mcae_58chs_4s_large4E.ckpt` — **not** compared tensor-wise → **paper footnote** |
| corpus | PhysioMI / HGD / TSU / SEED / M3CV — **not** SHU 2022 / WBCIC-SHU |

sha256 mismatch raises.

## 3. Format conversion (protocol P4)

`[B, C, 1000] @ 250 Hz` → keep electrodes in EEGPT's 62-name `CHANNEL_DICT` (canonical 10-20: T3→T7, T4→T8, T5→P7, T6→P8) → sinc resample 256 Hz, 1024 samples = 16 patches of 64. **No** extra filter, **no** amplitude scaling, **no** Euclidean alignment (Compass EEGPT preprocessor defaults `apply_EA=True`; forbidden here).

| cohort | in | kept | dropped |
|:--|---:|---:|:--|
| WBCIC-SHU 2C/3C | 58 | 58 | none |
| SHU 2022 | 32 | 30 | A1, A2 (earlobes, not in vocab) |

Read-out: flatten `[B, 16, 4, 512]` → 32768 → Dropout(0.5) → Linear. Full fine-tuning. Shared recipe.

## 4. Parameter counts (this tree)

| configuration | n_params |
|:--|--:|
| 58 ch / 2 classes | 25,356,128 |
| 58 ch / 3 classes | 25,388,897 |
| 32 ch in / 30 kept / 2 classes | 25,353,636 |

Difference 2-class vs 3-class = 32,769 = 32768 + 1 (linear head). Difference 58 vs 30 kept = chan_conv `n²+n`.
