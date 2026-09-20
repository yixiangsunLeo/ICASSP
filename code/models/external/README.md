---
title: "External published baselines (TCFormer / LaBraM / EEGPT / CBraMod) — official code, one shared protocol"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-09-19"
updated: "2026-09-20"
status: "active"
---

# `code/models/external/` — 2024–2025 baselines run under the shared cross-subject protocol

Purpose: add recent published models to the same experiment as the 18-entry ANN benchmark and the
spiking main run — same subject folds, same trials, same per-trial z-score, same recipe (Adam 1e-4,
batch 128, patience 25 on validation macro-F1, training seed 0, best-validation checkpoint) — so that
every entry is paired per (split, fold, subject) with every other one. Protocol and decision record:
`docs/PROTOCOL_external_baselines.md`, `docs/DECISIONS.md` D-034.

Hard rule, as for `paper_baselines/`: **only the original authors' code.** Upstream files are vendored
verbatim under `<name>/_official/` with their LICENSE; `<name>/PROVENANCE.md` records the upstream URL,
commit, per-file sha256, every edited line (import paths only) and the pretrained-weight provenance.
`<name>/adapter.py` only wraps the upstream module into this project's contract
`forward(x[B, C, T]) -> {"logits", "features", "confidence"}` + `describe()`.

| entry | paper | upstream | pretrained weights | input adapter (format only) |
|:---|:---|:---|:---|:---|
| `tcformer` | Altaheri, Karray, Karimi, *Sci Rep* 15:32959, 2025 | github.com/altaheri/TCFormer (MIT), paper config `configs/tcformer.yaml` | none (trained from scratch) | none (`[B, C, 1000]` as is) |
| `labram` | Jiang, Zhao, Lu, ICLR 2024 | github.com/935963004/LaBraM `modeling_finetune.py` (`labram_base_patch200_200`) | `labram-base.pth` (5.8 M) | 250→200 Hz, 4 × 200-sample segments, channel ids from `standard_1020` |
| `eegpt` | Wang et al., NeurIPS 2024 | github.com/BINE022/EEGPT (`EEGTransformer`, large: 512×4, depth 8, patch 64) | `eegpt.ckpt` (`target_encoder.*`, 25 M) | 250→256 Hz (1024 samples), channel ids from EEGPT's 62-name vocabulary; SHU: T3/T4/T5/T6→T7/T8/P7/P8, A1/A2 dropped |
| `cbramod` | Wang et al., ICLR 2025 | github.com/wjq-learning/CBraMod | `CBraMod.pth` (4.0 M) | 250→200 Hz, `[B, C, 4, 200]` patches (channel-agnostic) |
| `adahgnn` | AdaHGNN / EEG-AdaMSHyper (lab, unpublished at export) | `/share/home/yuan/LY/DeepHypergraph/project_V4` (2026-01-16); runtime = unmodified `_official/models.py::EEG_AdaMSHyper` (`dhg` 0.9.7). Registered, not run | none (from scratch) | none (`[B, C, 1000]` as is). 58-ch pyramid `[58,29,1]` is the only Config-table extension |
| `adahgnn@mi`, `adahgnn@mi_mix` | **AdaHGNN-MI** — the MI adaptation of the above (D-040; `adahgnn/mi_model.py`; design `docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`) | same lab source; shared 125/63 filter bank, differentiable top-k hypergraph, signal-domain Laplacian spatial filter, hemisphere-aware anatomical hierarchy (`per_dataset.channel_set`), node-identity read-out; `mi_mix` adds same-class Dirichlet mixup | none (from scratch) | none (`[B, C, 1000]` as is) |

Pretrained models are **fully fine-tuned** with a fresh linear head (the EEG-FM-Compass protocol). No
model-specific filtering, amplitude scaling or alignment is applied: what differs between entries is
the architecture (and, for the three foundation models, their pretraining), nothing else.

## Weights

Checkpoints are not in git. The adapters look for them in (first hit wins) `$ICASSP_WEIGHTS_DIR`,
`code/configs/paths.local.yaml: pretrained_weights_dir`, then `<repo>/weights/`. On the development
cluster `weights/` is a symlink to the sister project's `../Benchmark/weights/`.

| file | sha256 | origin |
|:---|:---|:---|
| `labram-base.pth` | `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c` | official repo `checkpoints/labram-base.pth` (commit c431221) |
| `CBraMod.pth` | `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178` | Hugging Face `weighting666/CBraMod` `pretrained_weights.pth` |
| `eegpt.ckpt` | `f406ec002fd56f618f323c170b9c9e68997ea3299b33575611cb3e26fb1be273` | **conversion** of HF `eeg-telecom-paris/eegpt-large-official` `weights.safetensors` (sha256 `c45c31cb…755a96`) into the Lightning `{"state_dict": …}` layout; identical module structure to the official `eegpt_mcae_58chs_4s_large4E.ckpt` (figshare, browser-only); not compared tensor-wise → footnote in the paper |

## Adding another entry

`<name>/_official/` (verbatim + LICENSE) → `<name>/adapter.py` with
`build(*, n_channels, n_times, n_classes, sfreq, params)` → `code/configs/models/<name>.yaml`
(`defaults` = upstream/paper values; `per_dataset` only for channel sets) → append the name to
`EXTERNAL_MODEL_NAMES` in `__init__.py` → `tests/models/test_<name>.py` → regenerate the experiment
configs with `scripts/make_baseline_configs.py`. Shared helpers (resampling, channel vocabulary
matching, weight path resolution) live in `_common.py`.
