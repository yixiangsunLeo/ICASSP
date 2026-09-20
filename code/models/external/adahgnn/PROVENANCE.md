# PROVENANCE — AdaHGNN / EEG-AdaMSHyper (`adahgnn`)

Lab model, not a published GitHub snapshot. Senior pointer 2026-09-20:
`/share/home/yuan/LY/DeepHypergraph`. Latest *complete, modular* version is
`project_V4` (mtime 2026-01-16 16:15). The newest Python file in the tree is
`project_V3/AdaHGNN_AVED/AVED_multi/adahgnn_DANN_LOSO_0116_1.py` (21:46 the same
day) — same `EEG_AdaMSHyper` class, still a single-file AVED/DANN script. V4 is
what we export.

Two runtime paths (`adapter.py`, selected by the yaml `arch` key):

* `adahgnn` — the vendored class, unmodified (D-038). Registered, **not run** (user decision
  2026-09-20). An earlier grouped-conv / dense-v2v rewrite (`network.py`) was deleted before
  it trained.
* `adahgnn@mi` / `adahgnn@mi_mix` — **AdaHGNN-MI** (`mi_model.py`, D-040): the motor-imagery
  adaptation designed in `docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`
  (audit items A1–A9, tiers T0+T1+T2; mixup arm = T3.1; DANN and NHC off). ~0.1 M parameters.
  Its hypergraph aggregation is the dense form of `dhg.Hypergraph.v2v(aggr="mean")` (unit test
  against dhg, 1e-5).

## 1. Upstream code

| item | value |
|:--|:--|
| path | `/share/home/yuan/LY/DeepHypergraph/project_V4` |
| date | 2026-01-16 |
| used class | `models.py::EEG_AdaMSHyper` |
| original task | AVED auditory-attention decoding, 32 ch, T=128 @ 128 Hz, 2-class, LOSO+DANN |
| license | none in the tree (lab internal) |

### Vendored files (`code/models/external/adahgnn/_official/`)

| vendored file | role | sha256 | changed vs lab source? |
|:--|:--|:--|:--|
| `models.py` | `EEG_AdaMSHyper` and submodules | `468468087dead1088d37dc73249feaed44b0f072abe50cd6db4d119919b0b7e1` | **no** — this is the runtime class |
| `config.py` | 32/64-ch scale table | `0bb0decd9b0032e9d054fb82edba7a42d25486ea194af8efc734923772adfa94` | **no** — audit copy; adapter does not import it |

`train.py` / `main.py` / `dataset.py` are **not** vendored: they are the AVED LOSO+DANN
loop (mixup, domain labels, ReduceLROnPlateau on test accuracy). Our cells use
`code/training/e2e_trainer.py` and the shared zs recipe (same rule as TCFormer: we
do not take their optimiser).

The original `models.py` does `import dhg; from dhg.nn import HGNNPConv`. `dhg==0.9.7`
is installed in `mi_torch_cu118` (2026-09-20).

## 2. Weights

None. Every cell trains from scratch. The AVED LOSO checkpoints in
`project_V3/.../results_AdaHGNN_LOSO_DANN/` are a different task and are not loaded.

## 3. What the adapter is allowed to do

`adapter.py` only:

1. Builds a Config-like namespace with the fields `EEG_AdaMSHyper` reads
   (`num_channels`, `time_sample_num`, `patch_size`, `d_model`, `k`, `num_classes`,
   `scale_node_nums`, `hyper_num`) and dummy `num_domain_classes=2` so the DANN head
   is constructed exactly as in the source.
2. Instantiates `EEG_AdaMSHyper` and calls `model(x, alpha=0)`.
3. Maps the first return (task logits) into `{logits, features, confidence}`.
   Domain logits and the NHC constraint tensor are computed inside the official
   `forward` (the file is not patched) and then discarded — they are not added to
   our shared CE loss.

Kernel 63, radix 2, dropout 0.5, `LearnableTemporalShift(sampling_rate=128,
max_shift_sec=0.1)` stay **hardcoded in `_official/models.py`**. They are not
overridden.

| key | value | note |
|:--|:--|:--|
| `d_model` | 128 | source Config |
| `patch_size` | 8 | T=1000 → 125 patches (source T=128 → 16) |
| `k` | 5 | source Config |
| scale 32 ch | `[32, 8, 2]`, hyper `[30, 15, 8]` | source Config, unchanged |
| scale 64 ch | `[64, 16, 4]`, hyper `[40, 20, 10]` | source Config, unused here |
| scale 58 ch | `[58, 29, 1]`, hyper `[40, 20, 10]` | **only extension**: source `Config` raises on anything but 32/64. Pooling is still consecutive-index average (`_generate_mappings`). Hyper-edge counts taken from the 64-ch row (nearest width). |

Shared recipe (not the AVED trainer): Adam 1e-4, batch 128, wd 0, no scheduler,
patience 25 on val macro-F1, seed 0, `[B, C, 1000]` @ 250 Hz, `per_sample_zscore`.

## 4. What this is *not*

- Not motor-imagery code originally (AVED AAD). The classifier head is sized for
  2/3 MI classes; nothing is transferred from AVED checkpoints.
- The class `AnatomicalSpatialPooling` is consecutive-index averaging in the source
  Config, not a 10-20 anatomical atlas. We keep that rule.
- Hypergraph embeddings go through a non-differentiable `adj > 0.5` (no STE) in the
  source, so they have no CE gradient. Same here — the file is unmodified.
- We do **not** run DANN / mixup / AdamW / test-set plateau. That is protocol
  alignment with D-034, not a rewrite of the network.
