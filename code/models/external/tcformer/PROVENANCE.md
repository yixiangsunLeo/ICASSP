# PROVENANCE — TCFormer (`tcformer`)

Paper: H. Altaheri, F. Karray, A.-H. Karimi, *Temporal convolutional transformer for
EEG based motor imagery decoding*, Scientific Reports 15:32959, 2025
(doi:10.1038/s41598-025-16219-7).

## 1. Upstream code

| item | value |
|:--|:--|
| repository | <https://github.com/altaheri/TCFormer> |
| commit | `701a361640708ef1cc07cf22660d47edc6d9bf3c` (2026-08-15) |
| license | MIT, Copyright (c) 2025 Hamdi Altaheri — copied to `_official/LICENSE` |
| used class | `models/tcformer.py::TCFormerModule` (pure `nn.Module`). The Lightning wrapper `TCFormer(ClassificationModule)` is **not** instantiated |

### Vendored files (`code/models/external/tcformer/_official/`)

| vendored file | role | sha256 of vendored copy | changed? |
|:--|:--|:--|:--|
| `models/tcformer.py` | `TCFormerModule` | `2c98b35856b7bd8b9b4306f55844b9818d4bf8cae3396823a4f2ceb55e8e219a` | **import paths only** (below) |
| `models/modules.py` | `CausalConv1d`, `Conv1dWithConstraint` | `cb24a800c47cf3e417864b9928ba8da61017b1e928ae53706879c6af8a910c14` | no (relative imports already local) |
| `models/channel_group_attention.py` | group attention | `5623b68c9e1524faad00a8450a8f2fb909d3f39c7d4d20bef7ae769b8de8aabb` | no |
| `models/attention_blocks.py` | unused by `TCFormerModule`; kept for the record | `5c0b2a31a7e2ecf0a2b25f4c782ab267e495277fd73dcee5001112d9ea9da468` | no |
| `models/classification_module.py` | Lightning training loop; **not imported** | `bd0b2646b89fa0499c59b114a03e048bf02d9e19407c7ed71ae83137a05ce52d` | no |
| `utils/weight_initialization.py` | `glorot_weight_zero_bias` | `e92148c28b1ae39dc7c7a0d52704594545fb09cd9eaa6b1b819c4a3836ad01e3` | no |
| `utils/latency.py` | unused at run time | `3910278e8470fd123aacaf364a4ee1378ced3dc741749981535b1031670c8690` | no |
| `configs/tcformer.yaml` | paper `model_kwargs` | `114588b9e6e2c1e14f99e2f7510ffb365dae704c1c0d07e06d26c866a4df4fac` | no |
| `LICENSE` | MIT | `daea0e9f8596568cf5aae39b69f45ffc9f1b00d600d5315c09e90668cd99ee97` | no |
| `README.md` | upstream | `de83665f0eeea98d0656e7bd3d324448ed90206001701ab6d99dd63d6b6edb33` | no |

**Ours, not upstream:** `models/__init__.py` (upstream's file imports eleven other models we did not vendor); `../_compat.py` (Lightning-free `ClassificationModule` stub so `tcformer.py` can import).

Import-path edits in `models/tcformer.py` (the only edits to upstream Python):

```diff
-from models.classification_module import ClassificationModule
-from models.modules import CausalConv1d, Conv1dWithConstraint
-from models.channel_group_attention import ChannelGroupAttention
-from utils.weight_initialization import glorot_weight_zero_bias
-from utils.latency  import measure_latency
+from ..._compat import ClassificationModule
+from .modules import CausalConv1d, Conv1dWithConstraint
+from .channel_group_attention import ChannelGroupAttention
+from ..utils.weight_initialization import glorot_weight_zero_bias
+from ..utils.latency  import measure_latency
```

No architecture, initialisation or forward maths is touched. `pytorch_lightning` / `torchmetrics` are not imported.

## 2. Weights

None. Every cell trains from scratch. Upstream releases no pretrained checkpoint.

## 3. Hyperparameters

Defaults = upstream `configs/tcformer.yaml::model_kwargs` (F1=32, kernels [20,32,64], d_group=16, D=2, dropout_conv=0.4, q_heads=4, kv_heads=2, trans_depth=5, drop_path_max=0.25 class default). **Not** the Python class defaults of `TCFormerModule` (F1=16, kernels (16,32,64), q_heads=8, kv_heads=4).

The README's 77.8 k figure is `trans_depth=2` at BCI IV-2a (22 ch / 4 cls) = **77,820** (`tcformer@n2`). The yaml ships depth 5 (= 127,212 at 22 ch / 4 cls). Comparison uses the yaml (depth 5).

Upstream optimiser (Adam lr 9e-4, β1=0.5, wd 1e-3, warm-up+cosine, inter-trial aug) is **not** used — shared recipe (Adam 1e-4, batch 128, patience 25, no scheduler).

Input: raw `[B, C, 1000] @ 250 Hz`. No resampling. T=1000 → 17 tokens after pool 8 then 7.

## 4. Parameter counts (this tree, paper yaml)

| configuration | n_params |
|:--|--:|
| 58 ch / 2 classes (2C) | 133,988 |
| 58 ch / 3 classes (3C) | 134,056 |
| 32 ch / 2 classes (SHU) | 128,996 |
| 22 ch / 4 cls, depth 5, T=1125 (yaml) | 127,212 |
| 22 ch / 4 cls, depth 2, T=1125 (`@n2`, README) | 77,820 |
