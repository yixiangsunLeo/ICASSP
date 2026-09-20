"""Lightning-free stand-in for upstream ``models/classification_module.py::ClassificationModule``.

The vendored ``_official/models/tcformer.py`` defines ``class TCFormer(ClassificationModule)`` at
module level, so *some* class of that name must exist for the file to import. Upstream's class is a
``pytorch_lightning.LightningModule`` that carries the authors' training loop (Adam with
``beta_1=0.5``, ``lr=9e-4``, ``weight_decay=1e-3``, linear warm-up + cosine decay, torchmetrics
accuracy/kappa, optional channel masking augmentation). None of that is used in this project:
every model is trained by the one shared recipe in ``code/training/e2e_trainer.py``, and
pytorch_lightning / torchmetrics are not installed in ``mi_torch_cu118``.

This stub therefore only mirrors the constructor signature and the forward pass. The adapter never
instantiates ``TCFormer``; it uses the pure ``nn.Module`` ``TCFormerModule``. The original file is
kept verbatim next to the model for the record (``_official/models/classification_module.py``).
"""

from __future__ import annotations

import torch.nn as nn


class ClassificationModule(nn.Module):
    """Forward-only replacement: ``ClassificationModule(model, n_classes, **training_kwargs)``."""

    def __init__(self, model: nn.Module, n_classes: int, **kwargs):
        super().__init__()
        self.model = model
        self.n_classes = int(n_classes)
        # Upstream stores these through ``save_hyperparameters``; keep them inspectable only.
        self.hparams = dict(kwargs)

    def forward(self, x):
        return self.model(x)


__all__ = ["ClassificationModule"]
