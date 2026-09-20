"""LaBraM-base (Jiang et al., ICLR 2024) — official ``modeling_finetune.py`` vendored in
``_official/``, wrapped by ``adapter.py``. See ``PROVENANCE.md``."""

from .adapter import LaBraMClassifier, LaBraMConfig, build

__all__ = ["LaBraMClassifier", "LaBraMConfig", "build"]
