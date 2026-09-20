"""CBraMod (Wang et al., ICLR 2025) — official code vendored under ``_official/``, wrapped by ``adapter.py``.

See ``PROVENANCE.md`` for upstream commit, file hashes, the one import-line edit, the pretrained
checkpoint and the upstream-default-vs-used hyperparameter table.
"""

from .adapter import CBraModClassifier, CBraModConfig, build

__all__ = ["CBraModClassifier", "CBraModConfig", "build"]
