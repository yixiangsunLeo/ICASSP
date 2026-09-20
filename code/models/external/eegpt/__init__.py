"""EEGPT (Wang et al., NeurIPS 2024) — official large encoder behind this project's contract.

Imported lazily by ``code.models.external.build_external("eegpt", ...)``; see ``adapter.py``
for the wrapper and ``PROVENANCE.md`` for what was vendored, from where, and what differs.
"""

from .adapter import EEGPTClassifier, EEGPTConfig, build

__all__ = ["EEGPTClassifier", "EEGPTConfig", "build"]
