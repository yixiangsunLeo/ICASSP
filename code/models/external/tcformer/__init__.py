"""TCFormer (Altaheri et al., Sci Rep 2025) — official code vendored under ``_official/``.

``build(...)`` is what ``code.models.external.build_external`` calls; see ``adapter.py`` for the
contract and ``PROVENANCE.md`` for upstream commit, hashes and the edited import lines.
"""

from .adapter import TCFormerClassifier, TCFormerConfig, build

__all__ = ["TCFormerClassifier", "TCFormerConfig", "build"]
