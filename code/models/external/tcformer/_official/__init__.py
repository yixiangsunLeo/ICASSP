"""Verbatim copy of the official TCFormer release (https://github.com/altaheri/TCFormer, MIT).

Upstream commit ``701a361640708ef1cc07cf22660d47edc6d9bf3c`` (main, 2026-08-15). The upstream
repository has no package ``__init__.py`` at its root; this file is ours and only makes the
directory importable as ``code.models.external.tcformer._official``.

Everything the adapter executes is ``models/tcformer.py`` (``TCFormerModule``) plus its
dependencies ``models/modules.py``, ``models/channel_group_attention.py`` and
``utils/weight_initialization.py`` / ``utils/latency.py``. The only lines that differ from
upstream are three import paths in ``models/tcformer.py`` — listed with before/after in
``../PROVENANCE.md``. ``models/classification_module.py`` (pytorch_lightning training wrapper)
and ``models/attention_blocks.py`` (needs torchvision, unused by TCFormer) are kept for the
record and are never imported.
"""
