"""Verbatim copies of the official EEGPT code (https://github.com/BINE022/EEGPT, Apache-2.0).

Directory layout mirrors the upstream repository so ``diff -r`` against a checkout of commit
``a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705`` is trivial:

  * ``downstream/Modules/models/EEGPT_mcae.py`` — ``EEGTransformer`` (the pretrained encoder),
    ``CHANNEL_DICT`` (62-electrode vocabulary), predictor / reconstructor (unused here)
  * ``downstream/Modules/Network/utils.py``     — ``Conv1dWithConstraint`` (the 1x1 channel mixer
    the official downstream scripts put in front of the encoder)
  * ``LICENSE``                                 — upstream Apache License 2.0

Nothing in these files is modified (sha256 of each copy in ``../PROVENANCE.md``). The
``__init__.py`` files are ours and empty; they only make the tree importable.
"""
