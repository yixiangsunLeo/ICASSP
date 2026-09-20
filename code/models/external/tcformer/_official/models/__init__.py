"""Package marker for the vendored ``models/`` directory (this file is ours, not upstream's).

Upstream ``models/__init__.py`` (sha256 374c71bf...9965, quoted in full in ``../../PROVENANCE.md``)
imports eleven model files (``atcnet``, ``basenet``, ``eegconformer``, ``eegnet``, ``eegtcnet``,
``shallownet``, ``tsseffnet``, ``ctnet``, ``mscformer``, ``eegdeformer`` and ``tcformer``). Only
TCFormer is vendored here, so executing the upstream file would fail on the first missing import;
it is replaced by this empty marker. Import the model as
``from code.models.external.tcformer._official.models.tcformer import TCFormerModule``.
"""
