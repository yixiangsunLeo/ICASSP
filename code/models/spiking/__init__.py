"""Spiking neural network ports of the benchmark's strongest specialist models.

* ``seegnex`` — spiking EEGNeX (Chen et al. 2024), the top-ranked ANN of the cross-subject
  benchmark on SHU 2022 / WBCIC-SHU 2C (paired against 17 other models).
* ``seegnet``  — spiking EEGNet-8,2 (Lawhern et al. 2018), the canonical compact baseline.

Direct training with surrogate gradients (ATan), LIF / PLIF / heterogeneous-PLIF neurons,
``repeat`` (direct) or ``chunk`` (streaming) input encoding, energy accounting from observed
firing rates. See ``models.py`` for the layer-by-layer mapping and ``adapter.py`` for the
project contract.
"""

from .adapter import SPIKING_MODEL_NAMES, SpikingClassifier, SpikingConfig, build_spiking
from .models import SpikingEEGNeX, SpikingEEGNet
from .neurons import MultiStepLIF

__all__ = ["SPIKING_MODEL_NAMES", "SpikingClassifier", "SpikingConfig", "build_spiking",
           "SpikingEEGNeX", "SpikingEEGNet", "MultiStepLIF"]
