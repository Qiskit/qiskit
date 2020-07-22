# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A pulse that is described by complex-valued sample points."""
import warnings
from typing import Union, List, Optional

import numpy as np

from .waveform import Waveform


class SamplePulse(Waveform):
    """Deprecated.
    A pulse specified completely by complex-valued samples; each sample is played for the
    duration of the backend cycle-time, dt.
    """
    def __init__(self, samples: Union[np.ndarray, List[complex]],
                 name: Optional[str] = None,
                 epsilon: float = 1e-7):
        """Create new sample pulse envelope.

        Args:
            samples: Complex array of the samples in the pulse envelope.
            name: Unique name to identify the pulse.
            epsilon: Pulse sample norm tolerance for clipping.
                If any sample's norm exceeds unity by less than or equal to epsilon
                it will be clipped to unit norm. If the sample
                norm is greater than 1+epsilon an error will be raised.
        """
        warnings.warn("SamplePulse has been renamed to Waveform and is deprecated. " +
                      "Please replace SamplePulse(samples, channel) with "
                      "Waveform(samples, channel).", DeprecationWarning)
        super().__init__(samples, name=name, epsilon=epsilon)
