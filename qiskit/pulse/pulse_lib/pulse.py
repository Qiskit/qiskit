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

"""Any command which implements a transmit signal on a channel."""
from typing import Callable, List, Optional

from abc import ABC, abstractmethod

from ..channels import PulseChannel


class Pulse(ABC):
    """The abstract superclass for pulses."""

    @abstractmethod  #???
    def __init__(self, duration: int = None):
        if isinstance(duration, (int, np.integer)):
            self._duration = int(duration)
        else:
            raise PulseError('Pulse duration should be integer.')

    def __call__(self):
        """TODO, deprecated"""
        pass

    @abstractmethod
    def draw(self, dt: float = 1,
             style: Optional['PulseStyle'] = None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scale: float = 1, interactive: bool = False,
             scaling: float = None):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style: A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            scaling: Deprecated, see `scale`

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        pass
