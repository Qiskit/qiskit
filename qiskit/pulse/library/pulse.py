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

"""Pulses are descriptions of waveform envelopes. They can be transmitted by control electronics
to the device.
"""
from typing import Callable, Optional
from abc import ABC, abstractmethod

import numpy as np

from ..exceptions import PulseError


class Pulse(ABC):
    """The abstract superclass for pulses. Pulses are complex-valued waveform envelopes. The
    modulation phase and frequency are specified separately from ``Pulse``s.
    """

    @abstractmethod
    def __init__(self, duration: int, name: Optional[str] = None):
        if not isinstance(duration, (int, np.integer)):
            raise PulseError('Pulse duration should be integer.')
        self.duration = int(duration)
        self.name = name

    @property
    def id(self) -> int:  # pylint: disable=invalid-name
        """Unique identifier for this pulse."""
        return id(self)

    @abstractmethod
    def draw(self, dt: float = 1,
             style=None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scale: float = 1, interactive: bool = False):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style (Optional[PulseStyle]): A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: 'Pulse') -> bool:
        return isinstance(other, type(self))

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
