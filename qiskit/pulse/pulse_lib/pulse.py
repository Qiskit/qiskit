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
import warnings
from typing import Callable, Optional
from abc import ABC, abstractmethod

import numpy as np

from ..channels import PulseChannel
from ..exceptions import PulseError
from ..instructions.play import Play


class Pulse(ABC):
    """The abstract superclass for pulses."""

    @abstractmethod
    def __init__(self, duration: int, name: Optional[str] = None):
        if not isinstance(duration, (int, np.integer)):
            raise PulseError('Pulse duration should be integer.')
        self.duration = int(duration)
        self.name = (name if name is not None
                     else '{}{}'.format(str(self.__class__.__name__).lower(),
                                        self.__hash__()))

    def __call__(self, channel: PulseChannel) -> Play:
        """Return new ``Play`` instruction that is fully instantiated with both ``pulse`` and a
        ``channel``.

        Args:
            channel: The channel that will have the pulse.

        Return:
            Complete and ready to schedule ``Play``.
        """
        warnings.warn("Calling a ``Pulse`` with a channel is deprecated. Instantiate ``Play`` "
                      "directly with a pulse and a channel.", DeprecationWarning)
        return Play(self, channel)

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
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: 'Pulse') -> bool:
        return isinstance(other, type(self))

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError
