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

"""
Delay instruction.
"""
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction


class Delay(Instruction):
    """Do nothing and just delay/wait/idle for a specified duration."""

    def __init__(self, duration, unit='dt'):
        """Create new delay instruction."""
        super().__init__("delay", 1, 0, params=[duration], duration=duration)
        self.unit = unit

    def inverse(self):
        """Special case. Return self."""
        return self

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg for sublist in qargs for qarg in sublist], []

    def c_if(self, classical, val):
        raise CircuitError('Conditional Delay is not yet implemented.')

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self.params = [duration]
        self._duration = duration

    def to_matrix(self) -> np.ndarray:
        """Return the identity matrix."""
        return np.array([[1, 0],
                         [0, 1]], dtype=complex)

    def __repr__(self):
        """Return the official string representing the delay."""
        return "%s(duration=%s[unit=%s])" % \
               (self.__class__.__name__, self.params[0], self.unit)
