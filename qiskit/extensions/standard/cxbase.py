# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fundamental controlled-NOT gate.
"""
import warnings
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit


class CXBase(Gate):
    """Fundamental controlled-NOT gate."""

    def __init__(self):
        """Create new CX instruction."""
        warnings.warn('CXBase is deprecated and it will be removed after 0.9. '
                      'Use CnotGate instead.', DeprecationWarning, 2)
        super().__init__("CX", 2, [])

    def inverse(self):
        """Invert this gate."""
        warnings.warn('CXBase.inverse is deprecated and it will be removed after 0.9. '
                      'Use CnotGate.inverse instead.', DeprecationWarning, 2)
        return CXBase()  # self-inverse


def cx_base(self, ctl, tgt):
    """Apply CX ctl, tgt."""
    return self.append(CXBase(), [ctl, tgt], [])


QuantumCircuit.cx_base = cx_base
