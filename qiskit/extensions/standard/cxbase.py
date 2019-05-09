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
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit


class CXBase(Gate):  # pylint: disable=abstract-method
    """Fundamental controlled-NOT gate."""

    def __init__(self):
        """Create new CX instruction."""
        super().__init__("CX", 2, [])

    def inverse(self):
        """Invert this gate."""
        return CXBase()  # self-inverse


def cx_base(self, ctl, tgt):
    """Apply CX ctl, tgt."""
    return self.append(CXBase(), [ctl, tgt], [])


QuantumCircuit.cx_base = cx_base
CompositeGate.cx_base = cx_base
