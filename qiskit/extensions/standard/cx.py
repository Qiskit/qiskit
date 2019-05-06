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

# pylint: disable=invalid-name

"""
controlled-NOT gate.
"""

import numpy

from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.cxbase import CXBase


class CnotGate(Gate):
    """controlled-NOT gate."""

    def __init__(self):
        """Create new CNOT gate."""
        super().__init__("cx", 2, [])

    def _define(self):
        """
        gate cx c,t { CX c,t; }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (CXBase(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CnotGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Cx gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]], dtype=complex)


def cx(self, ctl, tgt):
    """Apply CX from ctl to tgt."""
    return self.append(CnotGate(), [ctl, tgt], [])


QuantumCircuit.cx = cx
CompositeGate.cx = cx
