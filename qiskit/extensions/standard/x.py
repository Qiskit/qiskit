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
Pauli X (bit-flip) gate.
"""

import numpy

from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard.u3 import U3Gate


class XGate(Gate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, label=None):
        """Create new X gate."""
        super().__init__("x", 1, [], label=label)

    def _define(self):
        """
        gate x a {
        u3(pi,0,pi) a;
        }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U3Gate(pi, 0, pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return XGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the X gate."""
        return numpy.array([[0, 1],
                            [1, 0]], dtype=complex)


def x(self, q):
    """Apply X to q."""
    return self.append(XGate(), [q], [])


QuantumCircuit.x = x
CompositeGate.x = x
