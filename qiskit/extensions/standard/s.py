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
S=diag(1,i) Clifford phase gate or its inverse.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard.u1 import U1Gate


class SGate(Gate):
    """S=diag(1,i) Clifford phase gate."""

    def __init__(self, label=None):
        """Create new S gate."""
        super().__init__("s", 1, [], label=label)

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(pi/2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SdgGate()

    def to_matrix(self):
        """Return a Numpy.array for the S gate."""
        return numpy.array([[1, 0],
                            [0, 1j]], dtype=complex)


class SdgGate(Gate):
    """Sdg=diag(1,-i) Clifford adjoint phase gate."""

    def __init__(self, label=None):
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [], label=label)

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(-pi/2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SGate()

    def to_matrix(self):
        """Return a Numpy.array for the Sdg gate."""
        return numpy.array([[1, 0],
                            [0, -1j]], dtype=complex)


def s(self, q):
    """Apply S to q."""
    return self.append(SGate(), [q], [])


def sdg(self, q):
    """Apply Sdg to q."""
    return self.append(SdgGate(), [q], [])


QuantumCircuit.s = s
QuantumCircuit.sdg = sdg
