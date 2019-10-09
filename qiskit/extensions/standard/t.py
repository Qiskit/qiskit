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
T=sqrt(S) phase gate or its inverse.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard.u1 import U1Gate


class TGate(Gate):
    """T Gate: pi/4 rotation around Z axis."""

    def __init__(self, label=None):
        """Create new T gate."""
        super().__init__("t", 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(pi/4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return TdgGate()

    def to_matrix(self):
        """Return a Numpy.array for the S gate."""
        return numpy.array([[1, 0],
                            [0, (1+1j) / numpy.sqrt(2)]], dtype=complex)


class TdgGate(Gate):
    """T Gate: -pi/4 rotation around Z axis."""

    def __init__(self, label=None):
        """Create new Tdg gate."""
        super().__init__("tdg", 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(-pi/4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return TGate()

    def to_matrix(self):
        """Return a Numpy.array for the S gate."""
        return numpy.array([[1, 0],
                            [0, (1-1j) / numpy.sqrt(2)]], dtype=complex)


def t(self, q):
    """Apply T to q."""
    return self.append(TGate(), [q], [])


def tdg(self, q):
    """Apply Tdg to q."""
    return self.append(TdgGate(), [q], [])


QuantumCircuit.t = t
QuantumCircuit.tdg = tdg
