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
Hadamard gate.
"""
import numpy

from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard.u2 import U2Gate


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, label=None):
        """Create new Hadamard gate."""
        super().__init__("h", 1, [], label=label)

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U2Gate(0, pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return HGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the H gate."""
        return numpy.array([[1, 1],
                            [1, -1]], dtype=complex) / numpy.sqrt(2)


def h(self, q):  # pylint: disable=invalid-name
    """Apply Hadamard (H) gate to a specified qubit (q).
    An H gate implements a rotation of pi about the axis (x + z)/sqrt(2) on the Bloch sphere.
    This gate is canonically used to rotate the qubit state from |0⟩ to |+⟩ or |1⟩ to |-⟩.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.h(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.h import HGate
            HGate().to_matrix()
    """
    return self.append(HGate(), [q], [])


QuantumCircuit.h = h
