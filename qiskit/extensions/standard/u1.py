# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Diagonal single qubit gate.
"""
import numpy
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand, _to_bits
from qiskit.extensions.standard.u3 import U3Gate


class U1Gate(Gate):
    """Diagonal single-qubit gate."""

    def __init__(self, theta, label=None):
        """Create new diagonal single-qubit gate."""
        super().__init__("u1", 1, [theta], label=label)

    def _define(self):
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U3Gate(0, 0, self.params[0]), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return U1Gate(-self.params[0])

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
        lam = self.params[0]
        lam = float(lam)
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=complex)


@_to_bits(1)
@_op_expand(1)
def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    return self.append(U1Gate(theta), [q], [])


QuantumCircuit.u1 = u1
CompositeGate.u1 = u1
