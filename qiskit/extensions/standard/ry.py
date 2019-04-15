# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Rotation around the y-axis.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.u3 import U3Gate


class RYGate(Gate):
    """rotation around the y-axis."""

    def __init__(self, theta):
        """Create new ry single qubit gate."""
        super().__init__("ry", 1, [theta])

    def _define(self):
        """
        gate ry(theta) a { u3(theta, 0, 0) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U3Gate(self.params[0], 0, 0), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        return RYGate(-self.params[0])


@_op_expand(1)
def ry(self, theta, q):
    """Apply Ry to q."""
    return self.append(RYGate(theta), [q], [])


QuantumCircuit.ry = ry
CompositeGate.ry = ry
