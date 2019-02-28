# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Rotation around the x-axis.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.qasm import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u3 import U3Gate


class RXGate(Gate):
    """rotation around the x-axis."""

    def __init__(self, theta, circ=None):
        """Create new rx single qubit gate."""
        super().__init__("rx", 1, [theta], circ)

    def _define_decompositions(self):
        """
        gate rx(theta) a {u3(theta, -pi/2, pi/2) a;}
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U3Gate(self.params[0], -pi/2, pi/2), [q[0]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        rx(theta)^dagger = rx(-theta)
        """
        self.params[0] = -self.params[0]
        self._decompositions = None
        return self


@_op_expand(1)
def rx(self, theta, q):
    """Apply Rx to q."""
    return self._attach(RXGate(theta, self), [q], [])


QuantumCircuit.rx = rx
CompositeGate.rx = rx
