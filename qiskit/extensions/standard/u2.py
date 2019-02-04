# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
One-pulse single-qubit gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm import pi
from qiskit.extensions.standard.u3 import U3Gate


class U2Gate(Gate):
    """One-pulse single-qubit gate."""

    def __init__(self, phi, lam, circ=None):
        """Create new one-pulse single-qubit gate."""
        super().__init__("u2", [phi, lam], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            U3Gate(pi/2, self.params[0], self.params[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst, [q[0]], [])
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        u2(phi,lamb)^dagger = u2(-lamb-pi,-phi+pi)
        """
        phi = self.params[0]
        self.params[0] = -self.params[1] - pi
        self.params[1] = -phi + pi
        self._decompositions = None
        return self


@_op_expand(1)
def u2(self, phi, lam, q):
    """Apply u2 to q."""
    return self._attach(U2Gate(phi, lam, self), [q], [])


QuantumCircuit.u2 = u2
CompositeGate.u2 = u2
