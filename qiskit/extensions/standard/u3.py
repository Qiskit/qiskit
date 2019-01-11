# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Two-pulse single-qubit gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _1q_gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.ubase import UBase


class U3Gate(Gate):
    """Two-pulse single-qubit gate."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        """Create new two-pulse single qubit gate."""
        super().__init__("u3", [theta, phi, lam], [qubit], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("U", 1, 0, 3)
        rule = [
            UBase(self.params[0], self.params[1], self.params[2], q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
        """
        self.params[0] = -self.params[0]
        phi = self.params[1]
        self.params[1] = -self.params[2]
        self.params[2] = -phi
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u3(self.params[0], self.params[1], self.params[2],
                                self.qargs[0]))


@_1q_gate
def u3(self, theta, phi, lam, q):
    """Apply u3 to q."""
    self._check_qubit(q)
    return self._attach(U3Gate(theta, phi, lam, q, self))


QuantumCircuit.u3 = u3
