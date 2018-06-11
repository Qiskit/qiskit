# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Two-pulse single-qubit gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class U3Gate(Gate):
    """Two-pulse single-qubit gate."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        """Create new two-pulse single qubit gate."""
        super().__init__("u3", [theta, phi, lam], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        phi = self.param[1]
        lam = self.param[2]
        return self._qasmif("u3(%s,%s,%s) %s[%d];" % (theta, phi, lam,
                                                      qubit[0].name,
                                                      qubit[1]))

    def inverse(self):
        """Invert this gate.

        u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
        """
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u3(self.param[0], self.param[1], self.param[2],
                                self.arg[0]))


def u3(self, theta, phi, lam, q):
    """Apply u3 to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u3(theta, phi, lam, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U3Gate(theta, phi, lam, q, self))


QuantumCircuit.u3 = u3
CompositeGate.u3 = u3
