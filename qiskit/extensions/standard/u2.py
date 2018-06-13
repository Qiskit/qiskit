# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
One-pulse single-qubit gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class U2Gate(Gate):
    """One-pulse single-qubit gate."""

    def __init__(self, phi, lam, qubit, circ=None):
        """Create new one-pulse single-qubit gate."""
        super().__init__("u2", [phi, lam], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        phi = self.param[0]
        lam = self.param[1]
        return self._qasmif("u2(%s,%s) %s[%d];" % (phi, lam,
                                                   qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        u2(phi,lamb)^dagger = u2(-lamb-pi,-phi+pi)
        """
        phi = self.param[0]
        self.param[0] = -self.param[1] - pi
        self.param[1] = -phi + pi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u2(self.param[0], self.param[1], self.arg[0]))


def u2(self, phi, lam, q):
    """Apply u2 to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u2(phi, lam, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U2Gate(phi, lam, q, self))


QuantumCircuit.u2 = u2
CompositeGate.u2 = u2
