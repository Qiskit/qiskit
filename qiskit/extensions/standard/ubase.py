# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Element of SU(2).
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class UBase(Gate):
    """Element of SU(2)."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        super().__init__("U", [theta, phi, lam], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        theta = self.param[0]
        phi = self.param[1]
        lamb = self.param[2]
        qubit = self.arg[0]
        return self._qasmif("U(%s,%s,%s) %s[%d];" % (
            theta, phi, lamb, qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate.

        U(theta,phi,lambda)^dagger = U(-theta,-lambda,-phi)
        """
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u_base(self.param[0], self.param[1], self.param[2],
                                    self.arg[0]))


def u_base(self, theta, phi, lam, q):
    """Apply U to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.u_base(theta, phi, lam, (q, j)))
        return gs

    self._check_qubit(q)
    return self._attach(UBase(theta, phi, lam, q, self))


QuantumCircuit.u_base = u_base
CompositeGate.u_base = u_base
