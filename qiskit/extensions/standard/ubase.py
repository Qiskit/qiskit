# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Element of SU(2).
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class UBase(Gate):  # pylint: disable=abstract-method
    """Element of SU(2)."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        super().__init__("U", [theta, phi, lam], [qubit], circ)

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
                                    self.qargs[0]))


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
