# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Rotation around the z-axis.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class RZGate(Gate):
    """rotation around the z-axis."""

    def __init__(self, phi, qubit, circ=None):
        """Create new rz single qubit gate."""
        super().__init__("rz", [phi], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        phi = self.param[0]
        return self._qasmif("rz(%s) %s[%d];" % (phi, qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate.

        rz(phi)^dagger = rz(-phi)
        """
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rz(self.param[0], self.arg[0]))


def rz(self, phi, q):
    """Apply Rz to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.rz(phi, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(RZGate(phi, q, self))


QuantumCircuit.rz = rz
CompositeGate.rz = rz
