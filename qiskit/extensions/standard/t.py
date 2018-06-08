# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
T=sqrt(S) phase gate or its inverse.
"""
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class TGate(CompositeGate):
    """T=sqrt(S) Clifford phase gate or its inverse."""

    def __init__(self, qubit, circ=None):
        """Create new T gate."""
        super().__init__("t", [], [qubit], circ)
        self.u1(pi / 4, qubit)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.t(self.arg[0]))

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.data[0].arg[0]
        phi = self.data[0].param[0]
        if phi > 0:
            return self.data[0]._qasmif("t %s[%d];" % (qubit[0].name, qubit[1]))

        return self.data[0]._qasmif("tdg %s[%d];" % (qubit[0].name, qubit[1]))


def t(self, q):
    """Apply T to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.t((q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(TGate(q, self))


def tdg(self, q):
    """Apply Tdg to q."""
    return self.t(q).inverse()


QuantumCircuit.t = t
QuantumCircuit.tdg = tdg
CompositeGate.t = t
CompositeGate.tdg = tdg
