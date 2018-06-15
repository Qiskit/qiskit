# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Pauli X (bit-flip) gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.cx import CnotGate


class XGate(Gate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new X gate."""
        super().__init__("x", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("x %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.x(self.arg[0]))

    def q_if(self, *qregs):
        """Add controls to this gate."""
        if not qregs:
            return self
        return CnotGate(qregs[0], self.arg[0], self._get_circuit()).q_if(*qregs[1:])


def x(self, q):
    """Apply X to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.x((q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(XGate(q, self))


QuantumCircuit.x = x
CompositeGate.x = x
