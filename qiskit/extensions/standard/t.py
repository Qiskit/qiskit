# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
T=sqrt(S) phase gate or its inverse.
"""
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class TGate(Gate):
    """T Gate: pi/4 rotation around Z axis."""

    def __init__(self, qubit, circ=None):
        """Create new T gate."""
        super().__init__("t", [], [qubit], circ)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.t(self.arg[0]))

    def inverse(self):
        """Invert this gate."""
        inv = TdgGate(self.arg[0])
        self.circuit.data[self.circuit.data.index(self)] = inv  # replaces the gate with the inverse
        return inv


class TdgGate(Gate):
    """T Gate: -pi/4 rotation around Z axis."""

    def __init__(self, qubit, circ=None):
        """Create new Tdg gate."""
        super().__init__("tdg", [], [qubit], circ)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.tdg(self.arg[0]))

    def inverse(self):
        """Invert this gate."""
        inv = TGate(self.arg[0])
        self.circuit.data[self.circuit.data.index(self)] = inv  # replaces the gate with the inverse
        return inv


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
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.tdg((q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(TdgGate(q, self))


QuantumCircuit.t = t
QuantumCircuit.tdg = tdg
