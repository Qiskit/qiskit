# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Hadamard gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.u2 import U2Gate


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, qubit, circ=None):
        """Create new Hadamard gate."""
        super().__init__("h", [], [qubit], circ)

    def _define_decompositions(self):
        """
        gate h a { u2(0,pi) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u2", 1, 0, 2)
        rule = [
            U2Gate(0, pi, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.h(self.qargs[0]))


def h(self, q):
    """Apply H to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.h((q, j)))
        return instructions

    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.h(q))
        return instructions

    self._check_qubit(q)
    return self._attach(HGate(q, self))


QuantumCircuit.h = h
