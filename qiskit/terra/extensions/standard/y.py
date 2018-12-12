# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Pauli Y (bit-phase-flip) gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm import pi
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.u3 import U3Gate


class YGate(Gate):
    """Pauli Y (bit-phase-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new Y gate."""
        super().__init__("y", [], [qubit], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u3", 1, 0, 3)
        rule = [
            U3Gate(pi, pi/2, pi/2, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circuit):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circuit.y(self.qargs[0]))


def y(self, q):
    """Apply Y to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.y((q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(YGate(q, self))


QuantumCircuit.y = y
