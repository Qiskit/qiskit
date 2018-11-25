# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Pauli Z (phase-flip) gate.
"""
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm import pi
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.u1 import U1Gate


class ZGate(Gate):
    """Pauli Z (phase-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new Z gate."""
        super().__init__("z", [], [qubit], circ)
        self._define_decompositions()

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u1", 1, 0, 1)
        rule = [
            U1Gate(pi, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self._define_decompositions()
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.z(self.qargs[0]))


def z(self, q):
    """Apply Z to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.z((q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(ZGate(q, self))


QuantumCircuit.z = z
