# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Diagonal single qubit gate.
"""
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.ubase import UBase


class U1Gate(Gate):
    """Diagonal single-qubit gate."""

    def __init__(self, theta, qubit, circ=None):
        """Create new diagonal single-qubit gate."""
        super().__init__("u1", [theta], [qubit], circ)
        self._define_decompositions()

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("U", 1, 0, 3)
        rule = [
            UBase(0, 0, self.param[0], q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        self._define_decompositions()
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u1(self.param[0], self.qargs[0]))


def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u1(theta, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U1Gate(theta, q, self))


QuantumCircuit.u1 = u1
