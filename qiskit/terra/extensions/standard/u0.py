# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Single qubit gate cycle idle.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.ubase import UBase


class U0Gate(Gate):
    """Wait gate."""

    def __init__(self, m, qubit, circ=None):
        """Create new u0 gate."""
        super().__init__("u0", [m], [qubit], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("U", 1, 0, 3)
        rule = [
            UBase(0, 0, 0, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u0(self.param[0], self.qargs[0]))


def u0(self, m, q):
    """Apply u0 with length m to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u0(m, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U0Gate(m, q, self))


QuantumCircuit.u0 = u0
