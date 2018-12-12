# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
One-pulse single-qubit gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm import pi
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.ubase import UBase


class U2Gate(Gate):
    """One-pulse single-qubit gate."""

    def __init__(self, phi, lam, qubit, circ=None):
        """Create new one-pulse single-qubit gate."""
        super().__init__("u2", [phi, lam], [qubit], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("U", 1, 0, 3)
        rule = [
            UBase(pi/2, self.param[0], self.param[1], q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        u2(phi,lamb)^dagger = u2(-lamb-pi,-phi+pi)
        """
        phi = self.param[0]
        self.param[0] = -self.param[1] - pi
        self.param[1] = -phi + pi
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u2(self.param[0], self.param[1], self.qargs[0]))


def u2(self, phi, lam, q):
    """Apply u2 to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u2(phi, lam, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U2Gate(phi, lam, q, self))


QuantumCircuit.u2 = u2
