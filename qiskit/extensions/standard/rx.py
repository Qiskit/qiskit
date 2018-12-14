# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Rotation around the x-axis.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.u3 import U3Gate


class RXGate(Gate):
    """rotation around the x-axis."""

    def __init__(self, theta, qubit, circ=None):
        """Create new rx single qubit gate."""
        super().__init__("rx", [theta], [qubit], circ)

    def _define_decompositions(self):
        """
        gate rx(theta) a {u3(theta, -pi/2, pi/2) a;}
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u3", 1, 0, 3)
        rule = [
            U3Gate(self.param[0], -pi/2, pi/2, q[0])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate.

        rx(theta)^dagger = rx(-theta)
        """
        self.param[0] = -self.param[0]
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rx(self.param[0], self.qargs[0]))


def rx(self, theta, q):
    """Apply Rx to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.rx(theta, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(RXGate(theta, q, self))


QuantumCircuit.rx = rx
