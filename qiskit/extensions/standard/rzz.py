# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
two-qubit ZZ-rotation gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.cx import CnotGate


class RZZGate(Gate):
    """Two-qubit ZZ-rotation gate."""

    def __init__(self, theta, ctl, tgt, circ=None):
        """Create new rzz gate."""
        super().__init__("rzz", [theta], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate rzz(theta) a, b { cx a, b; u1(theta) b; cx a, b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("u1", 1, 0, 1)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            CnotGate(q[0], q[1]),
            U1Gate(self.param[0], q[0]),
            CnotGate(q[0], q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rzz(self.param[0], self.qargs[0], self.qargs[1]))


def rzz(self, theta, ctl, tgt):
    """Apply RZZ to circuit."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.rzz(theta, (ctl, i), (tgt, i)))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(RZZGate(theta, ctl, tgt, self))


# Add to QuantumCircuit class
QuantumCircuit.rzz = rzz
