# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-NOT gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.cxbase import CXBase


class CnotGate(Gate):
    """controlled-NOT gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CNOT gate."""
        super().__init__("cx", [], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate cx c,t { CX c,t; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("CX", 2, 0, 0)
        rule = [
            CXBase(q[0], q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cx(self.qargs[0], self.qargs[1]))


def cx(self, ctl, tgt):
    """Apply CX from ctl to tgt."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cx((ctl, i), (tgt, i)))
        return instructions

    if isinstance(ctl, QuantumRegister):
        gs = InstructionSet()
        for j in range(ctl.size):
            gs.add(self.cx((ctl, j), tgt))
        return gs

    if isinstance(tgt, QuantumRegister):
        gs = InstructionSet()
        for j in range(tgt.size):
            gs.add(self.cx(ctl, (tgt, j)))
        return gs

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CnotGate(ctl, tgt, self))


QuantumCircuit.cx = cx
