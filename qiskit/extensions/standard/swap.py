# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
SWAP gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.cx import CnotGate


class SwapGate(Gate):
    """SWAP gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new SWAP gate."""
        super().__init__("swap", [], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            CnotGate(q[0], q[1]),
            CnotGate(q[1], q[0]),
            CnotGate(q[0], q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.swap(self.qargs[0], self.qargs[1]))


def swap(self, ctl, tgt):
    """Apply SWAP from ctl to tgt."""
    if isinstance(ctl, QuantumRegister):
        ctl = [(ctl, i) for i in range(len(ctl))]
    if isinstance(tgt, QuantumRegister):
        tgt = [(tgt, i) for i in range(len(tgt))]

    if ctl and tgt and isinstance(ctl, list) and \
       isinstance(tgt, list) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for ictl, itgt in zip(ctl, tgt):
            instructions.add(self.swap(ictl, itgt))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(SwapGate(ctl, tgt, self))


QuantumCircuit.swap = swap
