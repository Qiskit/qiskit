# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-Y gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.s import SGate
from qiskit.extensions.standard.s import SdgGate
from qiskit.extensions.standard.cx import CnotGate


class CyGate(Gate):
    """controlled-Y gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CY gate."""
        super().__init__("cy", [], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate cy a,b { sdg b; cx a,b; s b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("s", 1, 0, 0)
        decomposition.add_basis_element("sdg", 1, 0, 0)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            SdgGate(q[1]),
            CnotGate(q[0], q[1]),
            SGate(q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cy(self.qargs[0], self.qargs[1]))


def cy(self, ctl, tgt):
    """Apply CY to circuit."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cy((ctl, i), (tgt, i)))
        return instructions

    if isinstance(ctl, QuantumRegister):
        gs = InstructionSet()
        for j in range(ctl.size):
            gs.add(self.cy((ctl, j), tgt))
        return gs

    if isinstance(tgt, QuantumRegister):
        gs = InstructionSet()
        for j in range(tgt.size):
            gs.add(self.cy(ctl, (tgt, j)))
        return gs

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CyGate(ctl, tgt, self))


QuantumCircuit.cy = cy
