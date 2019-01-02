# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-Phase gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate


class CzGate(Gate):
    """controlled-Z gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CZ gate."""
        super().__init__("cz", [], [ctl, tgt], circ)

    def _define_decompositions(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("h", 1, 0, 0)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            HGate(q[1]),
            CnotGate(q[0], q[1]),
            HGate(q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cz(self.qargs[0], self.qargs[1]))


def cz(self, ctl, tgt):
    """Apply CZ to circuit."""
    if isinstance(ctl, QuantumRegister):
        ctl = [(ctl, i) for i in range(len(ctl))]
    if isinstance(tgt, QuantumRegister):
        tgt = [(tgt, i) for i in range(len(tgt))]

    if ctl and tgt and isinstance(ctl, list) and \
       isinstance(tgt, list) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for ictl, itgt in zip(ctl, tgt):
            instructions.add(self.cz(ictl, itgt))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CzGate(ctl, tgt, self))


QuantumCircuit.cz = cz
