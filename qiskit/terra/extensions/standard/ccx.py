# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Toffoli gate. Controlled-Controlled-X.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.t import TGate
from qiskit.extensions.standard.t import TdgGate


class ToffoliGate(Gate):
    """Toffoli gate."""

    def __init__(self, ctl1, ctl2, tgt, circ=None):
        """Create new Toffoli gate."""
        super().__init__("ccx", [], [ctl1, ctl2, tgt], circ)

    def _define_decompositions(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(3, "q")
        decomposition.add_qreg(q)
        decomposition.add_basis_element("h", 1, 0, 0)
        decomposition.add_basis_element("cx", 2, 0, 0)
        decomposition.add_basis_element("t", 1, 0, 0)
        decomposition.add_basis_element("tdg", 1, 0, 0)
        decomposition.add_basis_element("s", 1, 0, 0)
        decomposition.add_basis_element("sdg", 1, 0, 0)
        rule = [
            HGate(q[2]),
            CnotGate(q[1], q[2]),
            TdgGate(q[2]),
            CnotGate(q[0], q[2]),
            TGate(q[2]),
            CnotGate(q[1], q[2]),
            TdgGate(q[2]),
            CnotGate(q[0], q[2]),
            TGate(q[1]),
            TGate(q[2]),
            HGate(q[2]),
            CnotGate(q[0], q[1]),
            TGate(q[0]),
            TdgGate(q[1]),
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
        self._modifiers(circ.ccx(self.qargs[0], self.qargs[1], self.qargs[2]))


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to from ctl1 and ctl2 to tgt."""
    if isinstance(ctl1, QuantumRegister) and \
       isinstance(ctl2, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and \
       len(ctl1) == len(tgt) and len(ctl2) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl1.size):
            instructions.add(self.ccx((ctl1, i), (ctl2, i), (tgt, i)))
        return instructions

    self._check_qubit(ctl1)
    self._check_qubit(ctl2)
    self._check_qubit(tgt)
    self._check_dups([ctl1, ctl2, tgt])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt, self))


QuantumCircuit.ccx = ccx
