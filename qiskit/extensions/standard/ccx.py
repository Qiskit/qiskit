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
from qiskit.exceptions import QiskitError


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
    # expand registers to lists of qubits
    if isinstance(ctl1, QuantumRegister):
        ctl1 = [(ctl1, i) for i in range(len(ctl1))]
    if isinstance(ctl2, QuantumRegister):
        ctl2 = [(ctl2, i) for i in range(len(ctl2))]
    if isinstance(tgt, QuantumRegister):
        tgt = [(tgt, i) for i in range(len(tgt))]
    # expand single qubit target if controls are lists of qubits
    if isinstance(ctl1, list) and len(ctl1) == len(ctl2):
        if isinstance(tgt, tuple):
            tgt = [tgt]
        if len(tgt) == 1:
            tgt = tgt * len(ctl1)
        elif len(tgt) != len(ctl1):
            raise QiskitError('target register size should match controls or be one')

    if ctl1 and ctl2 and tgt:
        if isinstance(ctl1, list) and \
           isinstance(ctl2, list) and \
           isinstance(tgt, list):
            if len(ctl1) == len(tgt) and len(ctl2) == len(tgt):
                instructions = InstructionSet()
                for ictl1, ictl2, itgt in zip(ctl1, ctl2, tgt):
                    instructions.add(self.ccx(ictl1, ictl2, itgt))
                return instructions
            else:
                raise QiskitError('unequal register sizes')
    else:
        raise QiskitError('empty control or target argument')

    self._check_qubit(ctl1)
    self._check_qubit(ctl2)
    self._check_qubit(tgt)
    self._check_dups([ctl1, ctl2, tgt])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt, self))


QuantumCircuit.ccx = ccx
