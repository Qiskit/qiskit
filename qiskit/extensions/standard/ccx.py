# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Toffoli gate. Controlled-Controlled-X.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.t import TGate
from qiskit.extensions.standard.t import TdgGate


class ToffoliGate(Gate):
    """Toffoli gate."""

    def __init__(self, circ=None):
        """Create new Toffoli gate."""
        super().__init__("ccx", 3, [], circ)

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
        rule = [
            (HGate(), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CnotGate(), [q[0], q[2]], []),
            (TGate(), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CnotGate(), [q[0], q[2]], []),
            (TGate(), [q[1]], []),
            (TGate(), [q[2]], []),
            (HGate(), [q[2]], []),
            (CnotGate(), [q[0], q[1]], []),
            (TGate(), [q[0]], []),
            (TdgGate(), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(3, broadcastable=[True, True, False])
def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to from ctl1 and ctl2 to tgt."""
    return self.append(ToffoliGate(self), [ctl1, ctl2, tgt], [])


QuantumCircuit.ccx = ccx
CompositeGate.ccx = ccx
