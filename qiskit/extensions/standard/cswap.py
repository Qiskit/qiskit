# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Fredkin gate. Controlled-SWAP.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.ccx import ToffoliGate


class FredkinGate(Gate):
    """Fredkin gate."""

    def __init__(self, circ=None):
        """Create new Fredkin gate."""
        super().__init__("cswap", 3, [], circ)

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        definition = []
        q = QuantumRegister(3, "q")
        rule = [
            (CnotGate(), [q[2], q[1]], []),
            (ToffoliGate(), [q[0], q[1], q[2]], []),
            (CnotGate(), [q[2], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(3, broadcastable=[True, False, False])
def cswap(self, ctl, tgt1, tgt2):
    """Apply Fredkin to circuit."""
    return self.append(FredkinGate(self), [ctl, tgt1, tgt2], [])


QuantumCircuit.cswap = cswap
CompositeGate.cswap = cswap
