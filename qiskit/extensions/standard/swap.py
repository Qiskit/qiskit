# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
SWAP gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.cx import CnotGate


class SwapGate(Gate):
    """SWAP gate."""

    def __init__(self, circ=None):
        """Create new SWAP gate."""
        super().__init__("swap", 2, [], circ)

    def _define_decompositions(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        rule = [
            (CnotGate(), [q[0], q[1]], []),
            (CnotGate(), [q[1], q[0]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(2, broadcastable=[False, False])
def swap(self, qubit1, qubit2):
    """Apply SWAP from qubit1 to qubit2."""
    return self._attach(SwapGate(self), [qubit1, qubit2], [])


QuantumCircuit.swap = swap
CompositeGate.swap = swap
