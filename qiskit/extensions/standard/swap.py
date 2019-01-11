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
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _2q_gate
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


@_2q_gate
def swap(self, qubit1, qubit2):
    """Apply SWAP from qubit1 to qubit2."""
    self._check_qubit(qubit1)
    self._check_qubit(qubit2)
    self._check_dups([qubit1, qubit2])
    return self._attach(SwapGate(qubit1, qubit2, self))


QuantumCircuit.swap = swap
