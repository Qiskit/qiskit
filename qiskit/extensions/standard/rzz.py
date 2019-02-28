# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
two-qubit ZZ-rotation gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.cx import CnotGate


class RZZGate(Gate):
    """Two-qubit ZZ-rotation gate."""

    def __init__(self, theta, circ=None):
        """Create new rzz gate."""
        super().__init__("rzz", 2, [theta], circ)

    def _define_decompositions(self):
        """
        gate rzz(theta) a, b { cx a, b; u1(theta) b; cx a, b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        rule = [
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0]), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self.params[0] = -self.params[0]
        self._decompositions = None
        return self


@_op_expand(2, broadcastable=[False, False])
def rzz(self, theta, qubit1, qubit2):
    """Apply RZZ to circuit."""
    return self._attach(RZZGate(theta, self), [qubit1, qubit2], [])


# Add to QuantumCircuit and CompositeGate classes
QuantumCircuit.rzz = rzz
CompositeGate.rzz = rzz
