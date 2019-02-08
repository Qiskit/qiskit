# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Pauli X (bit-flip) gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm import pi
from qiskit.extensions.standard.u3 import U3Gate


class XGate(Gate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, circ=None):
        """Create new X gate."""
        super().__init__("x", [], circ)

    def _define_decompositions(self):
        """
        gate x a {
        u3(pi,0,pi) a;
        }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U3Gate(pi, 0, pi), [q[0]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(1)
def x(self, q):
    """Apply X to q."""
    return self._attach(XGate(self), [q], [])


QuantumCircuit.x = x
CompositeGate.x = x
