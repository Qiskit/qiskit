# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Hadamard gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.qasm import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u2 import U2Gate


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, circ=None):
        """Create new Hadamard gate."""
        super().__init__("h", 1, [], circ)

    def _define_decompositions(self):
        """
        gate h a { u2(0,pi) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U2Gate(0, pi), [q[0]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(1)
def h(self, q):
    """Apply H to q."""
    return self._attach(HGate(self), [q], [])


QuantumCircuit.h = h
CompositeGate.h = h
