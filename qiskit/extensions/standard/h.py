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
from qiskit.extensions.standard.u2 import U2Gate


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, circ=None):
        """Create new Hadamard gate."""
        super().__init__("h", 1, [], circ)

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U2Gate(0, pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(1)
def h(self, q):
    """Apply H to q."""
    return self.append(HGate(self), [q], [])


QuantumCircuit.h = h
CompositeGate.h = h
