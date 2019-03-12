# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Single qubit gate cycle idle.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.ubase import UBase


class U0Gate(Gate):
    """Wait gate."""

    def __init__(self, m, circ=None):
        """Create new u0 gate."""
        super().__init__("u0", 1, [m], circ)

    def _define(self):
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (UBase(0, 0, 0), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(1)
def u0(self, m, q):
    """Apply u0 with length m to q."""
    return self.append(U0Gate(m, self), [q], [])


QuantumCircuit.u0 = u0
CompositeGate.u0 = u0
