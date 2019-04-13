# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Identity gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.u3 import U3Gate


class IdGate(Gate):
    """Identity gate."""

    def __init__(self):
        """Create new Identity gate."""
        super().__init__("id", 1, [])

    def _define(self):
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U3Gate(0, 0, 0), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return IdGate()  # self-inverse


@_op_expand(1)
def iden(self, q):
    """Apply Identity to q."""
    return self.append(IdGate(), [q], [])


QuantumCircuit.iden = iden
CompositeGate.iden = iden
