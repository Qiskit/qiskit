# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-NOT gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.cxbase import CXBase


class CnotGate(Gate):
    """controlled-NOT gate."""

    def __init__(self, circ=None):
        """Create new CNOT gate."""
        super().__init__("cx", 2, [], circ)

    def _define(self):
        """
        gate cx c,t { CX c,t; }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (CXBase(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(2)
def cx(self, ctl, tgt):
    """Apply CX from ctl to tgt."""
    return self.append(CnotGate(self), [ctl, tgt], [])


QuantumCircuit.cx = cx
CompositeGate.cx = cx
