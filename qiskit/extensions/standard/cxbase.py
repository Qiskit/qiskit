# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Fundamental controlled-NOT gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.decorators import _op_expand


class CXBase(Gate):  # pylint: disable=abstract-method
    """Fundamental controlled-NOT gate."""

    def __init__(self):
        """Create new CX instruction."""
        super().__init__("CX", 2, [])

    def inverse(self):
        """Invert this gate."""
        return CXBase()  # self-inverse


@_op_expand(2)
def cx_base(self, ctl, tgt):
    """Apply CX ctl, tgt."""
    return self.append(CXBase(), [ctl, tgt], [])


QuantumCircuit.cx_base = cx_base
CompositeGate.cx_base = cx_base
