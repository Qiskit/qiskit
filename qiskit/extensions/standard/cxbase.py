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

    def __init__(self, ctl, tgt, circ=None):
        """Create new CX instruction."""
        super().__init__("CX", [], [ctl, tgt], circ)

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cx_base(self.qargs[0], self.qargs[1]))


@_op_expand(2)
def cx_base(self, ctl, tgt):
    """Apply CX ctl, tgt."""
    return self._attach(CXBase(ctl, tgt, self))


QuantumCircuit.cx_base = cx_base
CompositeGate.cx_base = cx_base
