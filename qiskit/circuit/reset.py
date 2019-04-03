# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qubit reset to computational zero.
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from .decorators import _op_expand


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self):
        """Create new reset instruction."""
        super().__init__("reset", 1, 0, [])


@_op_expand(1)
def reset(self, qubit):
    """Reset q."""
    return self.append(Reset(), [qubit], [])


QuantumCircuit.reset = reset
