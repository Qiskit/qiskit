# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qubit reset to computational zero.
"""
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, InstructionSet
from .decorators import _op_expand


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self, circ=None):
        """Create new reset instruction."""
        super().__init__("reset", 1, 0, [], circ)


@_op_expand(1)
def reset(self, qubit):
    """Reset q."""
    if isinstance(qubit, QuantumRegister):
        instructions = InstructionSet()
        for sizes in range(qubit.size):
            instructions.add(self.reset((qubit, sizes)))
        return instructions
    return self.append(Reset(self), [qubit], [])


QuantumCircuit.reset = reset
