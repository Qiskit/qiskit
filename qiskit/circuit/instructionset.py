# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction collection.
"""
from qiskit.qiskiterror import QiskitError
from .instruction import Instruction


class InstructionSet(object):
    """Instruction collection."""

    def __init__(self):
        """New collection of instructions."""
        self.instructions = []

    def add(self, gate):
        """Add instruction to set."""
        if not isinstance(gate, Instruction):
            raise QiskitError("attempt to add non-Instruction" +
                              " to InstructionSet")
        self.instructions.append(gate)

    def inverse(self):
        """Invert all instructions."""
        for index, instruction in enumerate(self.instructions):
            self.instructions[index] = instruction.inverse()
        return self

    def q_if(self, *qregs):
        """Add controls to all instructions."""
        for gate in self.instructions:
            gate.q_if(*qregs)
        return self

    def c_if(self, classical, val):
        """Add classical control register to all instructions."""
        for gate in self.instructions:
            gate.c_if(classical, val)
        return self
