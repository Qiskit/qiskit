"""
Instruction collection.

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._qiskitexception import QISKitException


class InstructionSet(object):
    """Instruction collection."""

    def __init__(self):
        """New collection of instructions."""
        self.instructions = set([])

    def add(self, gate):
        """Add instruction to set."""
        if not isinstance(gate, Instruction):
            raise QISKitException("attempt to add non-Instruction" +
                                  " to InstructionSet")
        self.instructions.add(gate)

    def inverse(self):
        """Invert all instructions."""
        for instruction in self.instructions:
            instruction.inverse()
        return self

    def q_if(self, *qregs):
        """Add controls to all instructions."""
        for gate in self.instructions:
            gate.q_if(*qregs)
        return self

    def c_if(self, c, val):
        """Add classical control register to all instructions."""
        for gate in self.instructions:
            gate.c_if(c, val)
        return self
