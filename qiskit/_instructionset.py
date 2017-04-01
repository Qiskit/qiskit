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
        self.gs = set([])

    def add(self, g):
        """Add instruction to set."""
        if not isinstance(g, Instruction):
            raise QISKitException("attempt to add non-Instruction" +
                                  " to InstructionSet")
        self.gs.add(g)

    def inverse(self):
        """Invert all instructions."""
        for g in self.gs:
            g.inverse()
        return self

    def q_if(self, *qregs):
        """Add controls to all instructions."""
        for g in self.gs:
            g.q_if(*qregs)
        return self

    def c_if(self, c, val):
        """Add classical control register to all instructions."""
        for g in self.gs:
            g.c_if(c, val)
        return self
