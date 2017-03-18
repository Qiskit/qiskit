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

    def invert(self):
        """Invert all instructions."""
        for g in self.gs:
            g.invert()

    def control(self, *qregs):
        """Add controls to all instructions."""
        for g in self.gs:
            g.control(*qregs)

    def doif(self, c, val):
        """Add classical control register to all instructions."""
        for g in self.gs:
            g.doif(c, val)
