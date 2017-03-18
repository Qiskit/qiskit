"""
Fundamental controlled-NOT gate.

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._gate import Gate


class CXBase(Gate):
    """Fundamental controlled-NOT gate."""

    def __init__(self, ctl, tgt):
        """Create new CX instruction."""
        super(Instruction, self).__init__("CX", [], [ctl, tgt])

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return "CX %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                      tgt[0].name, tgt[1])
