"""
Qubit reset to computational zero.

Author: Andrew Cross
"""
from ._instruction import Instruction


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self, qubit):
        """Create new reset instruction."""
        super(Instruction, self).__init__("reset", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return "reset %s[%d];" % (qubit[0].name, qubit[1])
