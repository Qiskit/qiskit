"""
Barrier instruction.

Author: Andrew Cross
"""
from ._instruction import Instruction


class Barrier(Instruction):
    """Barrier instruction."""

    def __init__(self, *args):
        """Create new barrier instruction."""
        super(Barrier, self).__init__("barrier", [], list(args))

    def inverse(self):
        """Special case. Return self."""
        return self

    def qasm(self):
        """Return OPENQASM string."""
        s = "barrier "
        for j in range(len(self.arg)):
            if len(self.arg[j]) == 1:
                s += "%s" % self.arg[j].name
            else:
                s += "%s[%d]" % (self.arg[j][0].name, self.arg[j][1])
            if j != len(self.arg) - 1:
                s += ","
        s += ";"
        return s  # no c_if on barrier instructions
