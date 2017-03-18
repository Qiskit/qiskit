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

    def qasm(self):
        """Return OPENQASM string."""
        s = "barrier "
        for j in range(len(self.args)):
            if len(self.args[j]) == 1:
                s += "%s" % self.args[j].name
            else:
                s += "%s[%d]" % (self.args[j][0].name, self.args[j][1])
            if j != len(self.args) - 1:
                s += ","
        s += ";"
        return s
