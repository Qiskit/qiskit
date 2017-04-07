"""
Classical register reference object.

Author: Andrew Cross
"""
from ._register import Register


class ClassicalRegister(Register):
    """Implement a classical register."""

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "creg %s[%d];" % (self.name, self.sz)

    def __str__(self):
        """Return a string representing the register."""
        return "ClassicalRegister(%s,%d)" % (self.name, self.sz)
