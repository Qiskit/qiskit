"""
Base register reference object.

Author: Andrew Cross
"""
from ._qiskitexception import QISKitException


class Register(object):
    """Implement a generic register."""

    def __init__(self, name, sz):
        """Create a new generic register."""
        self.name = name
        self.sz = sz
        # Allow the possibility that register references
        # are bound to multiple programs.
        self.bound_to = []
        if sz <= 0:
            raise QISKitException("register size must be positive")

    def bind_to(self, prog):
        """Bind register to program."""
        if prog not in self.bound_to:
            self.bound_to.append(prog)

    def _check_bound(self):
        """Check that the register is bound to a program."""
        if len(self.bound_to) == 0:
            raise QISKitException("register not bound to program")

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.sz:
            raise QISKitException("register index out of range")
