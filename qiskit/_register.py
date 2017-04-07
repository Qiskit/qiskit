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
        if sz <= 0:
            raise QISKitException("register size must be positive")

    def __str__(self):
        """Return a string representing the register."""
        return "Register(%s,%d)" % (self.name, self.sz)

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.sz:
            raise QISKitException("register index out of range")

    def __getitem__(self, key):
        """Return tuple (self, key) if key is valid."""
        if not isinstance(key, int):
            raise QISKitException("expected integer index into register")
        self.check_range(key)
        return (self, key)
