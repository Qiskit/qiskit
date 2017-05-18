"""
Base register reference object.

Author: Andrew Cross
"""
from ._qiskitexception import QISKitException
import re


class Register(object):
    """Implement a generic register."""

    def __init__(self, name, size):
        """Create a new generic register."""
        test = re.compile('[a-z][a-zA-Z0-9_]*')
        if test.match(name) is None:
            raise QISKitException("invalid OPENQASM register name")
        self.name = name
        self.size = size
        if size <= 0:
            raise QISKitException("register size must be positive")

    def __str__(self):
        """Return a string representing the register."""
        return "Register(%s,%d)" % (self.name, self.size)

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.size:
            raise QISKitException("register index out of range")

    def __getitem__(self, key):
        """Return tuple (self, key) if key is valid."""
        if not isinstance(key, int):
            raise QISKitException("expected integer index into register")
        self.check_range(key)
        return (self, key)
