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
        # Implemented as a list to allow the possibility that register
        # references are bound to qubits in multiple quantum circuits,
        # but we have implemented a bind_to that only allows one circuit.
        self.bound_to = []
        if sz <= 0:
            raise QISKitException("register size must be positive")

    def unbound_copy(self):
        """Return a new register that is not bound to any circuit."""
        return self.__class__(self.name, self.sz)

    def bind_to(self, prog):
        """Bind register to quantum circuit."""
        # Raise an exception if we are already bound to a circuit
        if len(self.bound_to) > 0 and self.bound_to[0] != prog:
            raise QISKitException("register already bound to a circuit")
        self.bound_to.append(prog)

    def _check_bound(self):
        """Check that the register is bound to a circuit."""
        if len(self.bound_to) == 0:
            raise QISKitException("register not bound to circuit")

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
