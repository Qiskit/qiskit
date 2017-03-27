"""
Quantum computer instruction.

Author: Andrew Cross
"""
from ._register import Register
from ._qiskitexception import QISKitException


class Instruction(object):
    """Generic quantum computer instruction."""

    def __init__(self, name, param, arg):
        """Create a new instruction.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        """
        for a in arg:
            if not isinstance(a[0], Register):
                raise QISKitException("argument not (Register, int) tuple")
        self.name = name
        self.param = param
        self.arg = arg
        self.control = None
        self.program = None

    def set_program(self, p):
        """Point back to the program that contains this instruction."""
        self.program = p

    def c_if(self, c, val):
        """Add classical control on register c and value val."""
        if self.program is None:
            raise QISKitException("gate is not associated to a program")
        self.program._check_creg(c)
        if val < 0:
            raise QISKitException("control value should be non-negative")
        self.control = (c, val)

    def _qasmif(self, s):
        """Print an if statement if needed."""
        if self.control is None:
            return s
        else:
            return "if(%s==%d) " % (self.control[0].name, self.control[1]) + s
