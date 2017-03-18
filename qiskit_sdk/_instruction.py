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
