"""
Element of SU(2).

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._gate import Gate
from ._qiskitexception import QISKitException


class UBase(Gate):
    """Element of SU(2)."""

    def __init__(self, param, qubit):
        """Create new reset instruction."""
        if len(param) != 3:
            raise QISKitException("expected 3 parameters")
        super(Instruction, self).__init__("U", param, [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        theta = self.param[0]
        phi = self.param[1]
        lamb = self.param[2]
        qubit = self.arg[0]
        return "U(%.15f,%.15f,%.15f) %s[%d];" % (theta, phi, lamb,
                                                 qubit[0].name, qubit[1])
