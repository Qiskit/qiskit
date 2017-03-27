"""
Unitary gate.

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._qiskitexception import QISKitException


class Gate(Instruction):
    """Unitary gate."""

    def inverse(self):
        """Invert this gate."""
        raise QISKitException("inverse not implemented")

    def control(self, *qregs):
        """Add controls to this gate."""
        raise QISKitException("control not implemented")

    def doif(self, c, val):
        """Add classical control register."""
        raise QISKitException("doif not implemented")
