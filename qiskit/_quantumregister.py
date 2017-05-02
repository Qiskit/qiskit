"""
Quantum register reference object.

Author: Andrew Cross
"""
from ._register import Register
from ._instructionset import InstructionSet


class QuantumRegister(Register):
    """Implement a quantum register."""

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)

    def __str__(self):
        """Return a string representing the register."""
        return "QuantumRegister(%s,%d)" % (self.name, self.size)
