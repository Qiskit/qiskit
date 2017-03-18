"""
Quantum measurement in the computational basis.

Author: Andrew Cross
"""
from ._instruction import Instruction


class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    def __init__(self, qubit, bit):
        """Create new measurement instruction."""
        super(Instruction, self).__init__("measure", [], [qubit, bit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        bit = self.arg[1]
        return "measure %s[%d] -> %s[%d];" % (qubit[0].name, qubit[1],
                                              bit[0].name, bit[1])
