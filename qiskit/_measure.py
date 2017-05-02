"""
Quantum measurement in the computational basis.

Author: Andrew Cross
"""
from ._instruction import Instruction


class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    def __init__(self, qubit, bit, circuit=None):
        """Create new measurement instruction."""
        super(Measure, self).__init__("measure", [], [qubit, bit], circuit)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        bit = self.arg[1]
        return self._qasmif("measure %s[%d] -> %s[%d];" % (qubit[0].name,
                                                           qubit[1],
                                                           bit[0].name,
                                                           bit[1]))

    def reapply(self, circuit):
        """Reapply this gate to corresponding qubits."""
        self._modifiers(circuit.measure(self.arg[0], self.arg[1]))
