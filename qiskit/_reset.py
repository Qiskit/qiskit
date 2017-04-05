"""
Qubit reset to computational zero.

Author: Andrew Cross
"""
from ._instruction import Instruction


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self, qubit):
        """Create new reset instruction."""
        super(Reset, self).__init__("reset", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("reset %s[%d];" % (qubit[0].name, qubit[1]))

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.reset(rearg[0]))
