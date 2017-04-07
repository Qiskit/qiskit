"""
Fredkin gate. Controlled-SWAP.

Author: Andrew Cross
"""
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit.extensions.standard import header
from qiskit.extensions.standard import cx, ccx


class FredkinGate(CompositeGate):
    """Fredkin gate."""

    def __init__(self, ctl, tgt1, tgt2, circ=None):
        """Create new Fredkin gate."""
        super(FredkinGate, self).__init__("fredkin", [], [ctl, tgt1, tgt2],
                                          circ)
        self.cx(tgt2, tgt1)
        self.ccx(ctl, tgt1, tgt2)
        self.cx(tgt2, tgt1)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cswap(self.arg[0], self.arg[1], self.arg[2]))


def cswap(self, ctl, tgt1, tgt2):
    """Apply Fredkin to circuit."""
    self._check_qubit(ctl)
    self._check_qubit(tgt1)
    self._check_qubit(tgt2)
    self._check_dups([ctl, tgt1, tgt2])
    return self._attach(FredkinGate(ctl, tgt1, tgt2, self))


QuantumCircuit.cswap = cswap
CompositeGate.cswap = cswap
