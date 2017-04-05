"""
Fredkin gate. Controlled-SWAP.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header
from qiskit.extensions.standard import cx, ccx


class FredkinGate(CompositeGate):
    """Fredkin gate."""

    def __init__(self, ctl, tgt1, tgt2):
        """Create new Fredkin gate."""
        super(FredkinGate, self).__init__("fredkin", [], [ctl, tgt1, tgt2])
        self.cx(tgt2, tgt1)
        self.ccx(ctl, tgt1, tgt2)
        self.cx(tgt2, tgt1)

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.cswap(rearg[0], rearg[1], rearg[2]))


def cswap(self, i, j, k):
    """Apply Fredkin from i to j, k in this register."""
    self._check_bound()
    gs = InstructionSet()
    self.check_range(i)
    self.check_range(j)
    self.check_range(k)
    for p in self.bound_to:
        gs.add(p.cswap((self, i), (self, j), (self, k)))
    return gs


QuantumRegister.cswap = cswap


def cswap(self, ctl, tgt1, tgt2):
    """Apply Fredkin to circuit."""
    self._check_qreg(ctl[0])
    ctl[0].check_range(ctl[1])
    self._check_qreg(tgt1[0])
    tgt1[0].check_range(tgt1[1])
    self._check_qreg(tgt2[0])
    tgt2[0].check_range(tgt2[1])
    self._check_dups([ctl, tgt1, tgt2])
    return self._attach(FredkinGate(ctl, tgt1, tgt2))


QuantumCircuit.cswap = cswap


def cswap(self, ctl, tgt1, tgt2):
    """Apply Fredkin to composite."""
    self._check_qubit(ctl)
    ctl[0].check_range(ctl[1])
    self._check_qubit(tgt1)
    tgt1[0].check_range(tgt1[1])
    self._check_qubit(tgt2)
    tgt2[0].check_range(tgt2[1])
    self._check_dups([ctl, tgt1, tgt2])
    return self._attach(FredkinGate(ctl, tgt1, tgt2))


CompositeGate.cswap = cswap
