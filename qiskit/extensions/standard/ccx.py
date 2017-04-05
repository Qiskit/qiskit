"""
Toffoli gate. Controlled-Controlled-X.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class ToffoliGate(Gate):
    """Toffoli gate."""

    def __init__(self, ctl1, ctl2, tgt):
        """Create new Toffoli gate."""
        super(ToffoliGate, self).__init__("ccx", [], [ctl1, ctl2, tgt])

    def qasm(self):
        """Return OPENQASM string."""
        ctl1 = self.arg[0]
        ctl2 = self.arg[1]
        tgt = self.arg[2]
        return self._qasmif("ccx %s[%d],%s[%d],%s[%d];" % (ctl1[0].name,
                                                           ctl1[1],
                                                           ctl2[0].name,
                                                           ctl2[1],
                                                           tgt[0].name,
                                                           tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.ccx(rearg[0], rearg[1], rearg[2]))


def ccx(self, i, j, k):
    """Apply Toffoli from i,j to k in this register."""
    self._check_bound()
    gs = InstructionSet()
    self.check_range(i)
    self.check_range(j)
    self.check_range(k)
    for p in self.bound_to:
        gs.add(p.ccx((self, i), (self, j), (self, k)))
    return gs


QuantumRegister.ccx = ccx


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to circuit."""
    self._check_qreg(ctl1[0])
    ctl1[0].check_range(ctl1[1])
    self._check_qreg(ctl2[0])
    ctl2[0].check_range(ctl2[1])
    self._check_qreg(tgt[0])
    tgt[0].check_range(tgt[1])
    self._check_dups([ctl1, ctl2, tgt])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt))


QuantumCircuit.ccx = ccx


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to composite."""
    self._check_qubit(ctl1)
    ctl1[0].check_range(ctl1[1])
    self._check_qubit(ctl2)
    ctl2[0].check_range(ctl2[1])
    self._check_qubit(tgt)
    tgt[0].check_range(tgt[1])
    self._check_dups([ctl1, ctl2, tgt])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt))


CompositeGate.ccx = ccx
