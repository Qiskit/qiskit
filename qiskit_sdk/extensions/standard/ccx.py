"""
Toffoli gate. Controlled-Controlled-X.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import h, cx, t


class ToffoliGate(CompositeGate):
    """Toffoli gate."""

    def __init__(self, ctl1, ctl2, tgt):
        """Create new Toffoli gate."""
        super(ToffoliGate, self).__init__("ccx", [], [ctl1, ctl2, tgt])
        self.h(tgt)
        self.cx(ctl2, tgt)
        self.t(tgt).inverse()
        self.cx(ctl1, tgt)
        self.t(tgt)
        self.cx(ctl2, tgt)
        self.t(tgt).inverse()
        self.cx(ctl1, tgt)
        self.t(ctl2)
        self.t(tgt)
        self.h(tgt)
        self.cx(ctl1, ctl2)
        self.t(ctl1)
        self.t(ctl2).inverse()
        self.cx(ctl1, ctl2)


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
    """Apply Toffoli to program."""
    self._check_qreg(ctl1[0])
    ctl1[0].check_range(ctl1[1])
    self._check_qreg(ctl2[0])
    ctl2[0].check_range(ctl2[1])
    self._check_qreg(tgt[0])
    tgt[0].check_range(tgt[1])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt))


Program.ccx = ccx


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to composite."""
    self._check_qubit(ctl1)
    ctl1[0].check_range(ctl1[1])
    self._check_qubit(ctl2)
    ctl2[0].check_range(ctl2[1])
    self._check_qubit(tgt)
    tgt[0].check_range(tgt[1])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt))


CompositeGate.ccx = ccx
