"""
controlled-Phase gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import h, cx


class CzGate(CompositeGate):
    """CZ gate."""

    def __init__(self, ctl, tgt):
        """Create new CZ gate."""
        super(CzGate, self).__init__("cz", [], [ctl, tgt])
        self.h(tgt)
        self.cx(ctl, tgt)
        self.h(tgt)


def cz(self, i, j):
    """Apply CZ from i to j in this register."""
    self._check_bound()
    gs = InstructionSet()
    self.check_range(i)
    self.check_range(j)
    for p in self.bound_to:
        gs.add(p.cz((self, i), (self, j)))
    return gs


QuantumRegister.cz = cz


def cz(self, ctl, tgt):
    """Apply CZ to program."""
    self._check_qreg(ctl[0])
    ctl[0].check_range(ctl[1])
    self._check_qreg(tgt[0])
    tgt[0].check_range(tgt[1])
    return self._attach(CzGate(ctl, tgt))


Program.cz = cz


def cz(self, ctl, tgt):
    """Apply CZ to composite."""
    self._check_qubit(ctl)
    ctl[0].check_range(ctl[1])
    self._check_qubit(tgt)
    tgt[0].check_range(tgt[1])
    return self._attach(CzGate(ctl, tgt))


CompositeGate.cz = cz
