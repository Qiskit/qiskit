"""
controlled-Y gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import s, cx


class CyGate(CompositeGate):
    """CY gate."""

    def __init__(self, ctl, tgt):
        """Create new CY gate."""
        super(CyGate, self).__init__("cy", [], [ctl, tgt])
        self.s(tgt).inverse()
        self.cx(ctl, tgt)
        self.s(tgt)


def cy(self, i, j):
    """Apply CY from i to j in this register."""
    self._check_bound()
    gs = InstructionSet()
    self.check_range(i)
    self.check_range(j)
    for p in self.bound_to:
        gs.add(p.cy((self, i), (self, j)))
    return gs


QuantumRegister.cy = cy


def cy(self, ctl, tgt):
    """Apply CY to program."""
    self._check_qreg(ctl[0])
    ctl[0].check_range(ctl[1])
    self._check_qreg(tgt[0])
    tgt[0].check_range(tgt[1])
    return self._attach(CyGate(ctl, tgt))


Program.cy = cy


def cy(self, ctl, tgt):
    """Apply CY to composite."""
    self._check_qubit(ctl)
    ctl[0].check_range(ctl[1])
    self._check_qubit(tgt)
    tgt[0].check_range(tgt[1])
    return self._attach(CyGate(ctl, tgt))


CompositeGate.cy = cy
