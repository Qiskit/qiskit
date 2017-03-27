"""
Hadamard gate.

Author: Andrew Cross
"""
import math
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import u2


class HGate(CompositeGate):
    """Hadamard gate."""

    def __init__(self, qubit):
        """Create new Hadamard gate."""
        super(HGate, self).__init__("h", [], [qubit])
        self.u2(0.0, math.pi, qubit)


def h(self, j=-1):
    """Apply H to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.h((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.h((self, j)))
    return gs


QuantumRegister.h = h


def h(self, q):
    """Apply H to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(HGate(q))


Program.h = h


def h(self, q):
    """Apply H to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(HGate(q))


CompositeGate.h = h
