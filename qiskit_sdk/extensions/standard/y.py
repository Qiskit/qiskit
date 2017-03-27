"""
Pauli Y (bit-phase-flip) gate.

Author: Andrew Cross
"""
import math
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import u3


class YGate(CompositeGate):
    """Pauli Y (bit-phase-flip) gate."""

    def __init__(self, qubit):
        """Create new Y gate."""
        super(YGate, self).__init__("y", [], [qubit])
        self.u3(math.pi, math.pi/2.0, math.pi/2.0, qubit)


def y(self, j=-1):
    """Apply Y to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.y((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.y((self, j)))
    return gs


QuantumRegister.y = y


def y(self, q):
    """Apply Y to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(YGate(q))


Program.y = y


def y(self, q):
    """Apply Y to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(YGate(q))


CompositeGate.y = y
