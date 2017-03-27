"""
Pauli X (bit-flip) gate.

Author: Andrew Cross
"""
import math
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import u3


class XGate(CompositeGate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, qubit):
        """Create new X gate."""
        super(XGate, self).__init__("x", [], [qubit])
        self.u3(math.pi, 0.0, math.pi, qubit)


def x(self, j=-1):
    """Apply X to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.x((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.x((self, j)))
    return gs


QuantumRegister.x = x


def x(self, q):
    """Apply X to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(XGate(q))


Program.x = x


def x(self, q):
    """Apply X to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(XGate(q))


CompositeGate.x = x
