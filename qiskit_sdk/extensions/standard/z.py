"""
Pauli Z (phase-flip) gate.

Author: Andrew Cross
"""
import math
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import CompositeGate
from qiskit_sdk import InstructionSet
from qiskit_sdk.extensions.standard import u1


class ZGate(CompositeGate):
    """Pauli Z (phase-flip) gate."""

    def __init__(self, qubit):
        """Create new Z gate."""
        super(ZGate, self).__init__("z", [], [qubit])
        self.u1(math.pi, qubit)


def z(self, j=-1):
    """Apply Z to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.z((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.z((self, j)))
    return gs


QuantumRegister.z = z


def z(self, q):
    """Apply Z to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(ZGate(q))


Program.z = z


def z(self, q):
    """Apply Z to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(ZGate(q))


CompositeGate.z= z
