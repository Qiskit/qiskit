"""
Identity gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import Gate
from qiskit_sdk import InstructionSet
from qiskit_sdk import CompositeGate


class IdGate(Gate):
    """Identity gate."""

    def __init__(self, qubit):
        """Create new Identity gate."""
        super(IdGate, self).__init__("id", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("id %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


def iden(self, j=-1):
    """Apply Identity to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.iden((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.iden((self, j)))
    return gs


QuantumRegister.iden = iden


def iden(self, q):
    """Apply Identity to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(IdGate(q))


Program.iden = iden


def iden(self, q):
    """Apply Identity to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(IdGate(q))


CompositeGate.iden = iden
