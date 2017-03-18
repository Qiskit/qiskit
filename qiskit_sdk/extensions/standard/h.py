"""
Hadamard gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import Gate
from qiskit_sdk import InstructionSet


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, qubit):
        """Create new Hadamard gate."""
        super(HGate, self).__init__("h", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return "h %s[%d];" % (qubit[0].name, qubit[1])


def h(self, j=-1):
    """Apply Hadamard to jth qubit in this register (or all)."""
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
    """Apply Hadamard to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(HGate(q))


Program.h = h
