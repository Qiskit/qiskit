"""
Hadamard gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import Gate
from qiskit_sdk import GateSet


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, qubit):
        """Create new Hadamard gate."""
        super(Gate, self).__init__("h", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return "h %s[%d];" % (qubit[0].name, qubit[1])


def h(self, j=-1):
    """Apply Hadamard to jth qubit in this register (or all)."""
    if j == -1:
        gs = GateSet()
        for k in range(self.sz):
            gs.add(self.h(k))
        return gs
    else:
        self._check_range(j)
        return self._attach(HGate((self, j)))


QuantumRegister.h = h


def h(self, q):
    """Apply Hadamard to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(HGate(q), q)


Program.h = h
