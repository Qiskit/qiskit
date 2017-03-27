"""
Diagonal single qubit gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import Gate
from qiskit_sdk import InstructionSet
from qiskit_sdk import CompositeGate


class U1Gate(Gate):
    """Diagonal single qubit gate."""

    def __init__(self, theta, qubit):
        """Create new diagonal single qubit gate."""
        super(U1Gate, self).__init__("u1", [theta], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return "u1(%.15f) %s[%d];" % (theta, qubit[0].name, qubit[1])

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        return self


def u1(self, theta, j=-1):
    """Apply u1 with angle theta to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.u1(theta, (self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.u1(theta, (self, j)))
    return gs


QuantumRegister.u1 = u1


def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(U1Gate(theta, q))


Program.u1 = u1


def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(U1Gate(theta, q))


CompositeGate.u1 = u1
