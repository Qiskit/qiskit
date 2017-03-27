"""
One-pulse single qubit gate.

Author: Andrew Cross
"""
import math
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import Gate
from qiskit_sdk import InstructionSet
from qiskit_sdk import CompositeGate


class U2Gate(Gate):
    """One-pulse single qubit gate."""

    def __init__(self, phi, lam, qubit):
        """Create new one-pulse single qubit gate."""
        super(U2Gate, self).__init__("u2", [phi, lam], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        phi = self.param[0]
        lam = self.param[1]
        return "u2(%.15f,%.15f) %s[%d];" % (phi, lam, qubit[0].name, qubit[1])

    def inverse(self):
        """Invert this gate.

        u2(phi,lamb)^dagger = u2(-lamb-pi,-phi+pi)
        """
        phi = self.param[0]
        self.param[0] = -self.param[1] - math.pi
        self.param[1] = -phi + math.pi
        return self


def u2(self, phi, lam, j=-1):
    """Apply u2 to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.u2(phi, lam, (self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.u2(phi, lam, (self, j)))
    return gs


QuantumRegister.u2 = u2


def u2(self, phi, lam, q):
    """Apply u2 to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(U2Gate(phi, lam, q))


Program.u2 = u2


def u2(self, phi, lam, q):
    """Apply u2 to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(U2Gate(phi, lam, q))


CompositeGate.u2 = u2
