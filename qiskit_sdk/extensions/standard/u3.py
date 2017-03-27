"""
Two-pulse single qubit gate.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister
from qiskit_sdk import Program
from qiskit_sdk import Gate
from qiskit_sdk import InstructionSet
from qiskit_sdk import CompositeGate


class U3Gate(Gate):
    """Two-pulse single qubit gate."""

    def __init__(self, theta, phi, lam, qubit):
        """Create new two-pulse single qubit gate."""
        super(U3Gate, self).__init__("u3", [theta, phi, lam], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        phi = self.param[1]
        lam = self.param[2]
        return self._qasmif("u3(%.15f,%.15f,%.15f) %s[%d];" % (theta, phi, lam,
                                                               qubit[0].name,
                                                               qubit[1]))

    def inverse(self):
        """Invert this gate.

        u3(theta,phi,lamb)^dagger = u3(-theta, -lam, -phi)
        """
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self


def u3(self, theta, phi, lam, j=-1):
    """Apply u3 to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.u3(theta, phi, lam, (self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.u3(theta, phi, lam, (self, j)))
    return gs


QuantumRegister.u3 = u3


def u3(self, theta, phi, lam, q):
    """Apply u3 to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(U3Gate(theta, phi, lam, q))


Program.u3 = u3


def u3(self, theta, phi, lam, q):
    """Apply u3 to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(U3Gate(theta, phi, lam, q))


CompositeGate.u3 = u3
