"""
Two-pulse single qubit gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class U3Gate(Gate):
    """Two-pulse single qubit gate."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        """Create new two-pulse single qubit gate."""
        super(U3Gate, self).__init__("u3", [theta, phi, lam], [qubit], circ)

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

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u3(self.param[0], self.param[1], self.param[2],
                                self.arg[0]))


def u3(self, theta, phi, lam, q):
    """Apply u3 to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.u3(theta, phi, lam, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(U3Gate(theta, phi, lam, q, self))


QuantumCircuit.u3 = u3
CompositeGate.u3 = u3
