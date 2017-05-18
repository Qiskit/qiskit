"""
Diagonal single qubit gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class U1Gate(Gate):
    """Diagonal single-qubit gate."""

    def __init__(self, theta, qubit, circ=None):
        """Create new diagonal single-qubit gate."""
        super(U1Gate, self).__init__("u1", [theta], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return self._qasmif("u1(%.15f) %s[%d];" % (theta, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u1(self.param[0], self.arg[0]))


def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.u1(theta, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(U1Gate(theta, q, self))


QuantumCircuit.u1 = u1
CompositeGate.u1 = u1
