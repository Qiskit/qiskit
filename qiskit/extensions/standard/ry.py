"""Rotation around the y-axis."""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class RYGate(Gate):
    """rotation around the y-axis."""

    def __init__(self, theta, qubit, circ=None):
        """Create new ry single qubit gate."""
        super(RYGate, self).__init__("ry", [theta], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return self._qasmif("ry(%.15f) %s[%d];" % (theta, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ry(self.param[0], self.arg[0]))


def ry(self, theta, q):
    """Apply ry to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.ry(theta, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(RYGate(theta, q, self))


QuantumCircuit.ry = ry
CompositeGate.ry = ry
