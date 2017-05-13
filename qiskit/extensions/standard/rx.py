"""Rotation around the x-axis."""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class RXGate(Gate):
    """rotation around the x-axis."""

    def __init__(self, theta, qubit, circ=None):
        """Create new rx single qubit gate."""
        super(RXGate, self).__init__("rx", [theta], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return self._qasmif("rx(%.15f) %s[%d];" % (theta, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        rx(theta)^dagger = rx(-theta)
        """
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rx(self.param[0], self.arg[0]))


def rx(self, theta, q):
    """Apply rx to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.rx(theta, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(RXGate(theta, q, self))


QuantumCircuit.rx = rx
CompositeGate.rx = rx
