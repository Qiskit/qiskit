"""
Rotation around the z-axis.

Author: Jay using andrew code.
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class RZGate(Gate):
    """rotation around the z-axis."""

    def __init__(self, phi, qubit, circ=None):
        """Create new rz single qubit gate."""
        super(RZGate, self).__init__("rz", [phi], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        phi = self.param[0]
        return self._qasmif("rz(%.15f) %s[%d];" % (phi, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        rz(phi)^dagger = rz(-phi)
        """
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rz(self.param[0], self.arg[0]))


def rz(self, phi, q):
    """Apply rz to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.rx(phi, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(RZGate(phi, q, self))


QuantumCircuit.rz = rz
CompositeGate.rz = rz
