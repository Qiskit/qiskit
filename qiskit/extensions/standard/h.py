"""
Hadamard gate.

Author: Andrew Cross
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self, qubit, circ=None):
        """Create new Hadamard gate."""
        super(HGate, self).__init__("h", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("h %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.h(self.arg[0], self))


def h(self, q):
    """Apply H to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.h((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(HGate(q, self))


QuantumCircuit.h = h
CompositeGate.h = h
