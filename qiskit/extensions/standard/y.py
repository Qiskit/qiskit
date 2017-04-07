"""
Pauli Y (bit-phase-flip) gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class YGate(Gate):
    """Pauli Y (bit-phase-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new Y gate."""
        super(YGate, self).__init__("y", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("y %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.y(self.arg[0]))


def y(self, q):
    """Apply Y to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.y((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(YGate(q, self))


QuantumCircuit.y = y
CompositeGate.y = y
