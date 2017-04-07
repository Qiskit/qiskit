"""
Pauli X (bit-flip) gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class XGate(Gate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new X gate."""
        super(XGate, self).__init__("x", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("x %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.x(self.arg[0]))


def x(self, q):
    """Apply X to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.x((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(XGate(q, self))


QuantumCircuit.x = x
CompositeGate.x = x
