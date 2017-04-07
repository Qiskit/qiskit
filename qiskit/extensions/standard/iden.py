"""
Identity gate.

Author: Andrew Cross
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class IdGate(Gate):
    """Identity gate."""

    def __init__(self, qubit, circ=None):
        """Create new Identity gate."""
        super(IdGate, self).__init__("id", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("id %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.id(self.arg[0], self))


def iden(self, q):
    """Apply Identity to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.iden((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(IdGate(q, self))


QuantumCircuit.iden = iden
CompositeGate.iden = iden
