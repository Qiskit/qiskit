"""
S=diag(1,i) Clifford phase gate.

Author: Andrew Cross
"""
import math
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header
from qiskit.extensions.standard import u1


class SGate(CompositeGate):
    """S=diag(1,i) Clifford phase gate."""

    def __init__(self, qubit, circ=None):
        """Create new S gate."""
        super(SGate, self).__init__("s", [], [qubit], circ)
        self.u1(math.pi/2.0, qubit)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.s(self.arg[0]))


def s(self, q):
    """Apply S to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.s((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(SGate(q, self))


QuantumCircuit.s = s
CompositeGate.s = s
