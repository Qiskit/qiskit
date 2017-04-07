"""
T=sqrt(S) phase gate.

Author: Andrew Cross
"""
import math
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header
from qiskit.extensions.standard import u1


class TGate(CompositeGate):
    """T=sqrt(S) Clifford phase gate."""

    def __init__(self, qubit, circ=None):
        """Create new T gate."""
        super(TGate, self).__init__("t", [], [qubit], circ)
        self.u1(math.pi/4.0, qubit)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.t(self.arg[0]))


def t(self, q):
    """Apply T to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.t((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(TGate(q, self))


QuantumCircuit.t = t
CompositeGate.t = t
