"""
T=sqrt(S) phase gate or its inverse.

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
    """T=sqrt(S) Clifford phase gate or its inverse."""

    def __init__(self, qubit, circ=None):
        """Create new T gate."""
        super(TGate, self).__init__("t", [], [qubit], circ)
        self.u1(math.pi / 4.0, qubit)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.t(self.arg[0]))

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.data[0].arg[0]
        phi = self.data[0].param[0]
        if phi > 0:
            return self.data[0]._qasmif("t %s[%d];" % (qubit[0].name, qubit[1]))
        else:
            return self.data[0]._qasmif("tdg %s[%d];" % (qubit[0].name, qubit[1]))


def t(self, q):
    """Apply T to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.t((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(TGate(q, self))


def tdg(self, q):
    """Apply Tdg to q."""
    return self.t(q).inverse()


QuantumCircuit.t = t
QuantumCircuit.tdg = tdg
CompositeGate.t = t
CompositeGate.tdg = tdg
