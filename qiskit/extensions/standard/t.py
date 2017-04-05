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

    def __init__(self, qubit):
        """Create new T gate."""
        super(TGate, self).__init__("t", [], [qubit])
        self.u1(math.pi/4.0, qubit)

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.t(rearg[0]))


def t(self, j=-1):
    """Apply T to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.t((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.t((self, j)))
    return gs


QuantumRegister.t = t


def t(self, q):
    """Apply T to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(TGate(q))


QuantumCircuit.t = t


def t(self, q):
    """Apply T to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(TGate(q))


CompositeGate.t = t
