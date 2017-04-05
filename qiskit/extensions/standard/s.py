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

    def __init__(self, qubit):
        """Create new S gate."""
        super(SGate, self).__init__("s", [], [qubit])
        self.u1(math.pi/2.0, qubit)

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.s(rearg[0]))


def s(self, j=-1):
    """Apply S to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.s((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.s((self, j)))
    return gs


QuantumRegister.s = s


def s(self, q):
    """Apply S to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(SGate(q))


QuantumCircuit.s = s


def s(self, q):
    """Apply S to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(SGate(q))


CompositeGate.s = s
