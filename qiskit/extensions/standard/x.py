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

    def __init__(self, qubit):
        """Create new X gate."""
        super(XGate, self).__init__("x", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("x %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.x(rearg[0]))


def x(self, j=-1):
    """Apply X to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.x((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.x((self, j)))
    return gs


QuantumRegister.x = x


def x(self, q):
    """Apply X to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(XGate(q))


QuantumCircuit.x = x


def x(self, q):
    """Apply X to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(XGate(q))


CompositeGate.x = x
