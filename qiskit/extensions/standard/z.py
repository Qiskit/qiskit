"""
Pauli Z (phase-flip) gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class ZGate(Gate):
    """Pauli Z (phase-flip) gate."""

    def __init__(self, qubit):
        """Create new Z gate."""
        super(ZGate, self).__init__("z", [], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("z %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.z(rearg[0]))


def z(self, j=-1):
    """Apply Z to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.z((self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.z((self, j)))
    return gs


QuantumRegister.z = z


def z(self, q):
    """Apply Z to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(ZGate(q))


QuantumCircuit.z = z


def z(self, q):
    """Apply Z to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(ZGate(q))


CompositeGate.z= z
