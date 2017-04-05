"""
Diagonal single qubit gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class U1Gate(Gate):
    """Diagonal single qubit gate."""

    def __init__(self, theta, qubit):
        """Create new diagonal single qubit gate."""
        super(U1Gate, self).__init__("u1", [theta], [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return self._qasmif("u1(%.15f) %s[%d];" % (theta, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.u1(self.param[0], rearg[0]))


def u1(self, theta, j=-1):
    """Apply u1 with angle theta to jth qubit in this register (or all)."""
    self._check_bound()
    gs = InstructionSet()
    if j == -1:
        for p in self.bound_to:
            for k in range(self.sz):
                gs.add(p.u1(theta, (self, k)))
    else:
        self.check_range(j)
        for p in self.bound_to:
            gs.add(p.u1(theta, (self, j)))
    return gs


QuantumRegister.u1 = u1


def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    self._check_qreg(q[0])
    q[0].check_range(q[1])
    return self._attach(U1Gate(theta, q))


QuantumCircuit.u1 = u1


def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    self._check_qubit(q)
    q[0].check_range(q[1])
    return self._attach(U1Gate(theta, q))


CompositeGate.u1 = u1
