"""
Barrier instruction.

Author: Andrew Cross
"""
from qiskit import Instruction
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import QuantumRegister


class Barrier(Instruction):
    """Barrier instruction."""

    def __init__(self, args, circ):
        """Create new barrier instruction."""
        super(Barrier, self).__init__("barrier", [], list(args), circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def qasm(self):
        """Return OPENQASM string."""
        s = "barrier "
        for j in range(len(self.arg)):
            if len(self.arg[j]) == 1:
                s += "%s" % self.arg[j].name
            else:
                s += "%s[%d]" % (self.arg[j][0].name, self.arg[j][1])
            if j != len(self.arg) - 1:
                s += ","
        s += ";"
        return s  # no c_if on barrier instructions

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.barrier(*self.arg))


def barrier(self, *tup):
    """Apply barrier to tuples (reg, idx)."""
    qubits = []
    for t in tup:
        if isinstance(t, QuantumRegister):
            for j in range(t.sz):
                self._check_qubit((t, j))
                qubits.append((t, j))
        else:
            self._check_qubit(t)
            qubits.append(t)
    self._check_dups(qubits)
    return self._attach(Barrier(qubits, self))


QuantumCircuit.barrier = barrier
CompositeGate.barrier = barrier
