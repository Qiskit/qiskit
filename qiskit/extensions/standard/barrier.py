"""
Barrier instruction.

Author: Andrew Cross
"""
from qiskit import Instruction
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import QuantumRegister
from qiskit import QISKitException


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
        string = "barrier "
        for j in range(len(self.arg)):
            if len(self.arg[j]) == 1:
                string += "%s" % self.arg[j].name
            else:
                string += "%s[%d]" % (self.arg[j][0].name, self.arg[j][1])
            if j != len(self.arg) - 1:
                string += ","
        string += ";"
        return string  # no c_if on barrier instructions

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.barrier(*self.arg))


def barrier(self, *tuples):
    """Apply barrier to tuples (reg, idx)."""
    tuples = list(tuples)
    if len(tuples) == 0:  # TODO: implement this for all single qubit gates
        if isinstance(self, QuantumCircuit):
            for register in self.regs.values():
                if isinstance(register, QuantumRegister):
                    tuples.append(register)
    if len(tuples) == 0:
        raise QISKitException("no arguments passed")
    qubits = []
    for tuple_element in tuples:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                self._check_qubit((tuple_element, j))
                qubits.append((tuple_element, j))
        else:
            self._check_qubit(tuple_element)
            qubits.append(tuple_element)
    self._check_dups(qubits)
    return self._attach(Barrier(qubits, self))


QuantumCircuit.barrier = barrier
CompositeGate.barrier = barrier
