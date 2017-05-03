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

    def __init__(self, qubit, circ=None):
        """Create new Z gate."""
        super(ZGate, self).__init__("z", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("z %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.z(self.arg[0]))


def z(self, quantum_register):
    """Apply Z to q."""
    if isinstance(quantum_register, QuantumRegister):
        intructions = InstructionSet()
        for register in range(quantum_register.size):
            intructions.add(self.z((quantum_register, register)))
        return intructions
    else:
        self._check_qubit(quantum_register)
        return self._attach(ZGate(quantum_register, self))


QuantumCircuit.z = z
CompositeGate.z = z
