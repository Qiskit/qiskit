"""
Pauli Y (bit-phase-flip) gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class YGate(Gate):
    """Pauli Y (bit-phase-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new Y gate."""
        super(YGate, self).__init__("y", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("y %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circuit):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circuit.y(self.arg[0]))


def y(self, quantum_register):
    """Apply Y to q."""
    if isinstance(quantum_register, QuantumRegister):
        intructions = InstructionSet()
        for register in range(quantum_register.size):
            intructions.add(self.y((quantum_register, register)))
        return intructions
    else:
        self._check_qubit(quantum_register)
        return self._attach(YGate(quantum_register, self))


QuantumCircuit.y = y
CompositeGate.y = y
