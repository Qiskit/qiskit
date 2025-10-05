# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An instruction to implement a Pauli Product Measurement."""

from qiskit.circuit import QuantumCircuit, CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info.operators.symplectic.pauli import Pauli


class PauliProductMeasurement(Instruction):
    """Pauli Product Measurement instruction"""

    def __init__(self, pauli: Pauli):
        super().__init__(
            name="PauliProductMeasurement", num_qubits=pauli.num_qubits, num_clbits=1, params=[]
        )
        self._pauli = pauli
        if pauli.phase not in [0, 2]:
            raise CircuitError("Pauli phase of i or -i is not acceptable.")

    def inverse(self, annotated=False):
        raise CircuitError("PauliProductMeasurement is not invertible.")

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, self.num_clbits)
        # Construct a quantum circuit:
        # Clifford gates + single Z-measurement + Clifford gates
        # similar to the code we have in the PauliEvolutionGate,
        # the only difference is that we apply a Z-measure
        # instead of a Z rotation in the middle
        # should be rewritten in rust reusing the code in:
        # https://github.com/Qiskit/qiskit/blob/main/crates/circuit_library/src/pauli_evolution.rs
        pauli = self._pauli
        num_qubits = self._pauli.num_qubits
        # Basis change layer
        pauli_qubits = []
        for i in range(num_qubits):
            if pauli[i] == Pauli("X"):
                qc.h(i)
                pauli_qubits.append(i)
            if pauli[i] == Pauli("Y"):
                qc.sx(i)
                pauli_qubits.append(i)
            if pauli[i] == Pauli("Z"):
                pauli_qubits.append(i)
        if pauli.phase == 2:
            qc.x(0)
        # CX layer
        rev_pauli_qubits = list(reversed(pauli_qubits))
        for i in range(len(rev_pauli_qubits) - 1):
            qc.cx(rev_pauli_qubits[i], rev_pauli_qubits[i + 1])
        # Z-measurement on qubit 0
        qc.measure(0, 0)
        # CX layer
        for i in range(len(pauli_qubits) - 1):
            qc.cx(pauli_qubits[i + 1], pauli_qubits[i])
        # Basis change layer
        if pauli.phase == 2:
            qc.x(0)
        for i in range(num_qubits):
            if pauli[i] == Pauli("X"):
                qc.h(i)
            if pauli[i] == Pauli("Y"):
                qc.sxdg(i)
        self.definition = qc
