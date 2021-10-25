"""The Pauli 2-Design circuit."""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate

def pauli_two_design(num_qubits, reps=3, seed=None):
    """Get a Pauli 2-Design circuit."""

    if seed is not None:
        np.random.seed(seed)

    paulis = [RXGate, RYGate, RZGate]

    def apply_random_pauli(circuit, param, qubit):
        gate = paulis[int(np.random.randint(3))]
        circuit.append(gate(param), [qubit])

    def rotation_layer(circuit, param_iter):
        for qubit in circuit.qubits:
            apply_random_pauli(circuit, next(param_iter), qubit)

    def entanglement_layer(circuit):
        for i in range(0, circuit.num_qubits - 1, 2):
            circuit.cz(i, i + 1)

        for i in range(1, circuit.num_qubits - 1, 2):
            circuit.cz(i, i + 1)

    circuit = QuantumCircuit(num_qubits)
    circuit.ry(np.pi / 4, circuit.qubits)

    params = ParameterVector('x', (reps + 1) * num_qubits)
    param_iter = iter(params)
    for n in range(reps + 1):
        if n > 0:
            entanglement_layer(circuit)
        rotation_layer(circuit, param_iter)

    return circuit, params[:]
