# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolved operator ansatz."""


from qiskit.circuit import QuantumCircuit
from qiskit.opflow import X, Y, Z, I, MatrixEvolution

from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.test import QiskitTestCase


class TestEvolvedOperatorAnsatz(QiskitTestCase):
    """Test the evolved operator ansatz."""

    def test_evolved_op_ansatz(self):
        """Test the default evolution."""
        num_qubits = 3

        ops = [Z ^ num_qubits, Y ^ num_qubits, X ^ num_qubits]
        strings = ["z" * num_qubits, "y" * num_qubits, "x" * num_qubits] * 2

        evo = EvolvedOperatorAnsatz(ops, 2)

        reference = QuantumCircuit(num_qubits)
        parameters = evo.parameters
        for string, time in zip(strings, parameters):
            reference.compose(evolve(string, time), inplace=True)

        self.assertEqual(evo.decompose(), reference)

    def test_custom_evolution(self):
        """Test using another evolution than the default (e.g. matrix evolution)."""

        op = X ^ I ^ Z
        matrix = op.to_matrix()
        evo = EvolvedOperatorAnsatz(op, evolution=MatrixEvolution())

        parameters = evo.parameters
        reference = QuantumCircuit(3)
        reference.hamiltonian(matrix, parameters[0], [0, 1, 2])

        self.assertEqual(evo.decompose(), reference)

    def test_changing_operators(self):
        """Test rebuilding after the operators changed."""

        ops = [X, Y, Z]
        evo = EvolvedOperatorAnsatz(ops)
        evo.operators = [X, Y]

        parameters = evo.parameters
        reference = QuantumCircuit(1)
        reference.rx(2 * parameters[0], 0)
        reference.ry(2 * parameters[1], 0)

        self.assertEqual(evo.decompose(), reference)

    def test_invalid_reps(self):
        """Test setting an invalid number of reps."""
        evo = EvolvedOperatorAnsatz(X, reps=0)
        with self.assertRaises(ValueError):
            _ = evo.count_ops()

    def test_insert_barriers(self):
        """Test using insert_barriers."""
        evo = EvolvedOperatorAnsatz(Z, reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for parameter in evo.parameters:
            ref.rz(2.0 * parameter, 0)
            # ref.rx(2.0 * parameter, 0)
            if parameter != evo.parameters[-1]:
                ref.barrier()

        self.assertEqual(evo.decompose(), ref)


def evolve(pauli_string, time):
    """Get the reference evolution circuit for a single Pauli string."""

    num_qubits = len(pauli_string)
    forward = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(pauli_string):
        if pauli == "x":
            forward.h(i)
        elif pauli == "y":
            forward.sdg(i)
            forward.h(i)

    for i in range(1, num_qubits):
        forward.cx(i, 0)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(forward, inplace=True)
    circuit.rz(2 * time, 0)
    circuit.compose(forward.inverse(), inplace=True)

    return circuit
