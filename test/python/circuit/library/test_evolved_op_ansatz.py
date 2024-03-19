# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolved operator ansatz."""

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli

from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library.n_local import EvolvedOperatorAnsatz
from qiskit.synthesis.evolution import MatrixExponential
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestEvolvedOperatorAnsatz(QiskitTestCase):
    """Test the evolved operator ansatz."""

    def test_evolved_op_ansatz(self):
        """Test the default evolution."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]
        evo = EvolvedOperatorAnsatz(ops, 2)
        parameters = evo.parameters

        reference = QuantumCircuit(num_qubits)
        strings = ["z" * num_qubits, "y" * num_qubits, "x" * num_qubits] * 2
        for string, time in zip(strings, parameters):
            reference.compose(evolve(string, time), inplace=True)

        self.assertEqual(evo.decompose().decompose(), reference)

    def test_custom_evolution(self):
        """Test using another evolution than the default (e.g. matrix evolution)."""
        op = SparsePauliOp(["ZIX"])
        matrix = np.array(op)
        evolution = MatrixExponential()
        evo = EvolvedOperatorAnsatz(op, evolution=evolution)
        parameters = evo.parameters

        reference = QuantumCircuit(3)
        reference.append(HamiltonianGate(matrix, parameters[0]), [0, 1, 2])

        decomposed = evo.decompose().decompose()

        self.assertEqual(decomposed, reference)

    def test_changing_operators(self):
        """Test rebuilding after the operators changed."""
        ops = [Pauli("X"), Pauli("Y"), Pauli("Z")]
        evo = EvolvedOperatorAnsatz(ops)
        evo.operators = [Pauli("X"), Pauli("Y")]
        parameters = evo.parameters

        reference = QuantumCircuit(1)
        reference.rx(2 * parameters[0], 0)
        reference.ry(2 * parameters[1], 0)

        self.assertEqual(evo.decompose(reps=2), reference)

    def test_invalid_reps(self):
        """Test setting an invalid number of reps."""
        with self.assertRaises(ValueError):
            _ = EvolvedOperatorAnsatz(Pauli("X"), reps=-1)

    def test_insert_barriers(self):
        """Test using insert_barriers."""
        evo = EvolvedOperatorAnsatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for parameter in evo.parameters:
            ref.rz(2.0 * parameter, 0)
            ref.barrier()

        self.assertEqual(evo.decompose(reps=2), ref)

    def test_empty_build_fails(self):
        """Test setting no operators to evolve raises the appropriate error."""
        evo = EvolvedOperatorAnsatz()
        with self.assertRaises(ValueError):
            _ = evo.draw()

    def test_empty_operator_list(self):
        """Test setting an empty list of operators to be equal to an empty circuit."""
        evo = EvolvedOperatorAnsatz([])
        self.assertEqual(evo, QuantumCircuit())

    def test_matrix_operator(self):
        """Test passing a quantum_info.Operator uses the HamiltonianGate."""
        unitary = Operator([[0, 1], [1, 0]])
        evo = EvolvedOperatorAnsatz(unitary, reps=3).decompose()
        self.assertEqual(evo.count_ops()["hamiltonian"], 3)

    def test_flattened(self):
        """Test flatten option is actually flattened."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]
        evo = EvolvedOperatorAnsatz(ops, reps=3, flatten=True)
        self.assertNotIn("hamiltonian", evo.count_ops())
        self.assertNotIn("EvolvedOps", evo.count_ops())
        self.assertNotIn("PauliEvolution", evo.count_ops())


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
        forward.cx(num_qubits - i, num_qubits - i - 1)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(forward, inplace=True)
    circuit.rz(2 * time, 0)
    circuit.compose(forward.inverse(), inplace=True)

    return circuit
