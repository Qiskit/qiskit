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

"""Test QAOA ansatz from the library."""

import numpy as np
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import HGate, RXGate, YGate, RYGate, RZGate
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.opflow import I
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.test import QiskitTestCase


@ddt
class TestQAOAAnsatz(QiskitTestCase):
    """Test QAOAAnsatz."""

    def test_default_qaoa(self):
        """Test construction of the default circuit."""
        # To be changed once QAOAAnsatz drops support for opflow
        circuit = QAOAAnsatz(I, 1)
        parameters = circuit.parameters

        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0].operation, HGate)
        self.assertIsInstance(circuit.decompose().data[1].operation, RXGate)

    def test_custom_initial_state(self):
        """Test circuit with a custom initial state."""
        initial_state = QuantumCircuit(1)
        initial_state.y(0)
        circuit = QAOAAnsatz(initial_state=initial_state, cost_operator=Pauli("I"), reps=1)
        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0].operation, YGate)
        self.assertIsInstance(circuit.decompose().data[1].operation, RXGate)

    def test_invalid_reps(self):
        """Test negative reps."""
        with self.assertRaises(ValueError):
            _ = QAOAAnsatz(Pauli("I"), reps=-1)

    def test_zero_reps(self):
        """Test zero reps."""
        circuit = QAOAAnsatz(Pauli("IIII"), reps=0)
        reference = QuantumCircuit(4)
        reference.h(range(4))

        self.assertEqual(circuit.decompose(), reference)

    def test_custom_circuit_mixer(self):
        """Test circuit with a custom mixer as a circuit"""
        mixer = QuantumCircuit(1)
        mixer.ry(1, 0)
        circuit = QAOAAnsatz(cost_operator=Pauli("I"), reps=1, mixer_operator=mixer)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(0, len(parameters))
        self.assertIsInstance(circuit.data[0].operation, HGate)
        self.assertIsInstance(circuit.data[1].operation, RYGate)

    def test_custom_operator_mixer(self):
        """Test circuit with a custom mixer as an operator."""
        mixer = Pauli("Y")

        circuit = QAOAAnsatz(cost_operator=Pauli("I"), reps=1, mixer_operator=mixer)
        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0].operation, HGate)
        self.assertIsInstance(circuit.decompose().data[1].operation, RYGate)

    def test_parameter_bounds(self):
        """Test the parameter bounds."""

        circuit = QAOAAnsatz(Pauli("Z"), reps=2)
        bounds = circuit.parameter_bounds

        for lower, upper in bounds[:2]:
            self.assertAlmostEqual(lower, 0)
            self.assertAlmostEqual(upper, 2 * np.pi)

        for lower, upper in bounds[2:]:
            self.assertIsNone(lower)
            self.assertIsNone(upper)

    def test_all_custom_parameters(self):
        """Test circuit with all custom parameters."""
        initial_state = QuantumCircuit(1)
        initial_state.y(0)
        mixer = Pauli("Z")

        circuit = QAOAAnsatz(
            cost_operator=Pauli("I"), reps=2, initial_state=initial_state, mixer_operator=mixer
        )
        parameters = circuit.parameters
        circuit = circuit.decompose()

        self.assertEqual(2, len(parameters))
        self.assertIsInstance(circuit.data[0].operation, YGate)
        self.assertIsInstance(circuit.decompose().data[1].operation, RZGate)
        self.assertIsInstance(circuit.decompose().data[2].operation, RZGate)

    def test_configuration(self):
        """Test configuration checks."""
        mixer = QuantumCircuit(2)
        circuit = QAOAAnsatz(cost_operator=Pauli("I"), reps=1, mixer_operator=mixer)

        self.assertRaises(ValueError, lambda: circuit.parameters)

    def test_rebuild(self):
        """Test how a circuit can be rebuilt."""

        circuit = QAOAAnsatz(cost_operator=Pauli("IZ"))  # circuit with 2 qubits
        # force circuit to be built
        _ = circuit.parameters

        circuit.cost_operator = Pauli("Z")  # now it only has 1 qubit
        circuit.reps = 5  # and now 5 repetitions
        # rebuild the circuit
        self.assertEqual(1, circuit.num_qubits)
        self.assertEqual(10, circuit.num_parameters)

    def test_circuit_mixer(self):
        """Test using a parameterized circuit as mixer."""
        x1, x2 = Parameter("x1"), Parameter("x2")
        mixer = QuantumCircuit(2)
        mixer.rx(x1, 0)
        mixer.ry(x2, 1)

        reps = 4
        circuit = QAOAAnsatz(cost_operator=Pauli("ZZ"), mixer_operator=mixer, reps=reps)
        self.assertEqual(circuit.num_parameters, 3 * reps)

    def test_empty_op(self):
        """Test construction without cost operator"""
        circuit = QAOAAnsatz(reps=1)
        self.assertEqual(circuit.num_qubits, 0)
        with self.assertRaises(ValueError):
            circuit.decompose()

    @data(1, 2, 3, 4)
    def test_num_qubits(self, num_qubits):
        """Test num_qubits with {num_qubits} qubits"""

        circuit = QAOAAnsatz(cost_operator=Pauli("I" * num_qubits), reps=5)
        self.assertEqual(circuit.num_qubits, num_qubits)

    def test_identity(self):
        """Test construction with identity"""
        reps = 4
        num_qubits = 3
        pauli_sum_op = SparsePauliOp.from_list([("I" * num_qubits, 1)])
        pauli_op = Pauli("I" * num_qubits)
        for cost in [pauli_op, pauli_sum_op]:
            for mixer in [None, pauli_op, pauli_sum_op]:
                with self.subTest(f"cost: {type(cost)}, mixer:{type(mixer)}"):
                    circuit = QAOAAnsatz(cost_operator=cost, mixer_operator=mixer, reps=reps)
                    target = reps if mixer is None else 0
                    self.assertEqual(circuit.num_parameters, target)
