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

"""Test QAOA ansatz from the library."""

from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz, QAOAGate
from qiskit.circuit.library import HGate, RXGate, YGate, RYGate, RZGate
from qiskit.opflow import I, Y, Z
from qiskit.test import QiskitTestCase


class TestQAOAAnsatz(QiskitTestCase):
    """Test QAOAAnsatz."""

    def test_default_qaoa(self):
        """Test construction of the default circuit."""
        circuit = QAOAAnsatz(I, 1)

        parameters = circuit.parameters

        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        self.assertIsInstance(circuit.data[1][0], RXGate)

    def test_custom_initial_state(self):
        """Test circuit with a custom initial state."""
        initial_state = QuantumCircuit(1)
        initial_state.y(0)
        circuit = QAOAAnsatz(initial_state=initial_state, cost_operator=I, reps=1)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], YGate)
        self.assertIsInstance(circuit.data[1][0], RXGate)

    def test_invalid_reps(self):
        """Test negative reps."""
        circuit = QAOAAnsatz(I, reps=-1)
        with self.assertRaises(AttributeError):
            _ = circuit.count_ops()

    def test_zero_reps(self):
        """Test zero reps."""
        circuit = QAOAAnsatz(I ^ 4, reps=0)
        reference = QuantumCircuit(4)
        reference.h(range(4))
        self.assertEqual(circuit.decompose(), reference)

    def test_custom_circuit_mixer(self):
        """Test circuit with a custom mixer as a circuit"""
        mixer = QuantumCircuit(1)
        mixer.ry(1, 0)
        circuit = QAOAAnsatz(cost_operator=I, reps=1, mixer_operator=mixer)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(0, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        self.assertIsInstance(circuit.data[1][0], RYGate)

    def test_custom_operator_mixer(self):
        """Test circuit with a custom mixer as an operator."""
        mixer = Y
        circuit = QAOAAnsatz(cost_operator=I, reps=1, mixer_operator=mixer)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        self.assertIsInstance(circuit.data[1][0], RYGate)

    def test_all_custom_parameters(self):
        """Test circuit with all custom parameters."""
        initial_state = QuantumCircuit(1)
        initial_state.y(0)
        mixer = Z

        circuit = QAOAAnsatz(
            cost_operator=I, reps=2, initial_state=initial_state, mixer_operator=mixer
        )

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(2, len(parameters))
        self.assertIsInstance(circuit.data[0][0], YGate)
        self.assertIsInstance(circuit.data[1][0], RZGate)
        self.assertIsInstance(circuit.data[2][0], RZGate)

    def test_configuration(self):
        """Test configuration checks."""
        mixer = QuantumCircuit(2)
        circuit = QAOAAnsatz(cost_operator=I, reps=1, mixer_operator=mixer)

        self.assertRaises(AttributeError, lambda: circuit.parameters)

    def test_rebuild(self):
        """Test how a circuit can be rebuilt."""
        circuit = QAOAAnsatz(cost_operator=Z ^ I)  # circuit with 2 qubits
        # force circuit to be built
        _ = circuit.parameters

        circuit.cost_operator = Z  # now it only has 1 qubit
        circuit.reps = 5  # and now 5 repetitions
        # rebuild the circuit
        _ = circuit.parameters
        self.assertEqual(1, circuit.num_qubits)
        self.assertEqual(10, circuit.num_parameters)

    def test_qaoa_gate_inserted(self):
        """Test the ``QAOAGate`` is used an we can retrieve the operator information."""
        qaoa = QAOAAnsatz(Z ^ Z ^ I, reps=2)

        with self.subTest(msg="QAOAGate as only instruction"):
            self.assertIsInstance(qaoa.data[0][0], QAOAGate)

        with self.subTest(msg="extract operator info from gate"):
            gate = qaoa.data[0][0]
            self.assertEqual(gate.cost_operator, Z ^ Z ^ I)
            self.assertIsNotNone(gate.mixer_operator)

    def test_parameters(self):
        """Test that the parameter instances don't change between construction and definition."""
        qaoa = QAOAAnsatz(Z ^ Z ^ I, reps=2)
        parameters = qaoa.parameters

        with self.subTest(msg="test number of parameters"):
            self.assertEqual(len(parameters), 4)

        circuit = transpile(qaoa, basis_gates=['u', 'cx'])
        with self.subTest(msg="test number of parameters of transpiled circuit"):
            self.assertEqual(circuit.num_parameters, 4)

        with self.subTest(msg="test binding parameters per instance"):
            bound = circuit.bind_parameters(dict(zip(parameters, list(range(4)))))
            self.assertEqual(bound.num_parameters, 0)
