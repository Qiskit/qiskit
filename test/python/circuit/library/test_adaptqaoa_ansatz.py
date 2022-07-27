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
import unittest
import numpy as np
from ddt import ddt, data, idata, unpack

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import HGate, RXGate, YGate, RYGate, RZGate
from qiskit.circuit.library.n_local.adaptqaoa_ansatz import AdaptQAOAAnsatz
from qiskit.opflow import I, Y, Z, PauliSumOp
from qiskit.test import QiskitTestCase


@ddt
class TestAdaptQAOAAnsatz(QiskitTestCase):
    """Test QAOAAnsatz."""

    def test_default_qaoa(self):

        """Test construction of the default circuit."""
        circuit = AdaptQAOAAnsatz(I)

        parameters = circuit.parameters

        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        self.assertIsInstance(circuit.decompose().data[1][0], RXGate)

    def test_custom_initial_state(self):
        """Test circuit with a custom initial state."""
        initial_state = QuantumCircuit(1)
        initial_state.y(0)
        circuit = AdaptQAOAAnsatz(initial_state=initial_state, cost_operator=I)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], YGate)
        self.assertIsInstance(circuit.decompose().data[1][0], RXGate)

    def test_zero_depth(self):
        """Test zero reps."""
        circuit = AdaptQAOAAnsatz(I ^ 4, mixer_operators=[])
        reference = QuantumCircuit(4)
        reference.h(range(4))

        self.assertEqual(circuit.decompose(), reference)

    @idata(
        [
            [RYGate(0)],
            [[RYGate(0)]],
            [[RYGate(0), RZGate(0)]],
        ]
    )
    @unpack
    def test_custom_circuit_mixer(self, operators):
        """Test circuit with a custom mixer as a circuit or list of circuits"""
        mixers = []
        if not isinstance(operators, list):  # Input is a circuit
            mixers = QuantumCircuit(1)
            mixers.append(operators, [0])
            operators = [operators]
        else:  # Input is a list of circuits
            for op in operators:
                mixer = QuantumCircuit(1)
                mixer.append(op, [0])
                mixers.append(mixer)
        circuit = AdaptQAOAAnsatz(cost_operator=I, mixer_operators=mixers)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(0, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        for rep, op in enumerate(operators, 1):
            self.assertEqual(circuit.data[rep][0], op)

    @idata([[Y, RYGate], [[Y], [RYGate]], [[Y, Z], [RYGate, RZGate]]])
    @unpack
    def test_custom_operator_mixer(self, operators, target_class):
        """Test circuit with a custom mixer as an operator or list of operators."""
        circuit = AdaptQAOAAnsatz(cost_operator=I, mixer_operators=operators)
        operators = operators if isinstance(operators, list) else [operators]
        target_class = target_class if isinstance(target_class, list) else [target_class]

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(len(operators), len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        for rep, target in enumerate(target_class, 1):
            self.assertIsInstance(circuit.data[rep][0], target)

    def test_custom_operator_circuit_mixer(self):
        """Test circuit with a custom list of mixers containing operators and circuits."""
        operators = [Y]
        circuit_op = QuantumCircuit(1)
        circuit_op.rz(1, 0)
        operators.append(circuit_op)
        target_class = [RYGate, RZGate]
        circuit = AdaptQAOAAnsatz(cost_operator=I, mixer_operators=operators)

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        for rep, target in enumerate(target_class, 1):
            self.assertIsInstance(circuit.data[rep][0], target)

    def test_parameter_bounds(self):
        """Test the parameter bounds."""
        circuit = AdaptQAOAAnsatz(Z)
        bounds = circuit.parameter_bounds

        for lower, upper in bounds[:2]:
            self.assertEqual(lower, -2 * np.pi)
            self.assertAlmostEqual(upper, 2 * np.pi)

        for lower, upper in bounds[2:]:
            self.assertEqual(lower, -0.5 * np.pi)
            self.assertIsNone(upper, 0.5 * np.pi)

    def test_all_custom_parameters(self):
        """Test circuit with all custom parameters."""
        initial_state = QuantumCircuit(1)
        initial_state.y(0)
        mixer = Z

        circuit = AdaptQAOAAnsatz(
            cost_operator=I, initial_state=initial_state, mixer_operators=mixer
        )

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], YGate)
        self.assertIsInstance(circuit.data[1][0], RZGate)

    def test_configuration(self):
        """Test configuration checks."""
        mixer = QuantumCircuit(2)
        circuit = AdaptQAOAAnsatz(cost_operator=I, mixer_operators=mixer)
        self.assertRaises(ValueError, lambda: circuit.parameters)

    def test_rebuild(self):
        """Test how a circuit can be rebuilt."""
        circuit = AdaptQAOAAnsatz(cost_operator=Z ^ I)  # circuit with 2 qubits
        # force circuit to be built
        _ = circuit.parameters

        circuit.cost_operator = Z  # now it only has 1 qubit
        mixer_operators = 5 * [circuit.mixer_operator]
        circuit.mixer_operators = mixer_operators
        circuit.mixer_pool = None
        # rebuild the circuit
        self.assertEqual(1, circuit.num_qubits)
        self.assertEqual(10, circuit.num_parameters)

    def test_circuit_mixer(self):
        """Test using a parameterized circuit as mixer."""
        reps = 4
        x1, x2 = Parameter("x1"), Parameter("x2")
        mixer = QuantumCircuit(2)
        mixer.rx(x1, 0)
        mixer.ry(x2, 1)
        n_beta = len(mixer.parameters)
        mixer = reps * [mixer]

        circuit = AdaptQAOAAnsatz(cost_operator=Z ^ Z, mixer_operators=mixer)
        self.assertEqual(circuit.num_parameters, reps * (n_beta + 1))

    def test_empty_op(self):
        """Test construction without cost operator"""
        circuit = AdaptQAOAAnsatz()
        self.assertEqual(circuit.num_qubits, 0)
        with self.assertRaises(ValueError):
            circuit.decompose()

    @data(1, 2, 3, 4)
    def test_num_qubits(self, num_qubits):
        """Test num_qubits with {num_qubits} qubits"""
        circuit = AdaptQAOAAnsatz(cost_operator=I ^ num_qubits)
        self.assertEqual(circuit.num_qubits, num_qubits)

    def test_identity(self):
        """Test construction with identity"""
        num_qubits = 3
        pauli_sum_op = PauliSumOp.from_list([("I" * num_qubits, 1)])
        pauli_op = I ^ num_qubits
        for cost in [pauli_op, pauli_sum_op]:
            for mixer in [None, pauli_op, pauli_sum_op]:
                with self.subTest(f"cost: {type(cost)}, mixer:{type(mixer)}"):
                    circuit = AdaptQAOAAnsatz(cost_operator=cost, mixer_operators=mixer)
                    target = len(circuit.mixer_operators) if mixer is None else 0
                    self.assertEqual(circuit.num_parameters, target)


if __name__ == "__main__":
    unittest.main()
