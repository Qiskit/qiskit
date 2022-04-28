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
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import HGate, RXGate, YGate, RYGate, RZGate
from qiskit.circuit.library.n_local.adaptqaoa_ansatz import AdaptQAOAAnsatz
from qiskit.opflow import I, Y, Z, PauliSumOp
from qiskit.test import QiskitTestCase

"""Comments:
    test_default_qaoa:
        - last assertation of QAOAAnsatz checks for RXGate... Not sure what the parallel for Adapt would be.
    test_custom_initial_state:
        - 
        

    Tests to add:
        
        """
    
class unittesthelper:
    def __init__(self) -> None:
        pass

    def assertEqual(self, a, b):
        assert a == b

    def assertIsInstance(self, a, b):
        assert isinstance(a, b)

    def assertAlmostEqual(self, a, b):
        assert round(a-b, 7) == 0

    def assertIsNone(self, a):
        assert a == None

    def assertIn(self, a, b):
        assert a in b      
# @ddt
# class TestQAOAAnsatz(QiskitTestCase):
class TestQAOAAnsatz(unittesthelper):
    """Test QAOAAnsatz."""

    def test_default_qaoa(self):
        """Test construction of the default circuit."""
        circuit = AdaptQAOAAnsatz(I)

        parameters = circuit.parameters

        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))    # check number of params is == 1
        self.assertIsInstance(circuit.data[0][0], HGate)    # check first gate are hadamards
        self.assertIsInstance(circuit.decompose().data[1][0], RXGate)   # check next is some other gate

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

    def test_zero_reps(self):
        """Test zero reps."""
        circuit = AdaptQAOAAnsatz(I ^ 4)
        circuit.set_mixer_operators(mixer_operators = [])
        reference = QuantumCircuit(4)
        reference.h(range(4))

        self.assertEqual(circuit.decompose(), reference)

    def test_custom_circuit_mixer(self):
        """Test circuit with a custom mixer as a circuit"""
        mixer = QuantumCircuit(1)
        mixer.ry(1, 0)
        circuit = AdaptQAOAAnsatz(cost_operator=I)
        circuit.mixer_operators = mixer

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(0, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        self.assertIsInstance(circuit.data[1][0], RYGate)

    def test_custom_operator_mixer(self):
        """Test circuit with a custom mixer as an operator."""
        mixer = Y
        circuit = AdaptQAOAAnsatz(cost_operator=I)
        circuit.mixer_operators = mixer

        parameters = circuit.parameters
        circuit = circuit.decompose()
        self.assertEqual(1, len(parameters))
        self.assertIsInstance(circuit.data[0][0], HGate)
        self.assertIsInstance(circuit.data[1][0], RYGate)

    def test_parameter_bounds(self):
        """Test the parameter bounds."""
        circuit = AdaptQAOAAnsatz(Z)
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
        mixer = Z

        circuit = AdaptQAOAAnsatz(
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
        circuit = AdaptQAOAAnsatz(cost_operator=I, reps=1, mixer_operator=mixer)

        self.assertRaises(ValueError, lambda: circuit.parameters)

    def test_rebuild(self):
        """Test how a circuit can be rebuilt."""
        circuit = AdaptQAOAAnsatz(cost_operator=Z ^ I)  # circuit with 2 qubits
        # force circuit to be built
        _ = circuit.parameters

        circuit.cost_operator = Z  # now it only has 1 qubit
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
        circuit = AdaptQAOAAnsatz(cost_operator=Z ^ Z, mixer_operator=mixer, reps=reps)
        self.assertEqual(circuit.num_parameters, 3 * reps)

    def test_empty_op(self):
        """Test construction without cost operator"""
        circuit = AdaptQAOAAnsatz(reps=1)
        self.assertEqual(circuit.num_qubits, 0)
        with self.assertRaises(ValueError):
            circuit.decompose()

    @data(1, 2, 3, 4)
    def test_num_qubits(self, num_qubits):
        """Test num_qubits with {num_qubits} qubits"""

        circuit = AdaptQAOAAnsatz(cost_operator=I ^ num_qubits, reps=5)
        self.assertEqual(circuit.num_qubits, num_qubits)

    def test_identity(self):
        """Test construction with identity"""
        reps = 4
        num_qubits = 3
        pauli_sum_op = PauliSumOp.from_list([("I" * num_qubits, 1)])
        pauli_op = I ^ num_qubits
        for cost in [pauli_op, pauli_sum_op]:
            for mixer in [None, pauli_op, pauli_sum_op]:
                with self.subTest(f"cost: {type(cost)}, mixer:{type(mixer)}"):
                    circuit = AdaptQAOAAnsatz(cost_operator=cost, mixer_operator=mixer, reps=reps)
                    target = reps if mixer is None else 0
                    self.assertEqual(circuit.num_parameters, target)

if __name__ == "__main__":
    # unittest.main()
    tqaa = TestQAOAAnsatz()
    tqaa.test_default_qaoa()
    tqaa.test_custom_initial_state()
    tqaa.test_custom_circuit_mixer()
    tqaa.test_custom_operator_mixer()
    tqaa.test_parameter_bounds()
    # tqaa.test_all_custom_parameters()
    # tqaa.test_default_qaoa()
    # tqaa.test_default_qaoa()
