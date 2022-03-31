# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Estimator."""

import unittest

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator, EstimatorResult
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.test import QiskitTestCase


class TestEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_estimator(self):
        """test for a simple use case"""
        lst = [("XX", 1), ("YY", 2), ("ZZ", 3)]
        with self.subTest("PauliSumOp"):
            observable = PauliSumOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            with Estimator([ansatz], [observable]) as est:
                result = est(parameter_values=[0, 1, 1, 2, 3, 5])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.84209213])

        with self.subTest("SparsePauliOp"):
            observable = SparsePauliOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            with Estimator([ansatz], [observable]) as est:
                result = est(parameter_values=[0, 1, 1, 2, 3, 5])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.84209213])

    def test_estimator_param_reverse(self):
        """test for the reverse parameter"""
        observable = PauliSumOp.from_list([("XX", 1), ("YY", 2), ("ZZ", 3)])
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        with Estimator([ansatz], [observable], [ansatz.parameters[::-1]]) as est:
            result = est(parameter_values=[0, 1, 1, 2, 3, 5][::-1])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.84209213])

    def test_init_from_statevector(self):
        """test initialization from statevector"""
        vector = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        statevector = Statevector(vector)
        with Estimator([statevector], [self.observable]) as est:
            self.assertIsInstance(est.circuits[0], QuantumCircuit)
            np.testing.assert_allclose(est.circuits[0][0][0].params, vector)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-0.88272215])

    def test_init_observable_from_operator(self):
        """test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        matrix = Operator(
            [
                [-1.06365335, 0.0, 0.0, 0.1809312],
                [0.0, -1.83696799, 0.1809312, 0.0],
                [0.0, 0.1809312, -0.24521829, 0.0],
                [0.1809312, 0.0, 0.0, -1.06365335],
            ]
        )
        with Estimator([circuit], [matrix]) as est:
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_evaluate(self):
        """test for evaluate"""
        with Estimator([self.ansatz], [self.observable]) as est:
            result = est(parameter_values=[0, 1, 1, 2, 3, 5])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_evaluate_multi_params(self):
        """test for evaluate with multiple parameters"""
        with Estimator([self.ansatz], [self.observable]) as est:
            result = est(
                [0] * 2, [0] * 2, parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]]
            )
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733, -1.3187526349078742])

    def test_evaluate_no_params(self):
        """test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        with Estimator([circuit], [self.observable]) as est:
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_run_with_multiple_observables_and_none_parameters(self):
        """test for evaluate without parameters"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with Estimator(circuit, ["ZZZ", "III"]) as est:
            result = est(circuit_indices=[0, 0], observable_indices=[0, 1])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [0.0, 1.0])

    def test_estimator_example(self):
        """test for Estimator example"""
        psi1 = RealAmplitudes(num_qubits=2, reps=2)
        psi2 = RealAmplitudes(num_qubits=2, reps=3)

        params1 = psi1.parameters
        params2 = psi2.parameters

        op1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        op2 = SparsePauliOp.from_list([("IZ", 1)])
        op3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

        with Estimator([psi1, psi2], [op1, op2, op3], [params1, params2]) as est:
            theta1 = [0, 1, 1, 2, 3, 5]
            theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
            theta3 = [1, 2, 3, 4, 5, 6]

            # calculate [ <psi1(theta1)|op1|psi1(theta1)> ]
            result = est([0], [0], [theta1])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.5555572817900956])
            self.assertEqual(len(result.metadata), 1)

            # calculate [ <psi1(theta1)|op2|psi1(theta1)>, <psi1(theta1)|op3|psi1(theta1)> ]
            result = est([0, 0], [1, 2], [theta1] * 2)
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-0.5516530027638437, 0.07535238795415422])
            self.assertEqual(len(result.metadata), 2)

            # calculate [ <psi2(theta2)|op2|psi2(theta2)> ]
            result = est([1], [1], [theta2])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [0.17849238433885167])
            self.assertEqual(len(result.metadata), 1)

            # calculate [ <psi1(theta1)|op1|psi1(theta1)>, <psi1(theta3)|op1|psi1(theta3)> ]
            result = est([0, 0], [0, 0], [theta1, theta3])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.5555572817900956, 1.0656325933346835])
            self.assertEqual(len(result.metadata), 2)

            # calculate [ <psi1(theta1)|op1|psi1(theta1)>,
            #             <psi2(theta2)|op2|psi2(theta2)>,
            #             <psi1(theta3)|op3|psi1(theta3)> ]
            result = est([0, 1, 0], [0, 1, 2], [theta1, theta2, theta3])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(
                result.values, [1.5555572817900956, 0.17849238433885167, -1.0876631752254926]
            )
            self.assertEqual(len(result.metadata), 3)

    def test_1qubit(self):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        with Estimator([qc, qc2], [op, op2], [[]] * 2) as est:
            result = est([0], [0], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([0], [1], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([1], [0], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([1], [1], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1])

    def test_2qubits(self):
        """Test for 2-qubit cases (to check endian)"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        with Estimator([qc, qc2], [op, op2, op3], [[]] * 2) as est:
            result = est([0], [0], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([1], [0], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([0], [1], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([1], [1], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([0], [2], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

            result = est([1], [2], [[]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1])

    def test_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        with Estimator([qc, qc2], [op, op2], [[]] * 2) as est:
            with self.assertRaises(QiskitError):
                est([0], [1], [[]])
            with self.assertRaises(QiskitError):
                est([1], [0], [[]])
            with self.assertRaises(QiskitError):
                est([0], [0], [[1e4]])
            with self.assertRaises(QiskitError):
                est([1], [1], [[1, 2]])
            with self.assertRaises(QiskitError):
                est([0, 1], [1], [[1]])
            with self.assertRaises(QiskitError):
                est([0], [0, 1], [[1]])

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 2
        qc = QuantumCircuit(n)
        op = SparsePauliOp.from_list([("I" * n, 1)])
        with Estimator(circuits=[qc] * 10, observables=[op] * 10) as estimator:
            with self.subTest("one circuit"):
                result = estimator([0], [1], shots=1000)
                np.testing.assert_allclose(result.values, [1])
                self.assertEqual(len(result.metadata), 1)

            with self.subTest("two circuits"):
                result = estimator([2, 4], [3, 5], shots=1000)
                np.testing.assert_allclose(result.values, [1, 1])
                self.assertEqual(len(result.metadata), 2)

    def test_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        with Estimator(circuits=qc, observables=op) as estimator:
            target = estimator([0] * k, [0] * k, params_list)

            with self.subTest("ndarrary"):
                result = estimator([0] * k, [0] * k, params_array)
                self.assertEqual(len(result.metadata), k)
                np.testing.assert_allclose(result.values, target.values)

            with self.subTest("list of ndarray"):
                result = estimator([0] * k, [0] * k, params_list_array)
                self.assertEqual(len(result.metadata), k)
                np.testing.assert_allclose(result.values, target.values)


if __name__ == "__main__":
    unittest.main()
