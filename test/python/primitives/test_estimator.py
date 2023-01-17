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
from ddt import data, ddt, unpack

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, Estimator, EstimatorResult
from qiskit.primitives.utils import _observable_key
from qiskit.providers import JobV1
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli, PauliList
from qiskit.test import QiskitTestCase


class TestEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.0284380963435145, -1.284366511861733

        self.psi = (RealAmplitudes(num_qubits=2, reps=2), RealAmplitudes(num_qubits=2, reps=3))
        self.params = tuple(psi.parameters for psi in self.psi)
        self.hamiltonian = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        self.theta = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )

    def test_estimator(self):
        """test for a simple use case"""
        lst = [("XX", 1), ("YY", 2), ("ZZ", 3)]
        with self.subTest("PauliSumOp"):
            observable = PauliSumOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            with self.assertWarns(DeprecationWarning):
                est = Estimator([ansatz], [observable])
                result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.84209213])

        with self.subTest("SparsePauliOp"):
            observable = SparsePauliOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            with self.assertWarns(DeprecationWarning):
                est = Estimator([ansatz], [observable])
                result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]])
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.84209213])

    def test_estimator_param_reverse(self):
        """test for the reverse parameter"""
        observable = PauliSumOp.from_list([("XX", 1), ("YY", 2), ("ZZ", 3)])
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        with self.assertWarns(DeprecationWarning):
            est = Estimator([ansatz], [observable], [ansatz.parameters[::-1]])
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5][::-1]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.84209213])

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
        with self.assertWarns(DeprecationWarning):
            est = Estimator([circuit], [matrix])
            result = est([0], [0])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_evaluate(self):
        """test for evaluate"""
        with self.assertWarns(DeprecationWarning):
            est = Estimator([self.ansatz], [self.observable])
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_evaluate_multi_params(self):
        """test for evaluate with multiple parameters"""
        with self.assertWarns(DeprecationWarning):
            est = Estimator([self.ansatz], [self.observable])
            result = est(
                [0] * 2, [0] * 2, parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]]
            )
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733, -1.3187526349078742])

    def test_evaluate_no_params(self):
        """test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        with self.assertWarns(DeprecationWarning):
            est = Estimator([circuit], [self.observable])
            result = est([0], [0])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_run_with_multiple_observables_and_none_parameters(self):
        """test for evaluate without parameters"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with self.assertWarns(DeprecationWarning):
            est = Estimator(circuit, ["ZZZ", "III"])
            result = est(circuits=[0, 0], observables=[0, 1])
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

        with self.assertWarns(DeprecationWarning):
            est = Estimator([psi1, psi2], [op1, op2, op3], [params1, params2])
        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
        theta3 = [1, 2, 3, 4, 5, 6]

        # calculate [ <psi1(theta1)|op1|psi1(theta1)> ]
        with self.assertWarns(DeprecationWarning):
            result = est([0], [0], [theta1])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.5555572817900956])
        self.assertEqual(len(result.metadata), 1)

        # calculate [ <psi1(theta1)|op2|psi1(theta1)>, <psi1(theta1)|op3|psi1(theta1)> ]
        with self.assertWarns(DeprecationWarning):
            result = est([0, 0], [1, 2], [theta1] * 2)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-0.5516530027638437, 0.07535238795415422])
        self.assertEqual(len(result.metadata), 2)

        # calculate [ <psi2(theta2)|op2|psi2(theta2)> ]
        with self.assertWarns(DeprecationWarning):
            result = est([1], [1], [theta2])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [0.17849238433885167])
        self.assertEqual(len(result.metadata), 1)

        # calculate [ <psi1(theta1)|op1|psi1(theta1)>, <psi1(theta3)|op1|psi1(theta3)> ]
        with self.assertWarns(DeprecationWarning):
            result = est([0, 0], [0, 0], [theta1, theta3])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.5555572817900956, 1.0656325933346835])
        self.assertEqual(len(result.metadata), 2)

        # calculate [ <psi1(theta1)|op1|psi1(theta1)>,
        #             <psi2(theta2)|op2|psi2(theta2)>,
        #             <psi1(theta3)|op3|psi1(theta3)> ]
        with self.assertWarns(DeprecationWarning):
            result = est([0, 1, 0], [0, 1, 2], [theta1, theta2, theta3])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(
            result.values, [1.5555572817900956, 0.17849238433885167, -1.0876631752254926]
        )
        self.assertEqual(len(result.metadata), 3)

        # It is possible to pass objects.
        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        with self.assertWarns(DeprecationWarning):
            result = est([psi2], [op2], [theta2])
        np.testing.assert_allclose(result.values, [0.17849238433885167])
        self.assertEqual(len(result.metadata), 1)

    def test_1qubit(self):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        with self.assertWarns(DeprecationWarning):
            est = Estimator([qc, qc2], [op, op2], [[]] * 2)
            result = est([0], [0], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([0], [1], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([1], [0], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
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

        with self.assertWarns(DeprecationWarning):
            est = Estimator([qc, qc2], [op, op2, op3], [[]] * 2)
            result = est([0], [0], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([1], [0], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([0], [1], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([1], [1], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([0], [2], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        with self.assertWarns(DeprecationWarning):
            result = est([1], [2], [[]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1])

    def test_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        with self.assertWarns(DeprecationWarning):
            est = Estimator([qc, qc2], [op, op2], [[]] * 2)
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            est([0], [1], [[]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            est([1], [0], [[]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            est([0], [0], [[1e4]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            est([1], [1], [[1, 2]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            est([0, 1], [1], [[1]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            est([0], [0, 1], [[1]])

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 2
        qc = QuantumCircuit(n)
        op = SparsePauliOp.from_list([("I" * n, 1)])
        with self.assertWarns(DeprecationWarning):
            estimator = Estimator(circuits=[qc] * 10, observables=[op] * 10)
        with self.subTest("one circuit"):
            with self.assertWarns(DeprecationWarning):
                result = estimator([0], [1], shots=1000)
            np.testing.assert_allclose(result.values, [1])
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            with self.assertWarns(DeprecationWarning):
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
        with self.assertWarns(DeprecationWarning):
            estimator = Estimator(circuits=qc, observables=op)
            target = estimator([0] * k, [0] * k, params_list)

        with self.subTest("ndarrary"):
            with self.assertWarns(DeprecationWarning):
                result = estimator([0] * k, [0] * k, params_array)
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values)

        with self.subTest("list of ndarray"):
            with self.assertWarns(DeprecationWarning):
                result = estimator([0] * k, [0] * k, params_list_array)
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values)

    def test_passing_objects(self):
        """Test passsing object for Estimator."""

        with self.subTest("Valid test"):
            with self.assertWarns(DeprecationWarning):
                estimator = Estimator([self.ansatz], [self.observable])
                result = estimator(
                    circuits=[self.ansatz, self.ansatz],
                    observables=[self.observable, self.observable],
                    parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
                )
        self.assertAlmostEqual(result.values[0], self.expvals[0])
        self.assertAlmostEqual(result.values[1], self.expvals[1])

        with self.subTest("Invalid circuit test"):
            circuit = QuantumCircuit(2)
            with self.assertWarns(DeprecationWarning):
                estimator = Estimator([self.ansatz], [self.observable])
            with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
                result = estimator(
                    circuits=[self.ansatz, circuit],
                    observables=[self.observable, self.observable],
                    parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
                )

        with self.subTest("Invalid observable test"):
            observable = SparsePauliOp(["ZX"])
            with self.assertWarns(DeprecationWarning):
                estimator = Estimator([self.ansatz], [self.observable])
            with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
                result = estimator(
                    circuits=[self.ansatz, self.ansatz],
                    observables=[observable, self.observable],
                    parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
                )

    def test_deprecated_arguments(self):
        """test for deprecated arguments"""
        with self.assertWarns(DeprecationWarning):
            est = Estimator([self.ansatz], [self.observable])
            result = est(
                circuit_indices=[0],
                observable_indices=[0],
                parameter_values=[[0, 1, 1, 2, 3, 5]],
            )
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_with_shots_option(self):
        """test with shots option."""
        with self.assertWarns(DeprecationWarning):
            est = Estimator([self.ansatz], [self.observable])
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=1024, seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.307397243478641])

    def test_with_shots_option_none(self):
        """test with shots=None option. Seed is ignored then."""
        with self.assertWarns(DeprecationWarning):
            est = Estimator([self.ansatz], [self.observable])
            result_42 = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=None, seed=42)
            result_15 = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=None, seed=15)
        np.testing.assert_allclose(result_42.values, result_15.values)

    def test_estimator_run(self):
        """Test Estimator.run()"""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        estimator = Estimator()

        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        job = estimator.run([psi1], [hamiltonian1], [theta1])
        self.assertIsInstance(job, JobV1)
        result = job.result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.5555572817900956])

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        result2 = estimator.run([psi2], [hamiltonian1], [theta2]).result()
        np.testing.assert_allclose(result2.values, [2.97797666])

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        result3 = estimator.run([psi1, psi1], [hamiltonian2, hamiltonian3], [theta1] * 2).result()
        np.testing.assert_allclose(result3.values, [-0.551653, 0.07535239])

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        result4 = estimator.run(
            [psi1, psi2, psi1], [hamiltonian1, hamiltonian2, hamiltonian3], [theta1, theta2, theta3]
        ).result()
        np.testing.assert_allclose(result4.values, [1.55555728, 0.17849238, -1.08766318])

    def test_estiamtor_run_no_params(self):
        """test for estimator without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        est = Estimator()
        result = est.run([circuit], [self.observable]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733])

    def test_run_single_circuit_observable(self):
        """Test for single circuit and single observable case."""
        est = Estimator()

        with self.subTest("No parameter"):
            qc = QuantumCircuit(1)
            qc.x(0)
            op = SparsePauliOp("Z")
            param_vals = [None, [], [[]], np.array([]), np.array([[]]), [np.array([])]]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run(qc, op, val).result()
                np.testing.assert_allclose(result.values, target)
                self.assertEqual(len(result.metadata), 1)

        with self.subTest("One parameter"):
            param = Parameter("x")
            qc = QuantumCircuit(1)
            qc.ry(param, 0)
            op = SparsePauliOp("Z")
            param_vals = [
                [np.pi],
                [[np.pi]],
                np.array([np.pi]),
                np.array([[np.pi]]),
                [np.array([np.pi])],
            ]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run(qc, op, val).result()
                np.testing.assert_allclose(result.values, target)
                self.assertEqual(len(result.metadata), 1)

        with self.subTest("More than one parameter"):
            qc = self.psi[0]
            op = self.hamiltonian[0]
            param_vals = [
                self.theta[0],
                [self.theta[0]],
                np.array(self.theta[0]),
                np.array([self.theta[0]]),
                [np.array(self.theta[0])],
            ]
            target = [1.5555572817900956]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run(qc, op, val).result()
                np.testing.assert_allclose(result.values, target)
                self.assertEqual(len(result.metadata), 1)

    def test_run_1qubit(self):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = Estimator()
        result = est.run([qc], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc2], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc2], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1])

    def test_run_2qubits(self):
        """Test for 2-qubit cases (to check endian)"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = Estimator()
        result = est.run([qc], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc2], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc2], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc], [op3], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1])

        result = est.run([qc2], [op3], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1])

    def test_run_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = Estimator()
        with self.assertRaises(ValueError):
            est.run([qc], [op2], [[]]).result()
        with self.assertRaises(ValueError):
            est.run([qc2], [op], [[]]).result()
        with self.assertRaises(ValueError):
            est.run([qc], [op], [[1e4]]).result()
        with self.assertRaises(ValueError):
            est.run([qc2], [op2], [[1, 2]]).result()
        with self.assertRaises(ValueError):
            est.run([qc, qc2], [op2], [[1]]).result()
        with self.assertRaises(ValueError):
            est.run([qc], [op, op2], [[1]]).result()

    def test_run_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = Estimator()
        target = estimator.run([qc] * k, [op] * k, params_list).result()

        with self.subTest("ndarrary"):
            result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values)

        with self.subTest("list of ndarray"):
            result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values)

    def test_run_with_shots_option(self):
        """test with shots option."""
        est = Estimator()
        result = est.run(
            [self.ansatz],
            [self.observable],
            parameter_values=[[0, 1, 1, 2, 3, 5]],
            shots=1024,
            seed=15,
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.307397243478641])

    def test_options(self):
        """Test for options"""
        with self.subTest("init"):
            estimator = Estimator(options={"shots": 3000})
            self.assertEqual(estimator.options.get("shots"), 3000)
        with self.subTest("set_options"):
            estimator.set_options(shots=1024, seed=15)
            self.assertEqual(estimator.options.get("shots"), 1024)
            self.assertEqual(estimator.options.get("seed"), 15)
        with self.subTest("run"):
            result = estimator.run(
                [self.ansatz],
                [self.observable],
                parameter_values=[[0, 1, 1, 2, 3, 5]],
            ).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1.307397243478641])

    def test_negative_variance(self):
        """Test for negative variance caused by numerical error."""
        qc = QuantumCircuit(1)

        estimator = Estimator()
        result = estimator.run(qc, 1e-4 * SparsePauliOp("I"), shots=1024).result()
        self.assertEqual(result.values[0], 1e-4)
        self.assertEqual(result.metadata[0]["variance"], 0.0)

    def test_different_circuits(self):
        """Test collision of quantum observables."""

        def get_op(i):
            op = SparsePauliOp.from_list([("IXIX", i)])
            return op

        keys = [_observable_key(get_op(i)) for i in range(5)]
        self.assertEqual(len(keys), len(set(keys)))


@ddt
class TestObservableValidation(QiskitTestCase):
    """Test observables validation logic."""

    @data(
        ("IXYZ", (SparsePauliOp("IXYZ"),)),
        (Pauli("IXYZ"), (SparsePauliOp("IXYZ"),)),
        (PauliList("IXYZ"), (SparsePauliOp("IXYZ"),)),
        (SparsePauliOp("IXYZ"), (SparsePauliOp("IXYZ"),)),
        (PauliSumOp(SparsePauliOp("IXYZ")), (SparsePauliOp("IXYZ"),)),
        (
            ["IXYZ", "ZYXI"],
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ),
        (
            [Pauli("IXYZ"), Pauli("ZYXI")],
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ),
        (
            [PauliList("IXYZ"), PauliList("ZYXI")],
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ),
        (
            [SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")],
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ),
        (
            [PauliSumOp(SparsePauliOp("IXYZ")), PauliSumOp(SparsePauliOp("ZYXI"))],
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ),
    )
    @unpack
    def test_validate_observables(self, obsevables, expected):
        """Test obsevables standardization."""
        self.assertEqual(BaseEstimator._validate_observables(obsevables), expected)

    @data(None, "ERROR", "")
    def test_qiskit_error(self, observables):
        """Test qiskit error if invalid input."""
        with self.assertRaises(QiskitError):
            BaseEstimator._validate_observables(observables)

    @data((), [])
    def test_value_error(self, observables):
        """Test value error if no obsevables are provided."""
        with self.assertRaises(ValueError):
            BaseEstimator._validate_observables(observables)


if __name__ == "__main__":
    unittest.main()
