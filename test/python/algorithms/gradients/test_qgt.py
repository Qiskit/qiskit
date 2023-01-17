# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

""" Test QGT"""

import unittest

import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms.gradients import DerivativeType, LinCombQGT
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.test import QiskitTestCase

from .logging_primitives import LoggingEstimator


class TestQGT(QiskitTestCase):
    """Test QGT"""

    def setUp(self):
        super().setUp()
        self.estimator = Estimator()

    def test_qgt_derivative_type(self):
        """Test QGT derivative_type"""
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        param_list = [[np.pi / 4, 0], [np.pi / 2, np.pi / 4]]
        # test real derivative
        with self.subTest("Test with DerivativeType.REAL"):
            qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.REAL)
            correct_values = np.array([[[1, 0], [0, 0.5]], [[1, 0], [0, 1]]]) / 4
            for i, param in enumerate(param_list):
                qgt_result = qgt.run([qc], [param]).result().qgts
                np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

        # test imaginary derivative
        with self.subTest("Test with DerivativeType.IMAG"):
            qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.IMAG)

            correct_values = (
                np.array([[[0, 0.707106781], [-0.707106781, 0]], [[0, 1], [-1, 0]]]) / 4
            )
            for i, param in enumerate(param_list):
                qgt_result = qgt.run([qc], [param]).result().qgts
                np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

        # test real + imaginary derivative
        with self.subTest("Test with DerivativeType.COMPLEX"):
            qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.COMPLEX)
            correct_values = (
                np.array([[[1, 0.707106781j], [-0.707106781j, 0.5]], [[1, 1j], [-1j, 1]]]) / 4
            )
            for i, param in enumerate(param_list):
                qgt_result = qgt.run([qc], [param]).result().qgts
                np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

    def test_qgt_phase_fix(self):
        """Test the phase-fix argument in a QGT calculation"""
        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        param_list = [[np.pi / 4, 0], [np.pi / 2, np.pi / 4]]
        # test real derivative
        with self.subTest("Test phase fix with DerivativeType.REAL"):
            qgt = LinCombQGT(self.estimator, phase_fix=False, derivative_type=DerivativeType.REAL)
            correct_values = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]) / 4
            for i, param in enumerate(param_list):
                qgt_result = qgt.run([qc], [param]).result().qgts
                np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

        # test imaginary derivative
        with self.subTest("Test phase fix with DerivativeType.IMAG"):
            qgt = LinCombQGT(self.estimator, phase_fix=False, derivative_type=DerivativeType.IMAG)
            correct_values = (
                np.array([[[0, 0.707106781], [-0.707106781, 0]], [[0, 1], [-1, 0]]]) / 4
            )
            for i, param in enumerate(param_list):
                qgt_result = qgt.run([qc], [param]).result().qgts
                np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

        # test real + imaginary derivative
        with self.subTest("Test phase fix with DerivativeType.COMPLEX"):
            qgt = LinCombQGT(
                self.estimator, phase_fix=False, derivative_type=DerivativeType.COMPLEX
            )
            correct_values = (
                np.array([[[1, 0.707106781j], [-0.707106781j, 1]], [[1, 1j], [-1j, 1]]]) / 4
            )
            for i, param in enumerate(param_list):
                qgt_result = qgt.run([qc], [param]).result().qgts
                np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

    def test_qgt_coefficients(self):
        """Test the derivative option of QGT"""
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[2] + qc.parameters[3].sin(), 1)

        qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.REAL)
        # test imaginary derivative
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_values = (
            np.array(
                [
                    [
                        [5.707309, 4.2924833, 1.5295868, 0.1938604],
                        [4.2924833, 4.9142136, 0.75, 0.8838835],
                        [1.5295868, 0.75, 3.4430195, 0.0758252],
                        [0.1938604, 0.8838835, 0.0758252, 1.1357233],
                    ],
                    [
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 10.0, -0.0],
                        [0.0, 0.0, -0.0, 1.0],
                    ],
                ]
            )
            / 4
        )
        for i, param in enumerate(param_list):
            qgt_result = qgt.run([qc], [param]).result().qgts
            np.testing.assert_allclose(qgt_result[0], correct_values[i], atol=1e-3)

    def test_qgt_specify_parameters(self):
        """Test the QGT with specified parameters"""
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.ry(b, 0)
        qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.REAL)
        param_list = [np.pi / 4, np.pi / 4]
        qgt_result = qgt.run([qc], [param_list], [[a]]).result().qgts
        np.testing.assert_allclose(qgt_result[0], [[1 / 4]], atol=1e-3)

    def test_qgt_multi_arguments(self):
        """Test the QGT for multiple arguments"""
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.ry(b, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(a, 0)
        qc2.ry(b, 0)
        qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.REAL)

        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_values = [[[1 / 4]], [[1 / 4, 0], [0, 0]]]
        param_list = [[np.pi / 4, np.pi / 4], [np.pi / 2, np.pi / 2]]
        qgt_results = qgt.run([qc, qc2], param_list, [[a], None]).result().qgts
        for i, _ in enumerate(param_list):
            np.testing.assert_allclose(qgt_results[i], correct_values[i], atol=1e-3)

    def test_qgt_validation(self):
        """Test estimator QGT's validation"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qgt = LinCombQGT(self.estimator)
        parameter_values = [[np.pi / 4]]
        with self.subTest("assert number of circuits does not match"):
            with self.assertRaises(ValueError):
                qgt.run([qc, qc], parameter_values)
        with self.subTest("assert number of parameter values does not match"):
            with self.assertRaises(ValueError):
                qgt.run([qc], [[np.pi / 4], [np.pi / 2]])
        with self.subTest("assert number of parameters does not match"):
            with self.assertRaises(ValueError):
                qgt.run([qc], parameter_values, parameters=[[a], [a]])

    def test_options(self):
        """Test QGT's options"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        estimator = Estimator(options={"shots": 100})

        with self.subTest("estimator"):
            qgt = LinCombQGT(estimator)
            options = qgt.options
            result = qgt.run([qc], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 100)
            self.assertEqual(options.get("shots"), 100)

        with self.subTest("QGT init"):
            qgt = LinCombQGT(estimator, options={"shots": 200})
            result = qgt.run([qc], [[1]]).result()
            options = qgt.options
            self.assertEqual(result.options.get("shots"), 200)
            self.assertEqual(options.get("shots"), 200)

        with self.subTest("QGT update"):
            qgt = LinCombQGT(estimator, options={"shots": 200})
            qgt.update_default_options(shots=100)
            options = qgt.options
            result = qgt.run([qc], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 100)
            self.assertEqual(options.get("shots"), 100)

        with self.subTest("QGT run"):
            qgt = LinCombQGT(estimator, options={"shots": 200})
            result = qgt.run([qc], [[0]], shots=300).result()
            options = qgt.options
            self.assertEqual(result.options.get("shots"), 300)
            self.assertEqual(options.get("shots"), 200)

    def test_operations_preserved(self):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x, y = Parameter("x"), Parameter("y")
        circuit = QuantumCircuit(2)
        circuit.initialize([0.5, 0.5, 0.5, 0.5])  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled
        circuit.ry(y, 0)

        values = [np.pi / 2, np.pi]
        expect = np.diag([0.25, 0.5]) / 4

        ops = []

        def operations_callback(op):
            ops.append(op)

        estimator = LoggingEstimator(operations_callback=operations_callback)
        qgt = LinCombQGT(estimator, derivative_type=DerivativeType.REAL)

        job = qgt.run([circuit], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            np.testing.assert_allclose(result.qgts[0], expect, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
