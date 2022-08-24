# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

""" Test Quantum Gradient Framework """

import unittest
from test import combine

import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.algorithms.gradients.finite_diff_estimator_gradient import FiniteDiffEstimatorGradient
from qiskit.algorithms.gradients.lin_comb_estimator_gradient import LinCombEstimatorGradient
from qiskit.algorithms.gradients.param_shift_estimator_gradient import ParamShiftEstimatorGradient
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.test import QiskitTestCase


@ddt
class TestEstimatorGradient(QiskitTestCase):
    """Test Estimator Gradient"""

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_p(self, grad):
        """Test the estimator gradient for p"""
        estimator = Estimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        correct_results = [[-1 / np.sqrt(2)], [0], [-1]]
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            for j, value in enumerate(values):
                self.assertAlmostEqual(value, correct_results[i][j], 3)

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_u(self, grad):
        """Test the estimator gradient for u"""
        estimator = Estimator()
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])

        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        correct_results = [[-0.70710678, 0.0, 0.0], [-0.35355339, -0.85355339, -0.85355339]]
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            for j, value in enumerate(values):
                self.assertAlmostEqual(value, correct_results[i][j], 3)

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_efficient_su2(self, grad):
        """Test the estimator gradient for EfficientSU2"""
        estimator = Estimator()
        qc = EfficientSU2(2, reps=1)
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = grad(estimator)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_results = [
            [
                -0.35355339,
                -0.70710678,
                0,
                0.35355339,
                0,
                -0.70710678,
                0,
                0,
            ],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_rxx(self, grad):
        """Test the estimator gradient for rxx"""
        estimator = Estimator()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.rxx(a, 0, 1)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_ryy(self, grad):
        """Test the estimator gradient for ryy"""
        estimator = Estimator()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ryy(a, 0, 1)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_rzz(self, grad):
        """Test the estimator gradient for rzz"""
        estimator = Estimator()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.rzz(a, 0, 1)
        qc.h([0, 1])
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_rzx(self, grad):
        """Test the estimator gradient for rzx"""
        estimator = Estimator()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.rzx(a, 0, 1)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_parameter_coefficient(self, grad):
        """Test the estimator gradient for parameter variables with coefficients"""
        estimator = Estimator()
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        gradient = grad(estimator)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [-0.7266653, -0.4905135, -0.0068606, -0.9228880],
            [-3.5972095, 0.10237173, -0.3117748, 0],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_parameters(self, grad):
        """Test the estimator gradient for parameters"""
        estimator = Estimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rz(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [-0.70710678, 0],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        for i, param in enumerate(param_list):
            values = gradient.evaluate([qc], [op], [param]).values[0]
            np.testing.assert_almost_equal(values, correct_results[i])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_multi_arguments(self, grad):
        """Test the estimator gradient for multiple arguments"""
        estimator = Estimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        values = gradient.evaluate([qc, qc2], [op] * 2, param_list).values
        np.testing.assert_almost_equal(values, correct_results)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        correct_results2 = [
            [-0.70710678],
            [-0.5 if p == c else 0 for p in qc3.parameters],
            [-0.5, -0.5],
        ]
        values2 = gradient.evaluate(
            [qc, qc3, qc3], [op] * 3, param_list2, parameters=[[a], [c], None]
        ).values
        np.testing.assert_almost_equal(values2[0], correct_results2[0])
        np.testing.assert_almost_equal(values2[1], correct_results2[1])
        np.testing.assert_almost_equal(values2[2], correct_results2[2])

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient]
    )
    def test_gradient_validation(self, grad):
        """Test estimator gradient's validation"""
        estimator = Estimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        op = SparsePauliOp.from_list([("Z", 1)])
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc], [op], param_list)
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc], [op], [[np.pi / 4, np.pi / 4]])


if __name__ == "__main__":
    unittest.main()
