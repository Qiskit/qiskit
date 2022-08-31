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
from qiskit.algorithms.gradients import (FiniteDiffEstimatorGradient,
                                         LinCombEstimatorGradient,
                                         ParamShiftEstimatorGradient,
                                         SPSAEstimatorGradient)
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.circuit.library.standard_gates.rxx import RXXGate
from qiskit.circuit.library.standard_gates.ryy import RYYGate
from qiskit.circuit.library.standard_gates.rzx import RZXGate
from qiskit.circuit.library.standard_gates.rzz import RZZGate
from qiskit.primitives import Estimator, Sampler
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
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
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
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
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
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_almost_equal(gradients, correct_results[i], 3)

    @combine(
        grad=[FiniteDiffEstimatorGradient, ParamShiftEstimatorGradient, LinCombEstimatorGradient],
    )
    def test_gradient_2qubit_gate(self, grad):
        """Test the estimator gradient for 2 qubit gates"""
        estimator = Estimator()
        for gate in [RXXGate, RYYGate, RZZGate, RZXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [-0.70710678],
                [-1],
            ]
            op = SparsePauliOp.from_list([("ZI", 1)])
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                gradient = grad(estimator)

                if gate is RZZGate:
                    qc.h([0, 1])
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                    qc.h([0, 1])
                else:
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                gradients = gradient.run([qc], [op], [param]).result().gradients[0]
                np.testing.assert_almost_equal(gradients, correct_results[i], 3)

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
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_almost_equal(gradients, correct_results[i], 3)

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
        qc.rx(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [-0.70710678],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param], parameters=[[a]]).result().gradients[0]
            np.testing.assert_almost_equal(gradients, correct_results[i], 3)

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
        gradients = gradient.run([qc, qc2], [op] * 2, param_list).result().gradients
        np.testing.assert_almost_equal(gradients, correct_results, 3)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        correct_results2 = [
            [-0.70710678],
            [-0.5],
            [-0.5, -0.5],
        ]
        gradients2 = (
            gradient.run([qc, qc3, qc3], [op] * 3, param_list2, parameters=[[a], [c], None])
            .result()
            .gradients
        )
        np.testing.assert_almost_equal(gradients2[0], correct_results2[0], 3)
        np.testing.assert_almost_equal(gradients2[1], correct_results2[1], 3)
        np.testing.assert_almost_equal(gradients2[2], correct_results2[2], 3)

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
        with self.assertRaises(ValueError):
            _ = grad(Sampler())
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], [[np.pi / 4, np.pi / 4]])

    def test_spsa_gradient(self):
        """Test the SPSA estimator gradient"""
        estimator = Estimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        param_list = [[1, 1]]
        correct_results = [[-0.84147098, 0.84147098]]
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = SPSAEstimatorGradient(estimator, seed=123)
        gradients = gradient.run([qc], [op], param_list).result().gradients
        np.testing.assert_almost_equal(gradients, correct_results, 3)

        # multi parameters
        gradient = SPSAEstimatorGradient(estimator, seed=123)
        param_list2 = [[1, 1], [1, 1], [3, 3]]
        gradients2 = (
            gradient.run([qc] * 3, [op] * 3, param_list2, parameters=[None, [b], None])
            .result()
            .gradients
        )
        correct_results2 = [[-0.84147098, 0.84147098], [0.84147098], [-0.14112001, 0.14112001]]
        for grad, correct in zip(gradients2, correct_results2):
            np.testing.assert_almost_equal(grad, correct, 3)


if __name__ == "__main__":
    unittest.main()
