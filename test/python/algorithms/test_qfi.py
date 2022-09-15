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
from qiskit.circuit.parametervector import ParameterVector
from test import combine

import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.algorithms.gradients import (
    FiniteDiffEstimatorGradient,
    LinCombEstimatorGradient,
    ParamShiftEstimatorGradient,
    SPSAEstimatorGradient,
)
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.circuit.library.standard_gates import RXXGate, RYYGate, RZXGate, RZZGate
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.quantum_info.random import random_pauli_list
from qiskit.test import QiskitTestCase

from qiskit.algorithms.gradients.lin_comb_qfi import LinCombQFI


class TestQFI(QiskitTestCase):
    """Test QFI"""

    def setUp(self):
        super().setUp()
        self.estimator = Estimator()

    def test_qfi_simple(self):
        """Test if the quantum fisher information calculation is correct for a simple test case.

        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(a)]]
        """
        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)
        op = SparsePauliOp.from_list([("I", 1)])

        param_list = [[np.pi / 4, 0.1], [np.pi, 0.1], [np.pi / 2, 0.1]]
        correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]]

        qfi = LinCombQFI(self.estimator)
        for i, param in enumerate(param_list):
            print(param)
            qfis = qfi.run([qc], [op], [param]).result().qfis
            np.testing.assert_allclose(qfis[0], correct_values[i], atol=1e-3)

    def test_qfi_phase_fix(self):
        """Test the phase-fix argument in a QFI calculation

        QFI = [[1, 0], [0, 1]].
        """
        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        param = [np.pi / 4, 0.1]
        # test for different values
        correct_values = [[1, 0], [0, 1]]
        op = SparsePauliOp.from_list([("I", 1)])
        qfi = LinCombQFI(self.estimator)
        qfi_result = qfi.run([qc], [op], [param], [None], phase_fix=False).result().qfis
        np.testing.assert_allclose(qfi_result[0], correct_values, atol=1e-3)

    def test_qfi_maxcut(self):
        """Test the QFI for a simple MaxCut problem.

        This is interesting because it contains the same parameters in different gates.
        """
        # create maxcut circuit for the hamiltonian
        # H = (I ^ I ^ Z ^ Z) + (I ^ Z ^ I ^ Z) + (Z ^ I ^ I ^ Z) + (I ^ Z ^ Z ^ I)

        x = ParameterVector("x", 2)
        ansatz = QuantumCircuit(4)

        # initial hadamard layer
        ansatz.h(ansatz.qubits)

        # e^{iZZ} layers
        def expiz(qubit0, qubit1):
            ansatz.cx(qubit0, qubit1)
            ansatz.rz(2 * x[0], qubit1)
            ansatz.cx(qubit0, qubit1)

        expiz(2, 1)
        expiz(3, 0)
        expiz(2, 0)
        expiz(1, 0)

        # mixer layer with RX gates
        for i in range(ansatz.num_qubits):
            ansatz.rx(2 * x[1], i)

        reference = np.array([[16.0, -5.551], [-5.551, 18.497]])
        param = [0.4, 0.69]

        op = SparsePauliOp.from_list([("IIII", 1)])
        qfi = LinCombQFI(self.estimator)
        qfi_result = qfi.run([ansatz], [op], [param], [None]).result().qfis
        np.testing.assert_array_almost_equal(qfi_result[0], reference, decimal=3)

    def test_qfi_derivative(self):
        """Test the derivative option of QFI"""
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        op = SparsePauliOp.from_list([("I", 1)])
        qfi = LinCombQFI(self.estimator)
        # test imaginary derivative
        param_list = [[np.pi / 4, 0], [np.pi / 2, np.pi / 4]]
        correct_values = [[[0, 0.707106781], [0.707106781, -0.5]], [[0, 1], [1, 0]]]
        for i, param in enumerate(param_list):
            qfi_result = qfi.run([qc], [op], [param], [None], derivative="imag").result().qfis
            np.testing.assert_allclose(qfi_result[0], correct_values[i], atol=1e-3)

        # test real + imaginary derivative
        correct_values = [[[1, 0.707106781j], [0.707106781j, 0.5]], [[1, 1j], [1j, 1]]]
        for i, param in enumerate(param_list):
            qfi_result = qfi.run([qc], [op], [param], [None], derivative="both").result().qfis
            np.testing.assert_allclose(qfi_result[0], correct_values[i], atol=1e-3)

    def test_qfi_coefficients(self):
        """Test the derivative option of QFI"""
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[2] + qc.parameters[3].sin(), 1)

        op = SparsePauliOp.from_list([("II", 1)])

        qfi = LinCombQFI(self.estimator)
        # test imaginary derivative
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_values = [
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
        for i, param in enumerate(param_list):
            qfi_result = qfi.run([qc], [op], [param], [None]).result().qfis
            np.testing.assert_allclose(qfi_result[0], correct_values[i], atol=1e-3)

    def test_qfi_operators(self):
        """Test the QFI with different operators"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        param_list = [[np.pi / 4]]
        qfi = LinCombQFI(self.estimator)
        qfi_result = qfi.run(
            [qc], [op] , param_list
        ).result().qfis
        correct_values = [[-1.20710678]]
        np.testing.assert_allclose(qfi_result[0], correct_values, atol=1e-3)
        op = Operator.from_label("Z")
        qfi_result = qfi.run(
            [qc], [op] , param_list
        ).result().qfis
        np.testing.assert_allclose(qfi_result[0], correct_values, atol=1e-3)

    def test_qfi_specify_parameters(self):
        """Test the QFI with specified parameters"""
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.ry(b, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        qfi = LinCombQFI(self.estimator)
        param_list = [np.pi / 4, np.pi / 4]
        qfi_result = qfi.run(
            [qc], [op] , [param_list], [[a]]
        ).result().qfis
        np.testing.assert_allclose(qfi_result[0], [[-1.25]], atol=1e-3)

    def test_qfi_multi_arguments(self):
        """Test the QFI for multiple arguments"""
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.ry(b, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(a, 0)
        qc2.ry(b, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        qfi = LinCombQFI(self.estimator)

        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_values = [
            [[-1.25]],
            [[-1, 1], [1, 0]],
        ]
        param_list = [[np.pi / 4, np.pi / 4], [np.pi / 2, np.pi / 2]]
        qfi_results = qfi.run([qc, qc2], [op] * 2, param_list, [[a], None]).result().qfis
        for i, _ in enumerate(param_list):
            np.testing.assert_allclose(qfi_results[i], correct_values[i], atol=1e-3)



    def test_gradient_validation(self):
        """Test estimator gradient's validation"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qfi = LinCombQFI(self.estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        op = SparsePauliOp.from_list([("Z", 1)])
        with self.assertRaises(ValueError):
            qfi.run([qc], [op], param_list)
        with self.assertRaises(ValueError):
            qfi.run([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            qfi.run([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            qfi.run([qc], [op], [[np.pi / 4, np.pi / 4]])

    def test_run_options(self):
        """Test estimator gradient's run options"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        estimator = Estimator(run_options={"shots": 100})

        with self.subTest("estimator"):
            qfi = LinCombQFI(estimator)
            result = qfi.run([qc], [op], [[1]]).result()
            self.assertEqual(result.run_options.get("shots"), 100)

        with self.subTest("gradient init"):
            qfi = LinCombQFI(estimator, run_options={"shots": 200})
            result = qfi.run([qc], [op], [[1]]).result()
            self.assertEqual(result.run_options.get("shots"), 200)

        with self.subTest("gradient run"):
            qfi = LinCombQFI(estimator, run_options={"shots": 200})
            result = qfi.run([qc], [op], [[1]], shots=300).result()
            self.assertEqual(result.run_options.get("shots"), 300)


if __name__ == "__main__":
    unittest.main()
