# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test SSVQE """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from functools import partial
import numpy as np
from ddt import data, ddt

from qiskit import QuantumCircuit
from qiskit.algorithms.eigensolvers import SSVQE
from qiskit.algorithms import AlgorithmError
from qiskit.algorithms.optimizers import (
    COBYLA,
    L_BFGS_B,
    SLSQP,
    CG,
    TNC,
    P_BFGS,
    GradientDescent,
    OptimizerResult,
)

from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Sampler, Estimator
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.utils import algorithm_globals
from qiskit.quantum_info.operators import Operator

# pylint: disable=invalid-name
def _mock_optimizer(fun, x0, jac=None, bounds=None, inputs=None) -> OptimizerResult:
    """A mock of a callable that can be used as minimizer in SSVQE."""
    result = OptimizerResult()
    result.x = np.zeros_like(x0)
    result.fun = fun(result.x)
    result.nit = 0

    if inputs is not None:
        inputs.update({"fun": fun, "x0": x0, "jac": jac, "bounds": bounds})
    return result


I = PauliSumOp.from_list([("I", 1)])  # pylint: disable=invalid-name
X = PauliSumOp.from_list([("X", 1)])  # pylint: disable=invalid-name
Z = PauliSumOp.from_list([("Z", 1)])  # pylint: disable=invalid-name

H2_PAULI = (
    -1.052373245772859 * (I ^ I)
    + 0.39793742484318045 * (I ^ Z)
    - 0.39793742484318045 * (Z ^ I)
    - 0.01128010425623538 * (Z ^ Z)
    + 0.18093119978423156 * (X ^ X)
)

H2_OP = Operator(H2_PAULI.to_matrix())


@ddt
class TestSSVQE(QiskitAlgorithmsTestCase):
    """Test SSVQE"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

        self.h2_energy = -1.85727503
        self.h2_energy_excited = [-1.85727503, -1.24458455]

        self.ryrz_wavefunction = TwoLocal(
            rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1
        )
        self.ry_wavefunction = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        self.estimator = Estimator()
        self.estimator_shots = Estimator(options={"shots": 2048, "seed": self.seed})
        self.fidelity = ComputeUncompute(Sampler())
        self.betas = [50, 50]

    @data(H2_PAULI, H2_OP)
    def test_basic_operator(self, op):
        """Test SSVQE without aux_operators."""
        wavefunction = self.ryrz_wavefunction
        ssvqe = SSVQE(
            estimator=self.estimator,
            ansatz=wavefunction,
            optimizer=COBYLA(),
        )

        result = ssvqe.compute_eigenvalues(operator=op)

        with self.subTest(msg="test eigenvalue"):
            np.testing.assert_array_almost_equal(
                result.eigenvalues.real, self.h2_energy_excited, decimal=1
            )

        with self.subTest(msg="test dimension of optimal point"):
            self.assertEqual(len(result.optimal_point), 8)

        with self.subTest(msg="assert cost_function_evals is set"):
            self.assertIsNotNone(result.cost_function_evals)

        with self.subTest(msg="assert optimizer_time is set"):
            self.assertIsNotNone(result.optimizer_time)

    @data(H2_PAULI, H2_OP)
    def test_mismatching_num_qubits(self, op):
        """Ensuring circuit and operator mismatch is caught"""
        wavefunction = QuantumCircuit(1)
        optimizer = SLSQP(maxiter=50)
        ssvqe = SSVQE(estimator=self.estimator, k=1, ansatz=wavefunction, optimizer=optimizer)
        with self.assertRaises(AlgorithmError):
            _ = ssvqe.compute_eigenvalues(operator=op)

    @data(H2_PAULI, H2_OP)
    def test_missing_varform_params(self, op):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(op.num_qubits)
        ssvqe = SSVQE(estimator=self.estimator, ansatz=circuit, optimizer=SLSQP(), k=1)
        with self.assertRaises(AlgorithmError):
            ssvqe.compute_eigenvalues(operator=op)

    @data(H2_PAULI, H2_OP)
    def test_callback(self, op):
        """Test the callback on SSVQE."""
        history = {"eval_count": [], "parameters": [], "mean_energies": [], "metadata": []}

        def store_intermediate_result(eval_count, parameters, mean, metadata):
            history["eval_count"].append(eval_count)
            history["parameters"].append(parameters)
            history["mean_energies"].append(mean)
            history["metadata"].append(metadata)

        optimizer = COBYLA(maxiter=3)
        wavefunction = self.ry_wavefunction

        ssvqe = SSVQE(
            estimator=self.estimator,
            ansatz=wavefunction,
            optimizer=optimizer,
            callback=store_intermediate_result,
        )

        ssvqe.compute_eigenvalues(operator=op)

        self.assertTrue(all(isinstance(count, int) for count in history["eval_count"]))
        # self.assertTrue(all(isinstance(mean, float) for mean in history["mean_energies"]))
        self.assertTrue(
            all(
                isinstance(mean, float)
                for mean_list in history["mean_energies"]
                for mean in mean_list
            )
        )
        self.assertTrue(all(isinstance(metadata, dict) for metadata in history["metadata"]))
        for params in history["parameters"]:
            self.assertTrue(all(isinstance(param, float) for param in params))

        ref_eval_count = [1, 2, 3]
        ref_mean = [[-1.07, -1.44], [-1.45, -1.06], [-1.37, -0.94]]

        np.testing.assert_array_almost_equal(history["eval_count"], ref_eval_count, decimal=0)
        np.testing.assert_array_almost_equal(history["mean_energies"], ref_mean, decimal=2)

    @data(H2_PAULI, H2_OP)
    def test_ssvqe_optimizer(self, op):
        """Test running same SSVQE twice to re-use optimizer, then switch optimizer"""
        ssvqe = SSVQE(
            estimator=self.estimator, ansatz=RealAmplitudes(reps=6), optimizer=SLSQP(), k=2
        )

        def run_check():
            result = ssvqe.compute_eigenvalues(operator=op)
            np.testing.assert_array_almost_equal(
                result.eigenvalues.real, self.h2_energy_excited, decimal=3
            )

        run_check()

        with self.subTest("Optimizer re-use"):
            run_check()

        with self.subTest("Optimizer replace"):
            ssvqe.optimizer = L_BFGS_B()
            run_check()

    @data(H2_PAULI, H2_OP)
    def test_aux_operators_list(self, op):
        """Test list-based aux_operators."""
        wavefunction = self.ry_wavefunction
        ssvqe = SSVQE(estimator=self.estimator, ansatz=wavefunction, optimizer=SLSQP(), k=2)

        # Start with an empty list
        result = ssvqe.compute_eigenvalues(op, aux_operators=[])
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=2
        )
        self.assertIsNone(result.aux_operators_evaluated)

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]
        result = ssvqe.compute_eigenvalues(op, aux_operators=aux_ops)
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=2
        )
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operators_evaluated[0][0][0], 2, places=2)
        self.assertAlmostEqual(result.aux_operators_evaluated[0][1][0], 0, places=2)
        # metadata
        self.assertIsInstance(result.aux_operators_evaluated[0][1][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][1][1], dict)

        # Go again with additional None and zero operators
        extra_ops = [*aux_ops, None, 0]
        result = ssvqe.compute_eigenvalues(op, aux_operators=extra_ops)
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=2
        )
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operators_evaluated[0][0][0], 2, places=2)
        self.assertAlmostEqual(result.aux_operators_evaluated[0][1][0], 0, places=2)
        self.assertEqual(result.aux_operators_evaluated[0][2][0], 0.0)
        self.assertEqual(result.aux_operators_evaluated[0][3][0], 0.0)
        # metadata
        self.assertIsInstance(result.aux_operators_evaluated[0][0][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][1][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][3][1], dict)

    @data(H2_PAULI, H2_OP)
    def test_aux_operators_dict(self, op):
        """Test dictionary compatibility of aux_operators"""
        wavefunction = self.ry_wavefunction
        ssvqe = SSVQE(estimator=self.estimator, ansatz=wavefunction, optimizer=SLSQP())

        # Start with an empty dictionary
        result = ssvqe.compute_eigenvalues(op, aux_operators={})
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=2
        )
        self.assertIsNone(result.aux_operators_evaluated)

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = {"aux_op1": aux_op1, "aux_op2": aux_op2}
        result = ssvqe.compute_eigenvalues(op, aux_operators=aux_ops)
        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503, 2)
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        self.assertEqual(len(result.aux_operators_evaluated[0]), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operators_evaluated[0]["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operators_evaluated[0]["aux_op2"][0], 0, places=1)
        # metadata
        self.assertIsInstance(result.aux_operators_evaluated[0]["aux_op1"][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0]["aux_op2"][1], dict)

        # Go again with additional None and zero operators
        extra_ops = {**aux_ops, "None_operator": None, "zero_operator": 0}
        result = ssvqe.compute_eigenvalues(op, aux_operators=extra_ops)
        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503, places=5)
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        self.assertEqual(len(result.aux_operators_evaluated[0]), 3)
        # expectation values
        self.assertAlmostEqual(result.aux_operators_evaluated[0]["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operators_evaluated[0]["aux_op2"][0], 0.0, places=2)
        self.assertEqual(result.aux_operators_evaluated[0]["zero_operator"][0], 0.0)
        self.assertTrue("None_operator" not in result.aux_operators_evaluated[0].keys())
        # metadata
        self.assertIsInstance(result.aux_operators_evaluated[0]["aux_op1"][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0]["aux_op2"][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0]["zero_operator"][1], dict)

    @data(H2_PAULI, H2_OP)
    def test_aux_operator_std_dev(self, op):
        """Test non-zero standard deviations of aux operators."""
        wavefunction = self.ry_wavefunction
        ssvqe = SSVQE(
            estimator=self.estimator,
            ansatz=wavefunction,
            initial_point=[
                1.70256666,
                -5.34843975,
                -0.39542903,
                5.99477786,
                -2.74374986,
                -4.85284669,
                0.2442925,
                -1.51638917,
            ],
            optimizer=COBYLA(maxiter=0),
        )

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]
        result = ssvqe.compute_eigenvalues(op, aux_operators=aux_ops)
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operators_evaluated[0][0][0], 2.0, places=1)
        self.assertAlmostEqual(
            result.aux_operators_evaluated[0][1][0], 0.0019531249999999445, places=1
        )
        # metadata
        self.assertIsInstance(result.aux_operators_evaluated[0][0][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][1][1], dict)

        # Go again with additional None and zero operators
        aux_ops = [*aux_ops, None, 0]
        result = ssvqe.compute_eigenvalues(op, aux_operators=aux_ops)
        self.assertEqual(len(result.aux_operators_evaluated[0]), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operators_evaluated[0][0][0], 2.0, places=1)
        self.assertAlmostEqual(
            result.aux_operators_evaluated[0][1][0], 0.0019531249999999445, places=1
        )
        self.assertEqual(result.aux_operators_evaluated[0][2][0], 0.0)
        self.assertEqual(result.aux_operators_evaluated[0][3][0], 0.0)
        # metadata
        self.assertIsInstance(result.aux_operators_evaluated[0][0][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][1][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][2][1], dict)
        self.assertIsInstance(result.aux_operators_evaluated[0][3][1], dict)

    @data(
        CG(),
        L_BFGS_B(),
        P_BFGS(),
        SLSQP(),
        TNC(),
    )
    def test_with_gradient(self, optimizer):
        """Test SSVQE using gradient primitive."""
        estimator = Estimator()
        ssvqe = SSVQE(
            k=2,
            estimator=estimator,
            ansatz=self.ry_wavefunction,
            optimizer=optimizer,
            gradient=ParamShiftEstimatorGradient(estimator),
        )
        result = ssvqe.compute_eigenvalues(operator=H2_PAULI)
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=5
        )

    def test_gradient_passed(self):
        """Test the gradient is properly passed into the optimizer."""
        inputs = {}
        estimator = Estimator()
        ssvqe = SSVQE(
            k=2,
            estimator=estimator,
            ansatz=RealAmplitudes(),
            optimizer=partial(_mock_optimizer, inputs=inputs),
            gradient=ParamShiftEstimatorGradient(estimator),
        )
        _ = ssvqe.compute_eigenvalues(operator=H2_PAULI)

        self.assertIsNotNone(inputs["jac"])

    def test_gradient_run(self):
        """Test using the gradient to calculate the minimum."""
        estimator = Estimator()
        ssvqe = SSVQE(
            k=2,
            estimator=estimator,
            ansatz=RealAmplitudes(),
            optimizer=GradientDescent(maxiter=200, learning_rate=0.1),
            gradient=ParamShiftEstimatorGradient(estimator),
        )
        result = ssvqe.compute_eigenvalues(operator=H2_PAULI)
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=5
        )

    def test_max_evals_grouped(self):
        """Test with SLSQP with max_evals_grouped."""
        optimizer = SLSQP(maxiter=50, max_evals_grouped=5)
        ssvqe = SSVQE(
            k=2,
            estimator=Estimator(),
            ansatz=RealAmplitudes(reps=6),
            optimizer=optimizer,
        )
        result = ssvqe.compute_eigenvalues(operator=H2_PAULI)
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, self.h2_energy_excited, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
