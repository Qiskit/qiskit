# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of Fraxis and FQS optimizers"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import data, ddt

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import FQS, Fraxis
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals


@ddt
class TestFraxisFQS(QiskitAlgorithmsTestCase):
    """Test Fraxis and FQS optimizers with VQE"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        self.qubit_op = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expval = -1.857275
        self.ansatz = TwoLocal(rotation_blocks="u", entanglement_blocks="cx")
        self.optimizer = {"fraxis": Fraxis, "fqs": FQS}
        self.num_evals = {"fraxis": 6, "fqs": 10}

    @data("fraxis", "fqs")
    def test_default(self, method):
        """Test optimizer with default parameters"""
        optimizer = self.optimizer[method]
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)

    @data("fraxis", "fqs")
    def test_maxiter(self, method):
        """Test optimizer with maxiter"""
        optimizer = self.optimizer[method]
        num_evals = self.num_evals[method]
        maxiter = 20
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        self.assertEqual(result.cost_function_evals, maxiter * num_evals)

    @data("fraxis", "fqs")
    def test_xtol(self, method):
        """Test optimizer with xtol"""
        optimizer = self.optimizer[method]
        num_evals = self.num_evals[method]
        xtol = 1e10
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertEqual(result.cost_function_evals, vqe.ansatz.num_parameters // 3 * num_evals)

    @data("fraxis", "fqs")
    def test_maxiter_xtol_1(self, method):
        """Test optimizer with maxiter and xtol 1"""
        optimizer = self.optimizer[method]
        num_evals = self.num_evals[method]
        maxiter = 20
        xtol = 0
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter, xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        self.assertEqual(result.cost_function_evals, maxiter * num_evals)

    @data("fraxis", "fqs")
    def test_maxiter_xtol_2(self, method):
        """Test optimizer with maxiter and xtol 2"""
        optimizer = self.optimizer[method]
        num_evals = self.num_evals[method]
        maxiter = 20
        xtol = 1e10
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter, xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertEqual(result.cost_function_evals, vqe.ansatz.num_parameters // 3 * num_evals)

    @data("fraxis", "fqs")
    def test_maxiter_xtol_3(self, method):
        """Test optimizer with maxiter and xtol 3"""
        optimizer = self.optimizer[method]
        num_evals = self.num_evals[method]
        maxiter = 1000
        xtol = 1e-2
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter, xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        self.assertLess(result.cost_function_evals, maxiter * num_evals)

    @data("fraxis", "fqs")
    def test_callback(self, method):
        """Test optimizer with callback"""
        optimizer = self.optimizer[method]
        history = []
        maxiter = 20

        def callback(_, state):
            history.append(state.fun)

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter, callback=callback),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        for fun1, fun2 in zip(history, history[1:]):
            self.assertGreaterEqual(fun1, fun2)

    @data("fraxis", "fqs")
    def test_callback_terminate(self, method):
        """Test optimizer with callback to terminate"""
        optimizer = self.optimizer[method]
        num_evals = self.num_evals[method]
        maxiter = 10
        count = 0

        def callback(*_):
            nonlocal count
            count += 1
            if count == maxiter:
                return True
            return False

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(callback=callback),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertEqual(result.cost_function_evals, maxiter * num_evals)

    @data("fraxis", "fqs")
    def test_restart(self, method):
        """Test optimizer with restart"""
        optimizer = self.optimizer[method]
        history = []
        maxiter = 10

        def callback(_, state):
            history.append(state.fun)

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter, callback=callback),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        # restart with the previous VQE result
        vqe.initial_point = result.optimal_point
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        for fun1, fun2 in zip(history, history[1:]):
            self.assertGreaterEqual(fun1, fun2)

    @data("fraxis", "fqs")
    def test_initial_point(self, method):
        """Test optimizer with initial_point"""
        optimizer = self.optimizer[method]
        maxiter = 10
        self.ansatz.num_qubits = self.qubit_op.num_qubits
        initial_point = np.arange(self.ansatz.num_parameters)
        ref = initial_point.copy()

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=optimizer(maxiter=maxiter),
            initial_point=initial_point,
        )
        _ = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        np.testing.assert_allclose(initial_point, ref)

    @data("fraxis", "fqs")
    def test_errors(self, method):
        """Test optimizer with errors"""
        optimizer = self.optimizer[method]
        ansatzes = [
            TwoLocal(rotation_blocks="ry", entanglement_blocks="cx", reps=3),
            TwoLocal(rotation_blocks="u", entanglement_blocks="crx", reps=2),
        ]
        for ansatz in ansatzes:
            with self.assertRaises(ValueError):
                vqe = VQE(
                    estimator=Estimator(),
                    ansatz=ansatz,
                    optimizer=optimizer(),
                )
                _ = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)


if __name__ == "__main__":
    unittest.main()
