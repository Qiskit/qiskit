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

""" Test of Fraxis optimizer """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import Fraxis
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals


class TestOptimizerFraxis(QiskitAlgorithmsTestCase):
    """Test Fraxis optimizer with VQE"""

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

    def test_fraxis_default(self):
        """Test Fraxis optimizer with default parameters"""

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)

    def test_fraxis_maxiter(self):
        """Test Fraxis optimizer with maxiter"""
        maxiter = 20
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(maxiter=maxiter),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        self.assertEqual(result.cost_function_evals, maxiter * 6)

    def test_fraxis_xtol(self):
        """Test Fraxis optimizer with xtol"""
        xtol = 1e10
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertEqual(result.cost_function_evals, vqe.ansatz.num_parameters // 3 * 6)

    def test_fraxis_maxiter_xtol_1(self):
        """Test Fraxis optimizer with maxiter and xtol 1"""
        maxiter = 20
        xtol = 0
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(maxiter=maxiter, xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        self.assertEqual(result.cost_function_evals, maxiter * 6)

    def test_fraxis_maxiter_xtol_2(self):
        """Test Fraxis optimizer with maxiter and xtol 2"""
        maxiter = 20
        xtol = 1e10
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(maxiter=maxiter, xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertEqual(result.cost_function_evals, vqe.ansatz.num_parameters // 3 * 6)

    def test_fraxis_maxiter_xtol_3(self):
        """Test Fraxis optimizer with maxiter and xtol 3"""
        maxiter = 1000
        xtol = 1e-2
        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(maxiter=maxiter, xtol=xtol),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        self.assertLess(result.cost_function_evals, maxiter * 6)

    def test_fraxis_callback(self):
        """Test Fraxis optimizer with callback"""
        history = []
        xtol = 1e-2

        def callback(_, state):
            history.append(state.fun)

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(xtol=xtol, callback=callback),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        for fun1, fun2 in zip(history, history[1:]):
            self.assertGreaterEqual(fun1, fun2)

    def test_fraxis_callback_terminate(self):
        """Test Fraxis optimizer with callback to terminate"""
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
            optimizer=Fraxis(callback=callback),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertEqual(result.cost_function_evals, maxiter * 6)

    def test_fraxis_restart(self):
        """Test Fraxis optimizer with restart"""
        history = []
        maxiter = 10

        def callback(_, state):
            history.append(state.fun)

        vqe = VQE(
            estimator=Estimator(),
            ansatz=self.ansatz,
            optimizer=Fraxis(maxiter=maxiter, callback=callback),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        # restart with the previous VQE result
        vqe.initial_point = result.optimal_point
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        self.assertAlmostEqual(result.eigenvalue.real, self.expval, places=6)
        for fun1, fun2 in zip(history, history[1:]):
            self.assertGreaterEqual(fun1, fun2)

    def test_fraxis_errors(self):
        """Test Fraxis optimizer with errors"""
        ansatzes = [
            TwoLocal(rotation_blocks="ry", entanglement_blocks="cx", reps=3),
            TwoLocal(rotation_blocks="u", entanglement_blocks="crx", reps=2),
        ]
        for ansatz in ansatzes:
            with self.assertRaises(ValueError):
                vqe = VQE(
                    estimator=Estimator(),
                    ansatz=ansatz,
                    optimizer=Fraxis(),
                )
                _ = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)


if __name__ == "__main__":
    unittest.main()
