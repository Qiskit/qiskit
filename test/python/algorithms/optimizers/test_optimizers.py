# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Optimizers """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from scipy.optimize import rosen, rosen_der

from qiskit.algorithms.optimizers import (
    ADAM,
    CG,
    COBYLA,
    GSLS,
    GradientDescent,
    L_BFGS_B,
    NELDER_MEAD,
    Optimizer,
    P_BFGS,
    POWELL,
    SLSQP,
    SPSA,
    QNSPSA,
    TNC,
    SciPyOptimizer,
)
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.utils import algorithm_globals


class TestOptimizers(QiskitAlgorithmsTestCase):
    """Test Optimizers"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 52

    def _optimize(self, optimizer, grad=False):
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        if grad:
            res = optimizer.optimize(
                len(x_0), rosen, gradient_function=rosen_der, initial_point=x_0
            )
        else:
            res = optimizer.optimize(len(x_0), rosen, initial_point=x_0)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x_0), decimal=2)
        return res

    def test_adam(self):
        """adam test"""
        optimizer = ADAM(maxiter=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cg(self):
        """cg test"""
        optimizer = CG(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_gradient_descent(self):
        """cg test"""
        optimizer = GradientDescent(maxiter=100000, tol=1e-06, learning_rate=1e-3)
        res = self._optimize(optimizer, grad=True)
        self.assertLessEqual(res[2], 100000)

    def test_cobyla(self):
        """cobyla test"""
        optimizer = COBYLA(maxiter=100000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_l_bfgs_b(self):
        """l_bfgs_b test"""
        optimizer = L_BFGS_B(maxfun=1000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_p_bfgs(self):
        """parallel l_bfgs_b test"""
        optimizer = P_BFGS(maxfun=1000, max_processes=4)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_nelder_mead(self):
        """nelder mead test"""
        optimizer = NELDER_MEAD(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_powell(self):
        """powell test"""
        optimizer = POWELL(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_slsqp(self):
        """slsqp test"""
        optimizer = SLSQP(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        """spsa test"""
        optimizer = SPSA(maxiter=10000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_tnc(self):
        """tnc test"""
        optimizer = TNC(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_gsls(self):
        """gsls test"""
        optimizer = GSLS(
            sample_size_factor=40,
            sampling_radius=1.0e-12,
            maxiter=10000,
            max_eval=10000,
            min_step_size=1.0e-12,
        )
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        _, x_value, n_evals = optimizer.optimize(len(x_0), rosen, initial_point=x_0)

        # Ensure value is near-optimal
        self.assertLessEqual(x_value, 0.01)
        self.assertLessEqual(n_evals, 10000)

    def test_scipy_optimizer(self):
        """scipy_optimizer test"""
        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_scipy_optimizer_callback(self):
        """scipy_optimizer callback test"""
        values = []

        def callback(x):
            values.append(x)

        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000}, callback=callback)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)
        self.assertTrue(values)  # Check the list is nonempty.


class TestOptimizerSerialization(QiskitAlgorithmsTestCase):
    """Tests concerning the serialization of optimizers."""

    def test_scipy(self):
        """Test the SciPyOptimizer is serializable."""
        method = "BFGS"
        options = {"maxiter": 1000, "eps": np.array([0.1])}

        optimizer = SciPyOptimizer(method, options=options)
        serialized = optimizer.to_dict()
        from_dict = SciPyOptimizer.from_dict(serialized)

        self.assertEqual(from_dict._method, method.lower())
        self.assertEqual(from_dict._options, options)

    def test_scipy_not_serializable(self):
        """Test serialization fails if the optimizer contains an attribute that's not supported."""

        def callback(x):
            print(x)

        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1}, callback=callback)

        with self.assertRaises(QiskitError):
            _ = optimizer.to_dict()

    def test_spsa(self):
        """Test SPSA optimizer is serializable."""
        options = {"maxiter": 100, "blocking": True, "allowed_increase": 0.1,
                   "second_order": True, "learning_rate": 0.02, "perturbation": 0.05,
                   "regularization": 0.1, "resamplings": 2, "perturbation_dims": 5,
                   "trust_region": False, "initial_hessian": None, "hessian_delay": 0}
        spsa = SPSA(**options)

        serialized = spsa.to_dict()
        expected = options.copy()
        expected["name"] = "SPSA"

        with self.subTest(msg="check constructed dictionary"):
            self.assertDictEqual(serialized, expected)

        reconstructed = Optimizer.from_dict(serialized)
        with self.subTest(msg="test reconstructed optimizer"):
            self.assertDictEqual(reconstructed.to_dict(), expected)

    def test_qnspsa(self):
        """Test QN-SPSA optimizer is serializable."""
        ansatz = RealAmplitudes(1)
        fidelity = QNSPSA.get_fidelity(ansatz)
        options = {"fidelity": fidelity,
                   "maxiter": 100, "blocking": True, "allowed_increase": 0.1,
                   "learning_rate": 0.02, "perturbation": 0.05,
                   "regularization": 0.1, "resamplings": 2, "perturbation_dims": 5,
                   "initial_hessian": None, "hessian_delay": 0}
        spsa = QNSPSA(**options)

        serialized = spsa.to_dict()
        expected = options.copy()
        expected.pop("fidelity")  # fidelity cannot be serialized
        expected["name"] = "QNSPSA"

        with self.subTest(msg="check constructed dictionary"):
            self.assertDictEqual(serialized, expected)

        # fidelity cannot be serialized, so it must be added back in
        serialized["fidelity"] = fidelity
        reconstructed = Optimizer.from_dict(serialized)
        with self.subTest(msg="test reconstructed optimizer"):
            self.assertDictEqual(reconstructed.to_dict(), expected)


if __name__ == "__main__":
    unittest.main()
