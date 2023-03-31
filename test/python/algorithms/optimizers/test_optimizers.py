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

"""Test Optimizers"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from typing import Optional, List, Tuple
from ddt import ddt, data, unpack
import numpy as np
from scipy.optimize import rosen, rosen_der

from qiskit.algorithms.optimizers import (
    ADAM,
    AQGD,
    BOBYQA,
    IMFIL,
    CG,
    CRS,
    COBYLA,
    DIRECT_L,
    DIRECT_L_RAND,
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
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils import algorithm_globals, optionals


@ddt
class TestOptimizers(QiskitAlgorithmsTestCase):
    """Test Optimizers"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 52

    def run_optimizer(
        self,
        optimizer: Optimizer,
        max_nfev: int,
        grad: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Test the optimizer.

        Args:
            optimizer: The optimizer instance to test.
            max_nfev: The maximal allowed number of function evaluations.
            grad: Whether to pass the gradient function as input.
            bounds: Optimizer bounds.
        """
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        jac = rosen_der if grad else None

        res = optimizer.minimize(rosen, x_0, jac, bounds)
        x_opt = res.x
        nfev = res.nfev

        np.testing.assert_array_almost_equal(x_opt, [1.0] * len(x_0), decimal=2)
        self.assertLessEqual(nfev, max_nfev)

    def test_adam(self):
        """adam test"""
        optimizer = ADAM(maxiter=10000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_cg(self):
        """cg test"""
        optimizer = CG(maxiter=1000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_gradient_descent(self):
        """cg test"""
        optimizer = GradientDescent(maxiter=100000, tol=1e-06, learning_rate=1e-3)
        self.run_optimizer(optimizer, grad=True, max_nfev=100000)

    def test_cobyla(self):
        """cobyla test"""
        optimizer = COBYLA(maxiter=100000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=100000)

    def test_l_bfgs_b(self):
        """l_bfgs_b test"""
        optimizer = L_BFGS_B(maxfun=1000)
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_p_bfgs(self):
        """parallel l_bfgs_b test"""
        optimizer = P_BFGS(maxfun=1000, max_processes=4)
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_nelder_mead(self):
        """nelder mead test"""
        optimizer = NELDER_MEAD(maxfev=10000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_powell(self):
        """powell test"""
        optimizer = POWELL(maxfev=10000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_slsqp(self):
        """slsqp test"""
        optimizer = SLSQP(maxiter=1000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        """spsa test"""
        optimizer = SPSA(maxiter=10000)
        self.run_optimizer(optimizer, max_nfev=100000)

    def test_tnc(self):
        """tnc test"""
        optimizer = TNC(maxiter=1000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

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

        algorithm_globals.random_seed = 1
        res = optimizer.minimize(rosen, x_0)
        x_value = res.fun
        n_evals = res.nfev

        # Ensure value is near-optimal
        self.assertLessEqual(x_value, 0.01)
        self.assertLessEqual(n_evals, 10000)

    def test_scipy_optimizer(self):
        """scipy_optimizer test"""
        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000})
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_scipy_optimizer_callback(self):
        """scipy_optimizer callback test"""
        values = []

        def callback(x):
            values.append(x)

        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000}, callback=callback)
        self.run_optimizer(optimizer, max_nfev=10000)
        self.assertTrue(values)  # Check the list is nonempty.

    # ESCH and ISRES do not do well with rosen
    @data(
        (CRS, True),
        (DIRECT_L, True),
        (DIRECT_L_RAND, True),
        (CRS, False),
        (DIRECT_L, False),
        (DIRECT_L_RAND, False),
    )
    @unpack
    def test_nlopt(self, optimizer_cls, use_bound):
        """NLopt test"""
        bounds = [(-6, 6)] * 5 if use_bound else None
        try:
            optimizer = optimizer_cls()
            optimizer.set_options(**{"max_evals": 50000})
            self.run_optimizer(optimizer, max_nfev=50000, bounds=bounds)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))


@ddt
class TestOptimizerSerialization(QiskitAlgorithmsTestCase):
    """Tests concerning the serialization of optimizers."""

    @data(
        ("BFGS", {"maxiter": 100, "eps": np.array([0.1])}),
        ("CG", {"maxiter": 200, "gtol": 1e-8}),
        ("COBYLA", {"maxiter": 10}),
        ("L_BFGS_B", {"maxiter": 30}),
        ("NELDER_MEAD", {"maxiter": 0}),
        ("NFT", {"maxiter": 100}),
        ("P_BFGS", {"maxiter": 5}),
        ("POWELL", {"maxiter": 1}),
        ("SLSQP", {"maxiter": 400}),
        ("TNC", {"maxiter": 20}),
        ("dogleg", {"maxiter": 100}),
        ("trust-constr", {"maxiter": 10}),
        ("trust-ncg", {"maxiter": 100}),
        ("trust-exact", {"maxiter": 120}),
        ("trust-krylov", {"maxiter": 150}),
    )
    @unpack
    def test_scipy(self, method, options):
        """Test the SciPyOptimizer is serializable."""

        optimizer = SciPyOptimizer(method, options=options)
        serialized = optimizer.settings
        from_dict = SciPyOptimizer(**serialized)

        self.assertEqual(from_dict._method, method.lower())
        self.assertEqual(from_dict._options, options)

    def test_independent_reconstruction(self):
        """Test the SciPyOptimizers don't reset all settings upon creating a new instance.

        COBYLA is used as representative example here."""

        kwargs = {"coffee": "without sugar"}
        options = {"tea": "with milk"}
        optimizer = COBYLA(maxiter=1, options=options, **kwargs)
        serialized = optimizer.settings
        from_dict = COBYLA(**serialized)

        with self.subTest(msg="test attributes"):
            self.assertEqual(from_dict.settings["maxiter"], 1)

        with self.subTest(msg="test options"):
            # options should only contain values that are *not* already in the initializer
            # (e.g. should not contain maxiter)
            self.assertEqual(from_dict.settings["options"], {"tea": "with milk"})

        with self.subTest(msg="test kwargs"):
            self.assertEqual(from_dict.settings["coffee"], "without sugar")

        with self.subTest(msg="option ids differ"):
            self.assertNotEqual(id(serialized["options"]), id(from_dict.settings["options"]))

    def test_adam(self):
        """Test ADAM is serializable."""

        adam = ADAM(maxiter=100, amsgrad=True)
        settings = adam.settings

        self.assertEqual(settings["maxiter"], 100)
        self.assertTrue(settings["amsgrad"])

    def test_aqgd(self):
        """Test AQGD is serializable."""

        opt = AQGD(maxiter=[200, 100], eta=[0.2, 0.1], momentum=[0.25, 0.1])
        settings = opt.settings

        self.assertListEqual(settings["maxiter"], [200, 100])
        self.assertListEqual(settings["eta"], [0.2, 0.1])
        self.assertListEqual(settings["momentum"], [0.25, 0.1])

    @unittest.skipIf(not optionals.HAS_SKQUANT, "Install scikit-quant to run this test.")
    def test_bobyqa(self):
        """Test BOBYQA is serializable."""

        opt = BOBYQA(maxiter=200)
        settings = opt.settings

        self.assertEqual(settings["maxiter"], 200)

    @unittest.skipIf(not optionals.HAS_SKQUANT, "Install scikit-quant to run this test.")
    def test_imfil(self):
        """Test IMFIL is serializable."""

        opt = IMFIL(maxiter=200)
        settings = opt.settings

        self.assertEqual(settings["maxiter"], 200)

    def test_gradient_descent(self):
        """Test GradientDescent is serializable."""

        opt = GradientDescent(maxiter=10, learning_rate=0.01)
        settings = opt.settings

        self.assertEqual(settings["maxiter"], 10)
        self.assertEqual(settings["learning_rate"], 0.01)

    def test_gsls(self):
        """Test GSLS is serializable."""

        opt = GSLS(maxiter=100, sampling_radius=1e-3)
        settings = opt.settings

        self.assertEqual(settings["maxiter"], 100)
        self.assertEqual(settings["sampling_radius"], 1e-3)

    def test_spsa(self):
        """Test SPSA optimizer is serializable."""
        options = {
            "maxiter": 100,
            "blocking": True,
            "allowed_increase": 0.1,
            "second_order": True,
            "learning_rate": 0.02,
            "perturbation": 0.05,
            "regularization": 0.1,
            "resamplings": 2,
            "perturbation_dims": 5,
            "trust_region": False,
            "initial_hessian": None,
            "lse_solver": None,
            "hessian_delay": 0,
            "callback": None,
            "termination_checker": None,
        }
        spsa = SPSA(**options)

        self.assertDictEqual(spsa.settings, options)

    def test_spsa_custom_iterators(self):
        """Test serialization works with custom iterators for learning rate and perturbation."""
        rate = 0.99

        def powerlaw():
            n = 0
            while True:
                yield rate**n
                n += 1

        def steps():
            n = 1
            divide_after = 20
            epsilon = 0.5
            while True:
                yield epsilon
                n += 1
                if n % divide_after == 0:
                    epsilon /= 2

        learning_rate = powerlaw()
        expected_learning_rate = np.array([next(learning_rate) for _ in range(200)])

        perturbation = steps()
        expected_perturbation = np.array([next(perturbation) for _ in range(200)])

        spsa = SPSA(maxiter=200, learning_rate=powerlaw, perturbation=steps)
        settings = spsa.settings

        self.assertTrue(np.allclose(settings["learning_rate"], expected_learning_rate))
        self.assertTrue(np.allclose(settings["perturbation"], expected_perturbation))

    def test_qnspsa(self):
        """Test QN-SPSA optimizer is serializable."""
        ansatz = RealAmplitudes(1)
        fidelity = QNSPSA.get_fidelity(ansatz)
        options = {
            "fidelity": fidelity,
            "maxiter": 100,
            "blocking": True,
            "allowed_increase": 0.1,
            "learning_rate": 0.02,
            "perturbation": 0.05,
            "regularization": 0.1,
            "resamplings": 2,
            "perturbation_dims": 5,
            "lse_solver": None,
            "initial_hessian": None,
            "callback": None,
            "termination_checker": None,
            "hessian_delay": 0,
        }
        spsa = QNSPSA(**options)

        settings = spsa.settings
        expected = options.copy()

        with self.subTest(msg="check constructed dictionary"):
            self.assertDictEqual(settings, expected)

        reconstructed = QNSPSA(**settings)  # pylint: disable=unexpected-keyword-arg
        with self.subTest(msg="test reconstructed optimizer"):
            self.assertDictEqual(reconstructed.settings, expected)


if __name__ == "__main__":
    unittest.main()
