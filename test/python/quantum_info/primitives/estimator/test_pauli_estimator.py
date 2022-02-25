# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PauliExpectationValue."""

import unittest

import numpy as np

from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Operator, Statevector
from qiskit.quantum_info.primitives import PauliEstimator
from qiskit.quantum_info.primitives.results import EstimatorResult
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals

if optionals.HAS_AER:
    from qiskit import Aer


class TestPauliEstimator(QiskitTestCase):
    """Test PauliEstimator"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
    def test_init_from_statevector(self):
        """test initialization from statevector"""
        vector = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        statevector = Statevector(vector)
        backend = Aer.get_backend("aer_simulator")
        with PauliEstimator([statevector], [self.observable], backend=backend) as est:
            est.set_run_options(seed_simulator=15)
            self.assertIsInstance(est.circuits[0], QuantumCircuit)
            np.testing.assert_allclose(est.circuits[0][0][0].params, vector)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -0.88272215)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 0.316504211)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
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
        with PauliEstimator(
            [circuit], [matrix], backend=BasicAer.get_backend("qasm_simulator")
        ) as est:
            est.set_run_options(seed_simulator=15)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.3086290180956408)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 0.2992945988106311)

    def test_evaluate_basicaer(self):
        """test for evaluate with BasicAer"""
        backend = BasicAer.get_backend("qasm_simulator")
        with PauliEstimator([self.ansatz], [self.observable], backend=backend) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([0, 1, 1, 2, 3, 5])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.3086290180956408)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 0.2992945988106311)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
    def test_evaluate(self):
        """test for evaluate with Aer"""
        backend = Aer.get_backend("aer_simulator")
        with PauliEstimator([self.ansatz], [self.observable], backend=backend) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([0, 1, 1, 2, 3, 5])
            self.assertEqual(len(est.transpiled_circuits), 2)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.3138315875089022)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 0.29089871458670)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
    def test_evaluate_without_grouping(self):
        """test for evaluate without grouping"""
        backend = Aer.get_backend("aer_simulator")
        with PauliEstimator(
            [self.ansatz], [self.observable], backend=backend, strategy=False
        ) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([0, 1, 1, 2, 3, 5])
            self.assertEqual(len(est.transpiled_circuits), 5)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.302795595970)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 0.29068761565451)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
    def test_evaluate_multi_params(self):
        """test for evaluate with multiple parameters"""
        backend = Aer.get_backend("aer_simulator")
        with PauliEstimator([self.ansatz], [self.observable], backend=backend) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]])
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [-1.313832, -1.306558], rtol=1e-05)
        self.assertEqual(result.variances.dtype, np.float64)
        np.testing.assert_allclose(result.variances, [0.290899, 0.24758], rtol=1e-05)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
    def test_evaluate_no_params(self):
        """test for evaluate without parameters"""
        backend = Aer.get_backend("aer_simulator")
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        with PauliEstimator([circuit], [self.observable], backend=backend) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.3138315875089022)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 0.2908987145867)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer doesn't appear to be installed.")
    def test_run_with_multiple_observables_and_none_parameters(self):
        """test for evaluate without parameters"""
        backend = Aer.get_backend("aer_simulator")
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with PauliEstimator(circuit, ["ZZZ", "III"], backend=backend) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [0.00390625, 1.0])
        self.assertEqual(result.variances.dtype, np.float64)
        print(result.variances[0])
        np.testing.assert_allclose(result.variances, [0.999984712, 0.0])
