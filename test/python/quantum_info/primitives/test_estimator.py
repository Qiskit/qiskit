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

"""Tests for ExpectationValue."""

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.primitives import EstimatorResult
from qiskit.quantum_info import Estimator, Operator, Statevector
from qiskit.test import QiskitTestCase


class TestEstimator(QiskitTestCase):
    """Test Estimator"""

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

    def test_init_from_statevector(self):
        """test initialization from statevector"""
        vector = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        statevector = Statevector(vector)
        with Estimator([statevector], [self.observable]) as est:
            self.assertIsInstance(est.circuits[0], QuantumCircuit)
            np.testing.assert_allclose(est.circuits[0][0][0].params, vector)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertAlmostEqual(result.values[0], -0.88272215)

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
        with Estimator([circuit], [matrix]) as est:
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertAlmostEqual(result.values[0], -1.284366511861733)

    def test_evaluate(self):
        """test for evaluate"""
        with Estimator([self.ansatz], [self.observable]) as est:
            result = est(parameters=[0, 1, 1, 2, 3, 5])
        self.assertIsInstance(result, EstimatorResult)
        self.assertAlmostEqual(result.values[0], -1.284366511861733)

    def test_evaluate_multi_params(self):
        """test for evaluate with multiple parameters"""
        with Estimator([self.ansatz], [self.observable]) as est:
            result = est(parameters=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733, -1.3187526349078742])

    def test_evaluate_no_params(self):
        """test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        with Estimator([circuit], [self.observable]) as est:
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertAlmostEqual(result.values[0], -1.284366511861733)

    def test_run_with_multiple_observables_and_none_parameters(self):
        """test for evaluate without parameters"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with Estimator(circuit, ["ZZZ", "III"]) as est:
            result = est(circuits=[0, 0], observables=[0, 1])
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [0.0, 1.0])
