# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Estimator."""

import unittest

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.primitives import (
    Estimator,
    Sampler,
    allow_broadcasting,
    allow_objects,
    allow_optional,
)
from qiskit.test import QiskitTestCase


class TestAllowOptional(QiskitTestCase):
    """Test allow_optional."""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.ansatz_with_meas = self.ansatz.copy()
        self.ansatz_with_meas.measure_all()
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.0284380963435145, -1.284366511861733
        self.quasi_dists = (
            {
                0: 0.17158451004815306,
                1: 0.0041370682135240654,
                2: 0.20402129418492707,
                3: 0.6202571275533961,
            },
            {
                0: 0.1309248462975777,
                1: 0.3608720796028448,
                2: 0.09324865232050054,
                3: 0.41495442177907715,
            },
        )

    def test_allow_optional_sampler(self):
        """Test allow optional decorator for Sampler."""

        # pylint: disable=missing-class-docstring
        @allow_optional
        class CustomSampler(Sampler):
            ...

        with CustomSampler([self.ansatz_with_meas]) as sampler:
            result = sampler(parameter_values=[list(range(6))])
        self.assertDictAlmostEqual(result.quasi_dists[0], self.quasi_dists[0])

    def test_allow_optional_estimator(self):
        """Test allow optional decorator for Estimator."""

        # pylint: disable=missing-class-docstring
        @allow_optional
        class CustomEstimator(Estimator):
            ...

        with CustomEstimator([self.ansatz], [self.observable]) as estimator:
            result = estimator(parameter_values=[list(range(6))])

        self.assertEqual(result.values[0], self.expvals[0])

    def test_allow_broadcasting_sampler(self):
        """Test allow broadcasting decorator for Sampler."""

        # pylint: disable=missing-class-docstring
        @allow_broadcasting()
        class CustomSampler(Sampler):
            ...

        with CustomSampler([self.ansatz_with_meas]) as sampler:
            result = sampler(parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]])
        self.assertDictAlmostEqual(result.quasi_dists[0], self.quasi_dists[0])
        self.assertDictAlmostEqual(result.quasi_dists[1], self.quasi_dists[1])

    def test_allow_broadcasting_estimator(self):
        """Test allow broadcasting decorator for Estimator."""

        # pylint: disable=missing-class-docstring
        @allow_broadcasting()
        class CustomEstimator(Estimator):
            ...

        with CustomEstimator([self.ansatz], [self.observable]) as estimator:
            result = estimator(parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]])
        self.assertEqual(result.values[0], self.expvals[0])
        self.assertEqual(result.values[1], self.expvals[1])

    def test_allow_objects_sampler(self):
        """Test allow objects decorator for Sampler."""

        # pylint: disable=missing-class-docstring
        @allow_objects
        class CustomSampler(Sampler):
            ...

        with CustomSampler([self.ansatz_with_meas]) as sampler:
            result = sampler(
                circuit_indices=[self.ansatz_with_meas], parameter_values=[list(range(6))]
            )
        self.assertDictAlmostEqual(result.quasi_dists[0], self.quasi_dists[0])

    def test_allow_objects_estimator(self):
        """Test allow object decorator for Estimator."""

        # pylint: disable=missing-class-docstring
        @allow_objects
        class CustomEstimator(Estimator):
            ...

        with CustomEstimator([self.ansatz], [self.observable]) as estimator:
            result = estimator(
                circuit_indices=[self.ansatz, self.ansatz],
                observable_indices=[self.observable, self.observable],
                parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
            )
        self.assertEqual(result.values[0], self.expvals[0])
        self.assertEqual(result.values[1], self.expvals[1])


if __name__ == "__main__":
    unittest.main()
