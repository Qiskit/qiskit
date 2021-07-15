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

"""Test conversion to probability distribution"""
from math import sqrt

from qiskit.test import QiskitTestCase
from qiskit.result import QuasiDistribution


class TestQuasi(QiskitTestCase):
    """Tests for quasidistributions."""

    def test_known_quasi_conversion(self):
        """Reproduce conversion from Smolin PRL"""
        qprobs = {"000": 3 / 5, "001": 1 / 2, "010": 7 / 20, "011": 1 / 10, "100": -11 / 20}
        closest, dist = QuasiDistribution(qprobs).nearest_probability_distribution(
            return_distance=True
        )
        ans = {"000": 9 / 20, "001": 7 / 20, "010": 1 / 5}
        # Check probs are correct
        for key, val in closest.items():
            self.assertTrue(abs(ans[key] - val) < 1e-14)
        # Check if distance calculation is correct
        self.assertTrue(abs(dist - sqrt(0.38)) < 1e-14)
