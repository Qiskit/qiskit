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

from qiskit.result import QuasiDistribution
from qiskit.test import QiskitTestCase


class TestQuasiDistribution(QiskitTestCase):
    """Test QuasiDistribution class."""

    def test_known_quasi_conversion(self):
        """Reproduce conversion from Smolin PRL"""
        qprobs = {"0": 3 / 5, "1": 1 / 2, "2": 7 / 20, "3": 1 / 10, "4": -11 / 20}
        closest, dist = QuasiDistribution(
            qprobs).nearest_probability_distribution(return_distance=True)
        ans = {"0": 9 / 20, "1": 7 / 20, "2": 1 / 5}
        # Check probs are correct
        for key, val in closest.items():
            self.assertLess(abs(ans[key] - val), 1e-14)
        # Check if distance calculation is correct
        self.assertLess(abs(dist - sqrt(0.38)), 1e-14)
