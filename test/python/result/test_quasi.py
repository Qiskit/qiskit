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
import unittest
from math import sqrt
from qiskit.result import QuasiDistribution


class TestQuasi(unittest.TestCase):
    """Tests for quasidistributions."""

    def test_hex_quasi(self):
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_bin_quasi(self):
        qprobs = {"0b0": 3 / 5, "0b1": 1 / 2, "0b10": 7 / 20, "0b11": 1 / 10, "0b100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_bin_no_prefix_quasi(self):
        qprobs = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_hex_quasi_hex_out(self):
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual(qprobs, quasi.hex_probabilities())

    def test_bin_quasi_hex_out(self):
        qprobs = {"0b0": 3 / 5, "0b1": 1 / 2, "0b10": 7 / 20, "0b11": 1 / 10, "0b100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        self.assertEqual(expected, quasi.hex_probabilities())

    def test_bin_no_prefix_quasi_hex_out(self):
        qprobs = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        self.assertEqual(expected, quasi.hex_probabilities())

    def test_hex_quasi_bin_out(self):
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        self.assertEqual(expected, quasi.binary_probabilities())

    def test_bin_quasi_bin_out(self):
        qprobs = {"0b0": 3 / 5, "0b1": 1 / 2, "0b10": 7 / 20, "0b11": 1 / 10, "0b100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        self.assertEqual(expected, quasi.binary_probabilities())

    def test_bin_no_prefix_quasi_bin_out(self):
        qprobs = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual(qprobs, quasi.binary_probabilities())

    def test_empty(self):
        quasi = QuasiDistribution({})
        self.assertEqual(quasi, {})

    def test_empty_hex_out(self):
        quasi = QuasiDistribution({})
        self.assertEqual(quasi.hex_probabilities(), {})

    def test_empty_bin_out(self):
        quasi = QuasiDistribution({})
        self.assertEqual(quasi.binary_probabilities(), {})

    def test_invalid_keys(self):
        with self.assertRaises(TypeError):
            QuasiDistribution({1 + 2j: 3 / 5})

    def test_invalid_key_string(self):
        with self.assertRaises(ValueError):
            QuasiDistribution({"1a2b": 3 / 5})

    def test_known_quasi_conversion(self):
        """Reproduce conversion from Smolin PRL"""
        qprobs = {0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}
        closest, dist = QuasiDistribution(qprobs).nearest_probability_distribution(
            return_distance=True
        )
        ans = {0: 9 / 20, 1: 7 / 20, 2: 1 / 5}
        # Check probs are correct
        for key, val in closest.items():
            assert abs(ans[key] - val) < 1e-14
        # Check if distance calculation is correct
        assert abs(dist - sqrt(0.38)) < 1e-14
