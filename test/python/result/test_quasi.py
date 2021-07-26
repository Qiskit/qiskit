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

    def test_hex_quasi(self):
        """Test hexadecimal input."""
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_bin_quasi(self):
        """Test binary input."""
        qprobs = {"0b0": 3 / 5, "0b1": 1 / 2, "0b10": 7 / 20, "0b11": 1 / 10, "0b100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_bin_quasi_no_0b(self):
        """Test binary input without 0b in front."""
        qprobs = {"000": 3 / 5, "001": 1 / 2, "010": 7 / 20, "011": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_bin_no_prefix_quasi(self):
        """Test binary input without 0b prefix."""
        qprobs = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual({0: 3 / 5, 1: 1 / 2, 2: 7 / 20, 3: 1 / 10, 4: -11 / 20}, quasi)

    def test_hex_quasi_hex_out(self):
        """Test hexadecimal input and hexadecimal output."""
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual(qprobs, quasi.hex_probabilities())

    def test_bin_quasi_hex_out(self):
        """Test binary input and hexadecimal output."""
        qprobs = {"0b0": 3 / 5, "0b1": 1 / 2, "0b10": 7 / 20, "0b11": 1 / 10, "0b100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        self.assertEqual(expected, quasi.hex_probabilities())

    def test_bin_no_prefix_quasi_hex_out(self):
        """Test binary input without a 0b prefix and hexadecimal output."""
        qprobs = {"0": 3 / 5, "1": 1 / 2, "10": 7 / 20, "11": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        self.assertEqual(expected, quasi.hex_probabilities())

    def test_hex_quasi_bin_out(self):
        """Test hexadecimal input and binary output."""
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"000": 3 / 5, "001": 1 / 2, "010": 7 / 20, "011": 1 / 10, "100": -11 / 20}
        self.assertEqual(expected, quasi.binary_probabilities())

    def test_bin_quasi_bin_out(self):
        """Test binary input and binary output."""
        qprobs = {"0b0": 3 / 5, "0b1": 1 / 2, "0b10": 7 / 20, "0b11": 1 / 10, "0b100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"000": 3 / 5, "001": 1 / 2, "010": 7 / 20, "011": 1 / 10, "100": -11 / 20}
        self.assertEqual(expected, quasi.binary_probabilities())

    def test_bin_no_prefix_quasi_bin_out(self):
        """Test binary input without a 0b prefix and binary output."""
        qprobs = {"000": 3 / 5, "001": 1 / 2, "010": 7 / 20, "011": 1 / 10, "100": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        self.assertEqual(qprobs, quasi.binary_probabilities())

    def test_hex_quasi_bin_out_padded(self):
        """Test hexadecimal input and binary output, padded with zeros."""
        qprobs = {"0x0": 3 / 5, "0x1": 1 / 2, "0x2": 7 / 20, "0x3": 1 / 10, "0x4": -11 / 20}
        quasi = QuasiDistribution(qprobs)
        expected = {"0000": 3 / 5, "0001": 1 / 2, "0010": 7 / 20, "0011": 1 / 10, "0100": -11 / 20}
        self.assertEqual(expected, quasi.binary_probabilities(num_bits=4))

    def test_empty(self):
        """Test empty input."""
        quasi = QuasiDistribution({})
        self.assertEqual(quasi, {})

    def test_empty_hex_out(self):
        """Test empty input with hexadecimal output."""
        quasi = QuasiDistribution({})
        self.assertEqual(quasi.hex_probabilities(), {})

    def test_empty_bin_out(self):
        """Test empty input with binary output."""
        quasi = QuasiDistribution({})
        self.assertEqual(quasi.binary_probabilities(), {})

    def test_invalid_keys(self):
        """Test invalid key type raises."""
        with self.assertRaises(TypeError):
            QuasiDistribution({1 + 2j: 3 / 5})

    def test_invalid_key_string(self):
        """Test invalid key string format raises."""
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
