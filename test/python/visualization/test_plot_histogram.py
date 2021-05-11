# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for plot_histogram."""

import unittest
import matplotlib as mpl

from qiskit.test import QiskitTestCase
from qiskit.tools.visualization import plot_histogram


class TestPlotHistogram(QiskitTestCase):
    """Qiskit plot_histogram tests."""

    def test_different_counts_lengths(self):
        """Test plotting two different length dists works"""
        exact_dist = {
            "000000": 0.015624999999999986,
            "000001": 0.015624999999999986,
            "000011": 0.031249999999999965,
            "000111": 0.06249999999999992,
            "100000": 0.015624999999999986,
            "100001": 0.015624999999999986,
            "100011": 0.031249999999999965,
            "100111": 0.06249999999999992,
            "110000": 0.031249999999999965,
            "110001": 0.031249999999999965,
            "110011": 0.06249999999999992,
            "110111": 0.12499999999999982,
            "111111": 0.4999999999999991,
        }

        raw_dist = {
            "000000": 26,
            "000001": 29,
            "010000": 10,
            "010001": 12,
            "010010": 6,
            "010011": 14,
            "010100": 2,
            "010101": 6,
            "010110": 4,
            "010111": 24,
            "011000": 2,
            "011001": 5,
            "011011": 5,
            "011101": 4,
            "011110": 7,
            "011111": 77,
            "000010": 9,
            "100000": 31,
            "100001": 25,
            "100010": 8,
            "100011": 46,
            "100100": 3,
            "100101": 3,
            "100110": 9,
            "100111": 114,
            "101000": 3,
            "101001": 6,
            "101010": 1,
            "101011": 6,
            "101100": 7,
            "101101": 9,
            "101110": 6,
            "101111": 48,
            "000011": 82,
            "110000": 42,
            "110001": 53,
            "110010": 9,
            "110011": 102,
            "110100": 10,
            "110101": 8,
            "110110": 14,
            "110111": 215,
            "111000": 25,
            "111001": 12,
            "111010": 2,
            "111011": 41,
            "111100": 18,
            "111101": 24,
            "111110": 58,
            "111111": 621,
            "000100": 1,
            "000101": 7,
            "000110": 9,
            "000111": 73,
            "001000": 1,
            "001001": 5,
            "001011": 6,
            "001100": 1,
            "001101": 7,
            "001110": 1,
            "001111": 34,
        }

        fig = plot_histogram([raw_dist, exact_dist])
        self.assertIsInstance(fig, mpl.figure.Figure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
