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

from io import BytesIO
from PIL import Image

from qiskit.tools.visualization import plot_histogram, HAS_MATPLOTLIB

# from .visualization import path_to_diagram_reference
from .visualization import QiskitVisualizationTestCase


if HAS_MATPLOTLIB:
    import matplotlib as mpl
    import matplotlib.pyplot as plt


class TestPlotHistogram(QiskitVisualizationTestCase):
    """Qiskit plot_histogram tests."""

    @unittest.skipIf(not HAS_MATPLOTLIB, "matplotlib not available.")
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

    @unittest.skipIf(not HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_histogram_bars_align(self):
        """Test issue #6692"""
        data_noisy = {
            "00000": 0.22,
            "00001": 0.003,
            "00010": 0.005,
            "00100": 0.004,
            "00101": 0.001,
            "00110": 0.004,
            "00111": 0.001,
            "01000": 0.005,
            "01010": 0.002,
            "01100": 0.225,
            "01101": 0.001,
            "01110": 0.003,
            "01111": 0.003,
            "10000": 0.012,
            "10001": 0.002,
            "10010": 0.001,
            "10011": 0.001,
            "10100": 0.247,
            "10101": 0.004,
            "10110": 0.003,
            "10111": 0.001,
            "11000": 0.225,
            "11001": 0.005,
            "11010": 0.002,
            "11100": 0.015,
            "11101": 0.004,
            "11110": 0.001,
        }
        data_ideal = {
            "00000": 0.25,
            "01100": 0.25,
            "10100": 0.25,
            "11000": 0.25,
        }
        data_ideal_reduced = {
            "00000": 0.25,
            "01100": 0.25,
            "10100": 0.25,
            "11000": 0.25,
            "rest": 0,
        }
        data_noisy_reduced = {
            "00000": 0.22,
            "01100": 0.225,
            "10100": 0.247,
            "11000": 0.225,
            "11100": 0.015,
            "rest": 0.083,
        }
        fig_reduced = plot_histogram([data_ideal, data_noisy], number_to_keep=5)
        fig_manual_reduced = plot_histogram([data_ideal_reduced, data_noisy_reduced])
        self.assertIsInstance(fig_reduced, mpl.figure.Figure)

        # Check images nearly match (ordering of bars is different)
        # img_ref = path_to_diagram_reference("plot_histogram_reduced_states.png")
        # img_manual_ref = path_to_diagram_reference("plot_histogram_reduced_states_manual.png")
        with BytesIO() as img_buffer:
            fig_reduced.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            with BytesIO() as img_manual_buffer:
                fig_manual_reduced.savefig(img_manual_buffer, format="png")
                img_manual_buffer.seek(0)
                self.assertImagesAreEqual(
                    Image.open(img_buffer), Image.open(img_manual_buffer), 0.01
                )
                # self.assertImagesAreEqual(Image.open(img_manual_buffer), img_manual_ref, 0.01)
            # self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.01)
        plt.close(fig_reduced)
        plt.close(fig_manual_reduced)

    @unittest.skipIf(not HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_histogram_number_to_keep(self):
        """Test that histograms using number_to_keep produce outputs."""
        data_ideal = {
            "000": 0.25,
            "110": 0.25,
            "011": 0.25,
            "101": 0.25,
        }
        data_noisy = {
            "000": 0.24,
            "110": 0.25,
            "011": 0.24,
            "101": 0.24,
            "001": 0.03,
        }
        fig_few_items = plot_histogram([data_ideal, data_noisy], number_to_keep=4)
        self.assertIsInstance(fig_few_items, mpl.figure.Figure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
