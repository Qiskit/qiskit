# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, unexpected-keyword-arg, no-value-for-parameter

"""Tests continuous pulse functions."""

import numpy as np

from qiskit.test import QiskitTestCase
import qiskit.pulse.pulses.continuous as continuous


class TestContinuousPulses(QiskitTestCase):
    """Test continuous pulses."""

    def test_constant(self):
        """Test constant pulse."""
        constant_arr = continuous.constant(np.linspace(0, 10, 50), amp=0.5j)
        np.testing.assert_equal(constant_arr, 0.5j)
        self.assertEqual(len(constant_arr), 50)

    def test_zero(self):
        """Test constant pulse."""
        zero_arr = continuous.zero(np.linspace(0, 10, 50))
        np.testing.assert_equal(zero_arr, 0.0)
        self.assertEqual(len(zero_arr), 50)

    def test_square(self):
        """Test square wave."""
        square_arr = continuous.square(np.linspace(0, 10, 100), amp=0.5, period=5)
        # with new phase
        square_arr_phased = continuous.square(np.linspace(0, 10, 100), amp=0.5,
                                              period=5, phase=np.pi/2)
        self.assertAlmostEqual(square_arr[0], 0.5)
        self.assertAlmostEqual(square_arr[25], -0.5)
        self.assertAlmostEqual(square_arr_phased[0], -0.5)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= square_arr) & (square_arr <= 0.5)))
        self.assertEqual(len(square_arr), 100)

    def test_sawtooth(self):
        """Test sawtooth wave."""
        sawtooth_arr = continuous.sawtooth(np.linspace(0, 10, 101), amp=0.5, period=5)
        # with new phase
        sawtooth_arr_phased = continuous.sawtooth(np.linspace(0, 10, 101), amp=0.5,
                                                  period=5, phase=np.pi/2)
        self.assertAlmostEqual(sawtooth_arr[0], 0.0)
        self.assertAlmostEqual(sawtooth_arr[24], 0.48)
        self.assertAlmostEqual(sawtooth_arr[50], 0.)
        self.assertAlmostEqual(sawtooth_arr[75], -0.5)
        self.assertAlmostEqual(sawtooth_arr_phased[0], -0.5)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= sawtooth_arr) & (sawtooth_arr <= 0.5)))
        self.assertEqual(len(sawtooth_arr), 101)

    def test_triangle(self):
        """Test triangle wave."""
        triangle_arr = continuous.triangle(np.linspace(0, 10, 101), amp=0.5, period=5)
        # with new phase
        triangle_arr_phased = continuous.triangle(np.linspace(0, 10, 101), amp=0.5,
                                                  period=5, phase=np.pi/2)
        self.assertAlmostEqual(triangle_arr[0], 0.0)
        self.assertAlmostEqual(triangle_arr[12], 0.48)
        self.assertAlmostEqual(triangle_arr[13], 0.48)
        self.assertAlmostEqual(triangle_arr[50], 0.)
        self.assertAlmostEqual(triangle_arr_phased[0], 0.5)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= triangle_arr) & (triangle_arr <= 0.5)))
        self.assertEqual(len(triangle_arr), 101)

    def test_cos(self):
        """Test cosine wave."""
        cos_arr = continuous.cos(np.linspace(0, 10, 101), amp=0.5, freq=1/5)
        # with new phase
        cos_arr_phased = continuous.cos(np.linspace(0, 10, 101), amp=0.5,
                                        freq=1/5, phase=np.pi/2)
        # Assert starts at 1
        self.assertAlmostEqual(cos_arr[0], 0.5)
        self.assertAlmostEqual(cos_arr[6], 0.3644, places=2)
        self.assertAlmostEqual(cos_arr[25], -0.5)
        self.assertAlmostEqual(cos_arr[50], 0.5)
        self.assertAlmostEqual(cos_arr_phased[0], 0.0)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= cos_arr) & (cos_arr <= 0.5)))
        self.assertEqual(len(cos_arr), 101)

    def test_sin(self):
        """Test sine wave."""
        sin_arr = continuous.sin(np.linspace(0, 10, 101), amp=0.5, freq=1/5)
        # with new phase
        sin_arr_phased = continuous.sin(np.linspace(0, 10, 101), amp=0.5,
                                        freq=1/5, phase=np.pi/2)
        # Assert starts at 1
        self.assertAlmostEqual(sin_arr[0], 0.0)
        self.assertAlmostEqual(sin_arr[6], 0.34227, places=2)
        self.assertAlmostEqual(sin_arr[25], 0.0)
        self.assertAlmostEqual(sin_arr[13], 0.5, places=2)
        self.assertAlmostEqual(sin_arr_phased[0], 0.5)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= sin_arr) & (sin_arr <= 0.5)))
        self.assertEqual(len(sin_arr), 101)
