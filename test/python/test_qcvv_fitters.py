# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test qiskit.tools.qcvv.fitters."""

import numpy as np

from qiskit.tools.qcvv import fitters
from qiskit.test import QiskitTestCase


class TestQCVVFitters(QiskitTestCase):
    """Tests for functions in qiskit.tools.qcvv.fitters."""

    def test_exp_fit_fun(self):
        """Test exponential decay fit function."""
        a_in = 2
        x_in = 3
        tau_in = 4
        c_in = 1
        result = fitters.exp_fit_fun(x_in, a_in, tau_in, c_in)
        self.assertAlmostEqual(1.9447331054820294, result)

    def test_osc_fit_fun(self):
        """Test decay cosine fit function."""
        a_in = 2
        x_in = 3
        tau_in = 4
        c_in = 1
        f_in = 5
        phi_in = 90
        result = fitters.osc_fit_fun(x_in, a_in, tau_in, f_in, phi_in, c_in)
        self.assertAlmostEqual(0.5766900211497354, result)

    def test_rb_fit_fun(self):
        """Test randomized benchmark fit function."""
        a_in = 2
        x_in = 3
        alpha_in = 4
        b_in = 1
        result = fitters.rb_fit_fun(x_in, a_in, alpha_in, b_in)
        self.assertAlmostEqual(129, result)

    def test_shape_rb_data(self):
        """Test randomized benchmark data shaping function."""
        raw_data = np.zeros((2, 2, 2))
        # TODO: Come up with a more realistic input data set instead
        # of this synthetic example
        for i in range(2):
            raw_data[i][i][i] = i + 1
        result = fitters.shape_rb_data(raw_data)
        expected = [np.array([[0.5, 0.0],
                              [0.0, 1.0]]),
                    np.array([[0.5, 0.0],
                              [0.0, 1.0]])]
        for i, expec in enumerate(expected):
            self.assertEqual(expec.all(), result[i].all())

    def test_rb_epc(self):
        """Test error per clifford for randomized benchmark fit data."""
        rb_pattern = [[1], [0, 2]]
        fit = {'q2': {'fit': [2, 0.5, 0.8], 'fiterr': [0, 0, 0]},
               'q1': {'fit': [1, 0.2, 0.9], 'fiterr': [0, 0, 0]},
               'q0': {'fit': [1, 0.3, 0.7], 'fiterr': [0, 0, 0]}}
        result = fitters.rb_epc(fit, rb_pattern)
        expected = {
            'q2': {
                'fit': [2, 0.5, 0.8],
                'fiterr': [0, 0, 0],
                'fit_calcs': {'epc': [0.375, 0.0]}},
            'q1': {
                'fit': [1, 0.2, 0.9],
                'fiterr': [0, 0, 0],
                'fit_calcs': {'epc': [0.4, 0.0]}},
            'q0': {
                'fit': [1, 0.3, 0.7],
                'fiterr': [0, 0, 0],
                'fit_calcs': {'epc': [0.5249999999999999, 0.0]}}
        }

        for bit in fit:
            for list_type in ['fit', 'fiterr']:
                self.assertTrue(np.allclose(expected[bit][list_type],
                                            result[bit][list_type]))
            self.assertTrue(np.allclose(expected[bit]['fit_calcs']['epc'],
                                        result[bit]['fit_calcs']['epc']))
