# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A test for circuit tools"""
import unittest

from test import combine
from ddt import ddt
from qiskit.test import QiskitTestCase
from qiskit.circuit.tools.pi_check import pi_check


@ddt
class TestPiCheck(QiskitTestCase):
    """ qiskit/visualization/tools/pi_check.py """

    @combine(case=[(3.14, '3.14'),
                   (3.141592653589793, 'pi'),
                   (6.283185307179586, '2pi'),
                   (2.99, '2.99'),
                   (2.999999999999999, '3'),
                   (0.99, '0.99'),
                   (0.999999999999999, '1')])
    def test_default(self, case):
        """Default pi_check({case[0]})='{case[1]}'"""
        input_number = case[0]
        expected_string = case[1]
        result = pi_check(input_number)
        self.assertEqual(result, expected_string)


if __name__ == '__main__':
    unittest.main(verbosity=2)
