# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the phase oracle circuits."""

import unittest
from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit.circuit.library import PhaseOracle


@ddt
class TestPhaseOracle(QiskitTestCase):
    """Test phase oracle object."""

    @data(('x | x', '1', True),
          ('x & x', '0', False),
          ('(x0 & x1 | ~x2) ^ x4', '0110', False),
          ('xx & xxx | ( ~z ^ zz)', '0111', True))
    @unpack
    def test_evaluate_bitstring(self, expression, input_bitstring, expected):
        oracle = PhaseOracle(expression)
        result = oracle.evaluate_bitstring(input_bitstring)
        self.assertEqual(result, expected)

    # @data(('x | x', '1', True),
    #       ('x & x', '0', False),
    #       ('(x0 & x1 | ~x2) ^ x4', '0110', False),
    #       ('xx & xxx | ( ~z ^ zz)', '0111', True))
    # @unpack
    # def test_evaluate_bitstring(self, expression, input_bitstring, expected):
    #     initial_value.
    #     oracle = PhaseOracle(expression)
    #     result = oracle.evaluate_bitstring(input_bitstring)
    #     self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
