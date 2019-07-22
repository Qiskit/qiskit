# -*- coding: utf-8 -*-

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


"""Test Qiskit's sqrt instruction operation."""

import unittest
import numpy
from numpy.testing import assert_allclose
from numpy.linalg import matrix_power

from qiskit.extensions import UnitaryGate, SGate, YGate
from qiskit.test import QiskitTestCase

ATOL = 1e-8
RTOL = 1e-7

class TestGateSqrt(QiskitTestCase):
    """Test Gate.sqrt()"""

    def test_pauliy_roots(self):
        """
        Test whether roots of pauli-Y are found.

        We test a range of roots, for 1/x where x \in [1,10]
        """
        for x in range(1, 10):
            result = YGate().pow(1/x)
            self.assertEqual(result.name, 'unitary')
            assert_allclose(matrix_power(result.to_matrix(), x), YGate().to_matrix(), rtol=RTOL, atol=ATOL)

    def test_inverse_sgate(self):
        """
        Test whether inverse of Sgate is found
        """
        result = SGate().pow(-1)
        self.assertEqual(result.name, 'unitary')
        assert_allclose(result.to_matrix() @ SGate().to_matrix(), numpy.identity(2), rtol=RTOL, atol=ATOL)


if __name__ == '__main__':
    unittest.main()
