# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Test delay instruction for quantum circuits."""

import numpy as np
from qiskit.circuit import Delay
from qiskit.test.base import QiskitTestCase


class TestDelayClass(QiskitTestCase):
    """Test delay instruction for quantum circuits."""

    def test_to_matrix_return_identity_matrix(self):
        actual = Delay(1, 100).to_matrix()
        expected = np.array([[1, 0],
                             [0, 1]], dtype=complex)
        self.assertTrue(np.array_equal(actual, expected))
