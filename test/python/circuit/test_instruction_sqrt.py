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

from qiskit.extensions import UnitaryGate, SGate
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumRegister, QuantumCircuit


class TestGateSqrt(QiskitTestCase):
    """Test Gate.sqrt()"""

    def test_unitary(self):
        """Test standard UnitaryGate.sqrt() method.
        """
        expected = numpy.array([[0.5+0.5j,  0.5+0.5j],
                                [-0.5-0.5j,  0.5+0.5j]])
        result = UnitaryGate([[0, 1j], [-1j, 0]]).sqrt()
        self.assertTrue(numpy.allclose(result.to_matrix(), expected))


if __name__ == '__main__':
    unittest.main()
