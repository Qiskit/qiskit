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

# pylint: disable=invalid-name

"""Tests for PauliTable utility functions."""

import unittest

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.symplectic import PauliTable, pauli_basis


class TestPauliUtils(QiskitTestCase):
    """Test pauli utils"""

    def test_pauli_basis(self):
        """Test pauli_basis function."""

        with self.subTest(msg='1 qubit standard order'):
            labels = ['I', 'X', 'Y', 'Z']
            target = PauliTable.from_labels(labels)
            self.assertEqual(pauli_basis(1), target)

        with self.subTest(msg='1 qubit weight order'):
            labels = ['I', 'X', 'Y', 'Z']
            target = PauliTable.from_labels(labels)
            self.assertEqual(pauli_basis(1, weight=True), target)

        with self.subTest(msg='2 qubit standard order'):
            labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
                      'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
            target = PauliTable.from_labels(labels)
            self.assertEqual(pauli_basis(2), target)

        with self.subTest(msg='2 qubit weight order'):
            labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'YI', 'ZI',
                      'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
            target = PauliTable.from_labels(labels)
            self.assertEqual(pauli_basis(2, weight=True), target)

        with self.subTest(msg='3 qubit standard order'):
            labels = ['III', 'IIX', 'IIY', 'IIZ', 'IXI', 'IXX', 'IXY', 'IXZ',
                      'IYI', 'IYX', 'IYY', 'IYZ', 'IZI', 'IZX', 'IZY', 'IZZ',
                      'XII', 'XIX', 'XIY', 'XIZ', 'XXI', 'XXX', 'XXY', 'XXZ',
                      'XYI', 'XYX', 'XYY', 'XYZ', 'XZI', 'XZX', 'XZY', 'XZZ',
                      'YII', 'YIX', 'YIY', 'YIZ', 'YXI', 'YXX', 'YXY', 'YXZ',
                      'YYI', 'YYX', 'YYY', 'YYZ', 'YZI', 'YZX', 'YZY', 'YZZ',
                      'ZII', 'ZIX', 'ZIY', 'ZIZ', 'ZXI', 'ZXX', 'ZXY', 'ZXZ',
                      'ZYI', 'ZYX', 'ZYY', 'ZYZ', 'ZZI', 'ZZX', 'ZZY', 'ZZZ']
            target = PauliTable.from_labels(labels)
            self.assertEqual(pauli_basis(3), target)

        with self.subTest(msg='3 qubit weight order'):
            labels = ['III', 'IIX', 'IIY', 'IIZ', 'IXI', 'IYI', 'IZI', 'XII', 'YII', 'ZII',
                      'IXX', 'IXY', 'IXZ', 'IYX', 'IYY', 'IYZ', 'IZX', 'IZY', 'IZZ',
                      'XIX', 'XIY', 'XIZ', 'XXI', 'XYI', 'XZI',
                      'YIX', 'YIY', 'YIZ', 'YXI', 'YYI', 'YZI',
                      'ZIX', 'ZIY', 'ZIZ', 'ZXI', 'ZYI', 'ZZI',
                      'XXX', 'XXY', 'XXZ', 'XYX', 'XYY', 'XYZ', 'XZX', 'XZY', 'XZZ',
                      'YXX', 'YXY', 'YXZ', 'YYX', 'YYY', 'YYZ', 'YZX', 'YZY', 'YZZ',
                      'ZXX', 'ZXY', 'ZXZ', 'ZYX', 'ZYY', 'ZYZ', 'ZZX', 'ZZY', 'ZZZ']
            target = PauliTable.from_labels(labels)
            self.assertEqual(pauli_basis(3, weight=True), target)


if __name__ == '__main__':
    unittest.main()
