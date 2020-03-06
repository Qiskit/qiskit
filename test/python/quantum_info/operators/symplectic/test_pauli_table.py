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

"""Tests for PauliTable class."""

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from qiskit import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.symplectic import PauliTable


class TestPauliTable(QiskitTestCase):
    """Tests for PauliTable class."""

    def pauli_mat(self, label):
        """Return Pauli matrix from a Pauli label"""
        mat = np.eye(1, dtype=complex)
        for i in label:
            if i == 'I':
                mat = np.kron(mat, np.eye(2, dtype=complex))
            elif i == 'X':
                mat = np.kron(mat, np.array([[0, 1], [1, 0]], dtype=complex))
            elif i == 'Y':
                mat = np.kron(mat, np.array([[0, -1j], [1j, 0]], dtype=complex))
            elif i == 'Z':
                mat = np.kron(mat, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                raise QiskitError('Invalid Pauli string {}'.format(i))
        return mat

    def test_init(self):
        """Test initialization."""
        # Matrix array initialization
        with self.subTest(msg='bool array'):
            target = np.array([[False, False], [True, True]])
            value = PauliTable(target)._array
            self.assertTrue(np.all(value == target))

        with self.subTest(msg='bool array no copy'):
            target = np.array([[False, True], [True, True]])
            value = PauliTable(target)._array
            value[0, 0] = not value[0, 0]
            self.assertTrue(np.all(value == target))

        with self.subTest(msg='bool array raises'):
            array = np.array([[False, False, False],
                              [True, True, True]])
            self.assertRaises(QiskitError, PauliTable, array)

        # Vector array initialization
        with self.subTest(msg='bool vector'):
            target = np.array([False, False, False, False])
            value = PauliTable(target)._array
            self.assertTrue(np.all(value == target))

        with self.subTest(msg='bool vector no copy'):
            target = np.array([False, True, True, False])
            value = PauliTable(target)._array
            value[0, 0] = not value[0, 0]
            self.assertTrue(np.all(value == target))

        # String initialization
        with self.subTest(msg='str init "I"'):
            value = PauliTable('I')._array
            target = np.array([[False, False]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "X"'):
            value = PauliTable('X')._array
            target = np.array([[True, False]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "Y"'):
            value = PauliTable('Y')._array
            target = np.array([[True, True]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "Z"'):
            value = PauliTable('Z')._array
            target = np.array([[False, True]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "I"'):
            value = PauliTable('I')._array
            target = np.array([[False, False]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "IX"'):
            value = PauliTable('IX')._array
            target = np.array([[True, False, False, False]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "XI"'):
            value = PauliTable('XI')._array
            target = np.array([[False, True, False, False]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "YZ"'):
            value = PauliTable('YZ')._array
            target = np.array([[False, True, True, True]], dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "XIZ"'):
            value = PauliTable('XIZ')._array
            target = np.array([[False, False, True, True, False, False]],
                              dtype=np.bool)
            self.assertTrue(np.all(np.array(value == target)))

        # Pauli Table initialization
        with self.subTest(msg='PauliTable'):
            target = PauliTable.from_labels(['XI', 'IX', 'IZ'])
            value = PauliTable(target)
            self.assertEqual(value, target)

        with self.subTest(msg='PauliTable no copy'):
            target = PauliTable.from_labels(['XI', 'IX', 'IZ'])
            value = PauliTable(target)
            value[0] = 'II'
            self.assertEqual(value, target)

    def test_properties(self):
        """Test class property methods"""

        with self.subTest(msg='array'):
            pauli = PauliTable('II')
            array = np.zeros([2, 4], dtype=np.bool)
            self.assertTrue(np.all(pauli.array == array))

        with self.subTest(msg='set array'):
            pauli = PauliTable('XX')
            array = np.zeros([1, 4], dtype=np.bool)
            pauli.array = array
            self.assertTrue(np.all(pauli.array == array))

        with self.subTest(msg='set array raises'):

            def set_array_raise():
                pauli = PauliTable('XXX')
                pauli.array = np.eye(4)
                return pauli

            self.assertRaises(ValueError, set_array_raise)

        with self.subTest(msg='X'):
            pauli = PauliTable.from_labels(['XI', 'IZ', 'YY'])
            array = np.array([[False, True], [False, False], [True, True]],
                             dtype=np.bool)
            self.assertTrue(np.all(pauli.X == array))

        with self.subTest(msg='set X'):
            pauli = PauliTable.from_labels(['XI', 'IZ'])
            val = np.array([[False, False], [True, True]], dtype=np.bool)
            pauli.X = val
            self.assertEqual(pauli, PauliTable.from_labels(['II', 'XY']))

        with self.subTest(msg='set X raises'):

            def set_x():
                pauli = PauliTable.from_labels(['XI', 'IZ'])
                val = np.array([[False, False, False], [True, True, True]],
                               dtype=np.bool)
                pauli.X = val
                return pauli

            self.assertRaises(Exception, set_x)

        with self.subTest(msg='Z'):
            pauli = PauliTable.from_labels(['XI', 'IZ', 'YY'])
            array = np.array([[False, False], [True, False], [True, True]],
                             dtype=np.bool)
            self.assertTrue(np.all(pauli.Z == array))

        with self.subTest(msg='set Z'):
            pauli = PauliTable.from_labels(['XI', 'IZ'])
            val = np.array([[False, False], [True, True]], dtype=np.bool)
            pauli.Z = val
            self.assertEqual(pauli, PauliTable.from_labels(['XI', 'ZZ']))

        with self.subTest(msg='set Z raises'):

            def set_z():
                pauli = PauliTable.from_labels(['XI', 'IZ'])
                val = np.array([[False, False, False], [True, True, True]],
                               dtype=np.bool)
                pauli.Z = val
                return pauli

            self.assertRaises(Exception, set_z)

        with self.subTest(msg='shape'):
            shape = (3, 8)
            pauli = PauliTable(np.zeros(shape))
            self.assertEqual(pauli.shape, shape)

        with self.subTest(msg='size'):
            for j in range(1, 10):
                shape = (j, 8)
                pauli = PauliTable(np.zeros(shape))
                self.assertEqual(pauli.size, j)

        with self.subTest(msg='n_qubits'):
            for j in range(1, 10):
                shape = (5, 2 * j)
                pauli = PauliTable(np.zeros(shape))
                self.assertEqual(pauli.n_qubits, j)

    def test_from_labels(self):
        """Test from_labels method."""
        with self.subTest(msg='1-qubit'):
            labels = ['I', 'Z', 'Z', 'X', 'Y']
            array = np.array([[False, False],
                              [False, True],
                              [False, True],
                              [True, False],
                              [True, True]],
                             dtype=np.bool)
            target = PauliTable(array)
            value = PauliTable.from_labels(labels)
            self.assertEqual(target, value)

        with self.subTest(msg='2-qubit'):
            labels = ['II', 'YY', 'XZ']
            array = np.array([[False, False, False, False],
                              [True, True, True, True],
                              [False, True, True, False]],
                             dtype=np.bool)
            target = PauliTable(array)
            value = PauliTable.from_labels(labels)
            self.assertEqual(target, value)

        with self.subTest(msg='5-qubit'):
            labels = [5 * 'I', 5 * 'X', 5 * 'Y', 5 * 'Z']
            array = np.array([10 * [False],
                              5 * [True] + 5 * [False],
                              10 * [True],
                              5 * [False] + 5 * [True]],
                             dtype=np.bool)
            target = PauliTable(array)
            value = PauliTable.from_labels(labels)
            self.assertEqual(target, value)

    def test_to_labels(self):
        """Test to_labels method."""
        with self.subTest(msg='1-qubit'):
            pauli = PauliTable(np.array([[False, False],
                                         [False, True],
                                         [False, True],
                                         [True, False],
                                         [True, True]],
                                        dtype=np.bool))
            target = ['I', 'Z', 'Z', 'X', 'Y']
            value = pauli.to_labels()
            self.assertEqual(value, target)

        with self.subTest(msg='1-qubit array=True'):
            pauli = PauliTable(np.array([[False, False],
                                         [False, True],
                                         [False, True],
                                         [True, False],
                                         [True, True]],
                                        dtype=np.bool))
            target = np.array(['I', 'Z', 'Z', 'X', 'Y'])
            value = pauli.to_labels(array=True)
            self.assertTrue(np.all(value == target))

        with self.subTest(msg='labels round-trip'):
            target = ['III', 'IXZ', 'XYI', 'ZZZ']
            value = PauliTable.from_labels(target).to_labels()
            self.assertEqual(value, target)

        with self.subTest(msg='array=True'):
            labels = ['III', 'IXZ', 'XYI', 'ZZZ']
            target = np.array(labels)
            value = PauliTable.from_labels(labels).to_labels(array=True)
            self.assertTrue(np.all(value == target))

    def test_to_matrix(self):
        """Test to_matrix method."""
        with self.subTest(msg='dense matrix 1-qubit'):
            labels = ['X', 'I', 'Z', 'Y']
            targets = [self.pauli_mat(i) for i in labels]
            values = PauliTable.from_labels(labels).to_matrix()
            self.assertTrue(isinstance(values, list))
            for target, value in zip(targets, values):
                self.assertTrue(np.all(value == target))

        with self.subTest(msg='dense matrix 1-qubit, array=True'):
            labels = ['Z', 'I', 'Y', 'X']
            target = np.array([self.pauli_mat(i) for i in labels])
            value = PauliTable.from_labels(labels).to_matrix(array=True)
            self.assertTrue(isinstance(value, np.ndarray))
            self.assertTrue(np.all(value == target))

        with self.subTest(msg='sparse matrix 1-qubit'):
            labels = ['X', 'I', 'Z', 'Y']
            targets = [self.pauli_mat(i) for i in labels]
            values = PauliTable.from_labels(labels).to_matrix(sparse=True)
            for mat, targ in zip(values, targets):
                self.assertTrue(isinstance(mat, csr_matrix))
                self.assertTrue(np.all(targ == mat.toarray()))

        with self.subTest(msg='dense matrix 2-qubit'):
            labels = ['IX', 'YI', 'II', 'ZZ']
            targets = [self.pauli_mat(i) for i in labels]
            values = PauliTable.from_labels(labels).to_matrix()
            self.assertTrue(isinstance(values, list))
            for target, value in zip(targets, values):
                self.assertTrue(np.all(value == target))

        with self.subTest(msg='dense matrix 2-qubit, array=True'):
            labels = ['ZZ', 'XY', 'YX', 'IZ']
            target = np.array([self.pauli_mat(i) for i in labels])
            value = PauliTable.from_labels(labels).to_matrix(array=True)
            self.assertTrue(isinstance(value, np.ndarray))
            self.assertTrue(np.all(value == target))

        with self.subTest(msg='sparse matrix 2-qubit'):
            labels = ['IX', 'II', 'ZY', 'YZ']
            targets = [self.pauli_mat(i) for i in labels]
            values = PauliTable.from_labels(labels).to_matrix(sparse=True)
            for mat, targ in zip(values, targets):
                self.assertTrue(isinstance(mat, csr_matrix))
                self.assertTrue(np.all(targ == mat.toarray()))

        with self.subTest(msg='dense matrix 5-qubit'):
            labels = ['IXIXI', 'YZIXI', 'IIXYZ']
            targets = [self.pauli_mat(i) for i in labels]
            values = PauliTable.from_labels(labels).to_matrix()
            self.assertTrue(isinstance(values, list))
            for target, value in zip(targets, values):
                self.assertTrue(np.all(value == target))

        with self.subTest(msg='sparse matrix 5-qubit'):
            labels = ['XXXYY', 'IXIZY', 'ZYXIX']
            targets = [self.pauli_mat(i) for i in labels]
            values = PauliTable.from_labels(labels).to_matrix(sparse=True)
            for mat, targ in zip(values, targets):
                self.assertTrue(isinstance(mat, csr_matrix))
                self.assertTrue(np.all(targ == mat.toarray()))

    def test_magic_methods(self):
        """Test class magic method."""

        with self.subTest(msg='__eq__'):
            pauli1 = PauliTable.from_labels(['II', 'XI'])
            pauli2 = PauliTable.from_labels(['XI', 'II'])
            self.assertEqual(pauli1, pauli1)
            self.assertNotEqual(pauli1, pauli2)

        with self.subTest(msg='__len__'):
            for j in range(1, 10):
                labels = j * ['XX']
                pauli = PauliTable.from_labels(labels)
                self.assertEqual(len(pauli), j)

        with self.subTest(msg='__add__'):
            labels1 = ['XXI', 'IXX']
            labels2 = ['XXI', 'ZZI', 'ZYZ']
            pauli1 = PauliTable.from_labels(labels1)
            pauli2 = PauliTable.from_labels(labels2)
            target = PauliTable.from_labels(labels1 + labels2)
            self.assertEqual(target, pauli1 + pauli2)

        with self.subTest(msg='__getitem__ single'):
            labels = ['XI', 'IY']
            pauli = PauliTable.from_labels(labels)
            self.assertEqual(pauli[0], PauliTable(labels[0]))
            self.assertEqual(pauli[1], PauliTable(labels[1]))

        with self.subTest(msg='__getitem__ array'):
            labels = np.array(['XI', 'IY', 'IZ', 'XY', 'ZX'])
            pauli = PauliTable.from_labels(labels)
            inds = [0, 3]
            self.assertEqual(pauli[inds],
                             PauliTable.from_labels(labels[inds]))
            inds = np.array([4, 1])
            self.assertEqual(pauli[inds],
                             PauliTable.from_labels(labels[inds]))

        with self.subTest(msg='__getitem__ slice'):
            labels = np.array(['XI', 'IY', 'IZ', 'XY', 'ZX'])
            pauli = PauliTable.from_labels(labels)
            self.assertEqual(pauli[:], pauli)
            self.assertEqual(pauli[1:3],
                             PauliTable.from_labels(labels[1:3]))

        with self.subTest(msg='__setitem__ single'):
            labels = ['XI', 'IY']
            pauli = PauliTable.from_labels(['XI', 'IY'])
            pauli[0] = 'II'
            self.assertEqual(pauli[0], PauliTable('II'))
            pauli[1] = 'XX'
            self.assertEqual(pauli[1], PauliTable('XX'))

            def raises_single():
                # Wrong size Pauli
                pauli[0] = 'XXX'

            self.assertRaises(Exception, raises_single)

        with self.subTest(msg='__setitem__ array'):
            labels = np.array(['XI', 'IY', 'IZ'])
            pauli = PauliTable.from_labels(labels)
            target = PauliTable.from_labels(['II', 'ZZ'])
            inds = [2, 0]
            pauli[inds] = target
            self.assertEqual(pauli[inds], target)

            def raises_array():
                pauli[inds] = PauliTable.from_labels(['YY', 'ZZ', 'XX'])
            self.assertRaises(Exception, raises_array)

        with self.subTest(msg='__setitem__ slice'):
            labels = np.array(5 * ['III'])
            pauli = PauliTable.from_labels(labels)
            target = PauliTable.from_labels(5 * ['XXX'])
            pauli[:] = target
            self.assertEqual(pauli[:], target)
            target = PauliTable.from_labels(2 * ['ZZZ'])
            pauli[1:3] = target
            self.assertEqual(pauli[1:3], target)

    def test_sort(self):
        """Test sort method."""
        with self.subTest(msg='1 qubit standard order'):
            unsrt = ['X', 'Z', 'I', 'Y', 'X', 'Z']
            srt = ['I', 'X', 'X', 'Y', 'Z', 'Z']
            target = PauliTable.from_labels(srt)
            value = PauliTable.from_labels(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg='1 qubit weight order'):
            unsrt = ['X', 'Z', 'I', 'Y', 'X', 'Z']
            srt = ['I', 'X', 'X', 'Y', 'Z', 'Z']
            target = PauliTable.from_labels(srt)
            value = PauliTable.from_labels(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg='2 qubit standard order'):
            srt = ['II', 'IX', 'IY', 'IY', 'XI', 'XX', 'XY', 'XZ',
                   'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZI', 'ZX',
                   'ZY', 'ZZ', 'ZZ']
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliTable.from_labels(srt)
            value = PauliTable.from_labels(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg='2 qubit weight order'):
            srt = ['II', 'IX', 'IX', 'IY', 'IZ', 'XI', 'YI', 'YI', 'ZI',
                   'XX', 'XX', 'XY', 'XZ', 'YX', 'YY', 'YY', 'YZ', 'ZX',
                   'ZX', 'ZY', 'ZZ']
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliTable.from_labels(srt)
            value = PauliTable.from_labels(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg='3 qubit standard order'):
            srt = ['III', 'III', 'IIX', 'IIY', 'IIZ', 'IXI', 'IXX', 'IXY', 'IXZ',
                   'IYI', 'IYX', 'IYY', 'IYZ', 'IZI', 'IZX', 'IZY', 'IZY', 'IZZ',
                   'XII', 'XII', 'XIX', 'XIY', 'XIZ', 'XXI', 'XXX', 'XXY', 'XXZ',
                   'XYI', 'XYX', 'XYY', 'XYZ', 'XYZ', 'XZI', 'XZX', 'XZY', 'XZZ',
                   'YII', 'YIX', 'YIY', 'YIZ', 'YXI', 'YXX', 'YXY', 'YXZ', 'YXZ',
                   'YYI', 'YYX', 'YYX', 'YYY', 'YYZ', 'YZI', 'YZX', 'YZY', 'YZZ',
                   'ZII', 'ZIX', 'ZIY', 'ZIZ', 'ZXI', 'ZXX', 'ZXX', 'ZXY', 'ZXZ',
                   'ZYI', 'ZYI', 'ZYX', 'ZYY', 'ZYZ', 'ZZI', 'ZZX', 'ZZY', 'ZZZ']
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliTable.from_labels(srt)
            value = PauliTable.from_labels(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg='3 qubit weight order'):
            srt = ['III', 'IIX', 'IIY', 'IIZ', 'IXI', 'IYI', 'IZI', 'XII', 'YII', 'ZII',
                   'IXX', 'IXY', 'IXZ', 'IYX', 'IYY', 'IYZ', 'IZX', 'IZY', 'IZZ',
                   'XIX', 'XIY', 'XIZ', 'XXI', 'XYI', 'XZI', 'XZI',
                   'YIX', 'YIY', 'YIZ', 'YXI', 'YYI', 'YZI', 'YZI',
                   'ZIX', 'ZIY', 'ZIZ', 'ZXI', 'ZYI', 'ZZI', 'ZZI',
                   'XXX', 'XXY', 'XXZ', 'XYX', 'XYY', 'XYZ', 'XZX', 'XZY', 'XZZ',
                   'YXX', 'YXY', 'YXZ', 'YYX', 'YYY', 'YYZ', 'YZX', 'YZY', 'YZZ',
                   'ZXX', 'ZXY', 'ZXZ', 'ZYX', 'ZYY', 'ZYZ', 'ZZX', 'ZZY', 'ZZZ']
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliTable.from_labels(srt)
            value = PauliTable.from_labels(unsrt).sort(weight=True)
            self.assertEqual(target, value)

    def test_unique(self):
        """Test unique method."""
        with self.subTest(msg='1 qubit'):
            labels = ['X', 'Z', 'X', 'X', 'I', 'Y', 'I', 'X', 'Z', 'Z', 'X', 'I']
            unique = ['X', 'Z', 'I', 'Y']
            target = PauliTable.from_labels(unique)
            value = PauliTable.from_labels(labels).unique()
            self.assertEqual(target, value)

        with self.subTest(msg='2 qubit'):
            labels = ['XX', 'IX', 'XX', 'II', 'IZ', 'ZI', 'YX', 'YX', 'ZZ', 'IX', 'XI']
            unique = ['XX', 'IX', 'II', 'IZ', 'ZI', 'YX', 'ZZ', 'XI']
            target = PauliTable.from_labels(unique)
            value = PauliTable.from_labels(labels).unique()
            self.assertEqual(target, value)

        with self.subTest(msg='10 qubit'):
            labels = [10 * 'X', 10 * 'I', 10 * 'X']
            unique = [10 * 'X', 10 * 'I']
            target = PauliTable.from_labels(unique)
            value = PauliTable.from_labels(labels).unique()
            self.assertEqual(target, value)

    def test_delete(self):
        """Test delete method."""
        with self.subTest(msg='single row'):
            for j in range(1, 6):
                pauli = PauliTable.from_labels([j * 'X', j * 'Y'])
                self.assertEqual(pauli.delete(0), PauliTable(j * 'Y'))
                self.assertEqual(pauli.delete(1), PauliTable(j * 'X'))

        with self.subTest(msg='multiple rows'):
            for j in range(1, 6):
                pauli = PauliTable.from_labels([j * 'X', j * 'Y', j * 'Z'])
                self.assertEqual(pauli.delete([0, 2]), PauliTable(j * 'Y'))
                self.assertEqual(pauli.delete([1, 2]), PauliTable(j * 'X'))
                self.assertEqual(pauli.delete([0, 1]), PauliTable(j * 'Z'))

        with self.subTest(msg='single qubit'):
            pauli = PauliTable.from_labels(['IIX', 'IYI', 'ZII'])
            value = pauli.delete(0, qubit=True)
            target = PauliTable.from_labels(['II', 'IY', 'ZI'])
            self.assertEqual(value, target)
            value = pauli.delete(1, qubit=True)
            target = PauliTable.from_labels(['IX', 'II', 'ZI'])
            self.assertEqual(value, target)
            value = pauli.delete(2, qubit=True)
            target = PauliTable.from_labels(['IX', 'YI', 'II'])
            self.assertEqual(value, target)

        with self.subTest(msg='multiple qubits'):
            pauli = PauliTable.from_labels(['IIX', 'IYI', 'ZII'])
            value = pauli.delete([0, 1], qubit=True)
            target = PauliTable.from_labels(['I', 'I', 'Z'])
            self.assertEqual(value, target)
            value = pauli.delete([1, 2], qubit=True)
            target = PauliTable.from_labels(['X', 'I', 'I'])
            self.assertEqual(value, target)
            value = pauli.delete([0, 2], qubit=True)
            target = PauliTable.from_labels(['I', 'Y', 'I'])
            self.assertEqual(value, target)

    def test_insert(self):
        """Test insert method."""
        # Insert single row
        for j in range(1, 10):
            pauli = PauliTable(j * 'X')
            target0 = PauliTable.from_labels([j * 'I', j * 'X'])
            target1 = PauliTable.from_labels([j * 'X', j * 'I'])

            with self.subTest(msg='single row from str ({})'.format(j)):
                value0 = pauli.insert(0, j * 'I')
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, j * 'I')
                self.assertEqual(value1, target1)

            with self.subTest(msg='single row from PauliTable ({})'.format(j)):
                value0 = pauli.insert(0, PauliTable(j * 'I'))
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, PauliTable(j * 'I'))
                self.assertEqual(value1, target1)

            with self.subTest(msg='single row from array ({})'.format(j)):
                value0 = pauli.insert(0, PauliTable(j * 'I').array)
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, PauliTable(j * 'I').array)
                self.assertEqual(value1, target1)

        # Insert multiple rows
        for j in range(1, 10):
            pauli = PauliTable(j * 'X')
            insert = PauliTable.from_labels([j * 'I', j * 'Y', j * 'Z'])
            target0 = insert + pauli
            target1 = pauli + insert

            with self.subTest(msg='multiple-rows from PauliTable ({})'.format(j)):
                value0 = pauli.insert(0, insert)
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, insert)
                self.assertEqual(value1, target1)

            with self.subTest(msg='multiple-rows from array ({})'.format(j)):
                value0 = pauli.insert(0, insert.array)
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, insert.array)
                self.assertEqual(value1, target1)

        # Insert single column
        pauli = PauliTable.from_labels(['X', 'Y', 'Z'])
        for i in ['I', 'X', 'Y', 'Z']:
            target0 = PauliTable.from_labels(['X' + i, 'Y' + i, 'Z' + i])
            target1 = PauliTable.from_labels([i + 'X', i + 'Y', i + 'Z'])

            with self.subTest(msg='single-column single-val from str'):
                value = pauli.insert(0, i, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, i, qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg='single-column single-val from PauliTable'):
                value = pauli.insert(0, PauliTable(i), qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliTable(i), qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg='single-column single-val from array'):
                value = pauli.insert(0, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target1)

        # Insert single column with multiple values
        pauli = PauliTable.from_labels(['X', 'Y', 'Z'])
        for i in [('I', 'X', 'Y'), ('X', 'Y', 'Z'), ('Y', 'Z', 'I')]:
            target0 = PauliTable.from_labels(['X' + i[0], 'Y' + i[1], 'Z' + i[2]])
            target1 = PauliTable.from_labels([i[0] + 'X', i[1] + 'Y', i[2] + 'Z'])

            with self.subTest(msg='single-column multiple-vals from PauliTable'):
                value = pauli.insert(0, PauliTable.from_labels(i), qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliTable.from_labels(i), qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg='single-column multiple-vals from array'):
                value = pauli.insert(0, PauliTable.from_labels(i).array, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliTable.from_labels(i).array, qubit=True)
                self.assertEqual(value, target1)

        # Insert multiple columns from single
        pauli = PauliTable.from_labels(['X', 'Y', 'Z'])
        for j in range(1, 5):
            for i in [j * 'I', j * 'X', j * 'Y', j * 'Z']:
                target0 = PauliTable.from_labels(['X' + i, 'Y' + i, 'Z' + i])
                target1 = PauliTable.from_labels([i + 'X', i + 'Y', i + 'Z'])

            with self.subTest(msg='multiple-columns single-val from str'):
                value = pauli.insert(0, i, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, i, qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg='multiple-columns single-val from PauliTable'):
                value = pauli.insert(0, PauliTable(i), qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliTable(i), qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg='multiple-columns single-val from array'):
                value = pauli.insert(0, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target1)

        # Insert multiple columns multiple row values
        pauli = PauliTable.from_labels(['X', 'Y', 'Z'])
        for j in range(1, 5):
            for i in [(j * 'I', j * 'X', j * 'Y'),
                      (j * 'X', j * 'Z', j * 'Y'),
                      (j * 'Y', j * 'Z', j * 'I')]:
                target0 = PauliTable.from_labels(['X' + i[0], 'Y' + i[1], 'Z' + i[2]])
                target1 = PauliTable.from_labels([i[0] + 'X', i[1] + 'Y', i[2] + 'Z'])

                with self.subTest(msg='multiple-column multiple-vals from PauliTable'):
                    value = pauli.insert(0, PauliTable.from_labels(i), qubit=True)
                    self.assertEqual(value, target0)
                    value = pauli.insert(1, PauliTable.from_labels(i), qubit=True)
                    self.assertEqual(value, target1)

                with self.subTest(msg='multiple-column multiple-vals from array'):
                    value = pauli.insert(0, PauliTable.from_labels(i).array, qubit=True)
                    self.assertEqual(value, target0)
                    value = pauli.insert(1, PauliTable.from_labels(i).array, qubit=True)
                    self.assertEqual(value, target1)

    def test_iteration(self):
        """Test iteration methods."""

        labels = ['III', 'IXI', 'IYY', 'YIZ', 'ZIZ', 'XYZ', 'III']
        pauli = PauliTable.from_labels(labels)

        with self.subTest(msg='enumerate'):
            for idx, i in enumerate(pauli):
                self.assertEqual(i, PauliTable(labels[idx]))

        with self.subTest(msg='iter'):
            for idx, i in enumerate(iter(pauli)):
                self.assertEqual(i, PauliTable(labels[idx]))

        with self.subTest(msg='zip'):
            for label, i in zip(labels, pauli):
                self.assertEqual(i, PauliTable(label))

        with self.subTest(msg='label_iter'):
            for idx, i in enumerate(pauli.label_iter()):
                self.assertEqual(i, labels[idx])

        with self.subTest(msg='matrix_iter (dense)'):
            for idx, i in enumerate(pauli.matrix_iter()):
                self.assertTrue(np.all(i == self.pauli_mat(labels[idx])))

        with self.subTest(msg='matrix_iter (sparse)'):
            for idx, i in enumerate(pauli.matrix_iter(sparse=True)):
                self.assertTrue(isinstance(i, csr_matrix))
                self.assertTrue(np.all(i.toarray() == self.pauli_mat(labels[idx])))

    def test_tensor(self):
        """Test tensor and expand methods."""
        for j in range(1, 10):
            labels1 = ['XX', 'YY']
            labels2 = [j * 'I', j * 'Z']
            pauli1 = PauliTable.from_labels(labels1)
            pauli2 = PauliTable.from_labels(labels2)

            with self.subTest(msg='tensor ({})'.format(j)):
                value = pauli1.tensor(pauli2)
                target = PauliTable.from_labels(
                    [i + j for i in labels1 for j in labels2])
                self.assertEqual(value, target)

            with self.subTest(msg='expand ({})'.format(j)):
                value = pauli1.expand(pauli2)
                target = PauliTable.from_labels(
                    [j + i for i in labels1 for j in labels2])
                self.assertEqual(value, target)

    def test_dot(self):
        """Test dot and compose methods."""

        # Test single qubit Pauli dot products
        pauli = PauliTable.from_labels(['I', 'X', 'Y', 'Z'])

        target = PauliTable.from_labels(['I', 'X', 'Y', 'Z'])
        with self.subTest(msg='dot single I'):
            value = pauli.dot('I')
            self.assertEqual(target, value)

        with self.subTest(msg='compose single I'):
            value = pauli.compose('I')
            self.assertEqual(target, value)

        target = PauliTable.from_labels(['X', 'I', 'Z', 'Y'])
        with self.subTest(msg='dot single X'):
            value = pauli.dot('X')
            self.assertEqual(target, value)

        with self.subTest(msg='compose single X'):
            value = pauli.compose('X')
            self.assertEqual(target, value)

        target = PauliTable.from_labels(['Y', 'Z', 'I', 'X'])
        with self.subTest(msg='dot single Y'):
            value = pauli.dot('Y')
            self.assertEqual(target, value)

        with self.subTest(msg='compose single Y'):
            value = pauli.compose('Y')
            self.assertEqual(target, value)

        target = PauliTable.from_labels(['Z', 'Y', 'X', 'I'])
        with self.subTest(msg='dot single Z'):
            value = pauli.dot('Z')
            self.assertEqual(target, value)

        with self.subTest(msg='compose single Z'):
            value = pauli.compose('Z')
            self.assertEqual(target, value)

        # Dot product with qargs
        pauli1 = PauliTable.from_labels(['III', 'XXX'])

        # 1-qubit qargs
        pauli2 = PauliTable('Z')

        target = PauliTable.from_labels(['IIZ', 'XXY'])
        with self.subTest(msg='dot 1-qubit qargs=[0]'):
            value = pauli1.dot(pauli2, qargs=[0])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 1-qubit qargs=[0]'):
            value = pauli1.compose(pauli2, qargs=[0])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['IZI', 'XYX'])
        with self.subTest(msg='dot 1-qubit qargs=[1]'):
            value = pauli1.dot(pauli2, qargs=[1])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 1-qubit qargs=[1]'):
            value = pauli1.compose(pauli2, qargs=[1])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['ZII', 'YXX'])
        with self.subTest(msg='dot 1-qubit qargs=[2]'):
            value = pauli1.dot(pauli2, qargs=[2])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 1-qubit qargs=[2]'):
            value = pauli1.compose(pauli2, qargs=[2])
            self.assertEqual(value, target)

        # 2-qubit qargs
        pauli2 = PauliTable('ZY')

        target = PauliTable.from_labels(['IZY', 'XYZ'])
        with self.subTest(msg='dot 2-qubit qargs=[0, 1]'):
            value = pauli1.dot(pauli2, qargs=[0, 1])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 2-qubit qargs=[0, 1]'):
            value = pauli1.compose(pauli2, qargs=[0, 1])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['IYZ', 'XZY'])
        with self.subTest(msg='dot 2-qubit qargs=[1, 0]'):
            value = pauli1.dot(pauli2, qargs=[1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 2-qubit qargs=[1, 0]'):
            value = pauli1.compose(pauli2, qargs=[1, 0])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['ZIY', 'YXZ'])
        with self.subTest(msg='dot 2-qubit qargs=[0, 2]'):
            value = pauli1.dot(pauli2, qargs=[0, 2])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 2-qubit qargs=[0, 2]'):
            value = pauli1.compose(pauli2, qargs=[0, 2])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['YIZ', 'ZXY'])
        with self.subTest(msg='dot 2-qubit qargs=[2, 0]'):
            value = pauli1.dot(pauli2, qargs=[2, 0])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 2-qubit qargs=[2, 0]'):
            value = pauli1.compose(pauli2, qargs=[2, 0])
            self.assertEqual(value, target)

        # 3-qubit qargs
        pauli2 = PauliTable('XYZ')

        target = PauliTable.from_labels(['XYZ', 'IZY'])
        with self.subTest(msg='dot 3-qubit qargs=None'):
            value = pauli1.dot(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 3-qubit qargs=None'):
            value = pauli1.compose(pauli2)
            self.assertEqual(value, target)

        with self.subTest(msg='dot 3-qubit qargs=[0, 1, 2]'):
            value = pauli1.dot(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 3-qubit qargs=[0, 1, 2]'):
            value = pauli1.compose(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['ZYX', 'YZI'])
        with self.subTest(msg='dot 3-qubit qargs=[2, 1, 0]'):
            value = pauli1.dot(pauli2, qargs=[2, 1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 3-qubit qargs=[2, 1, 0]'):
            value = pauli1.compose(pauli2, qargs=[2, 1, 0])
            self.assertEqual(value, target)

        target = PauliTable.from_labels(['XZY', 'IYZ'])
        with self.subTest(msg='dot 3-qubit qargs=[1, 0, 2]'):
            value = pauli1.dot(pauli2, qargs=[1, 0, 2])
            self.assertEqual(value, target)

        with self.subTest(msg='compose 3-qubit qargs=[1, 0, 2]'):
            value = pauli1.compose(pauli2, qargs=[1, 0, 2])
            self.assertEqual(value, target)

    def test_commutes(self):
        """Test commutes method."""
        # Single qubit Pauli
        pauli = PauliTable.from_labels(['I', 'X', 'Y', 'Z'])
        with self.subTest(msg='commutes single-Pauli I'):
            value = list(pauli.commutes('I'))
            target = [True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli X'):
            value = list(pauli.commutes('X'))
            target = [True, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli Y'):
            value = list(pauli.commutes('Y'))
            target = [True, False, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli Z'):
            value = list(pauli.commutes('Z'))
            target = [True, False, False, True]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliTable.from_labels(['II', 'IX', 'YI', 'XY', 'ZZ'])
        with self.subTest(msg='commutes single-Pauli II'):
            value = list(pauli.commutes('II'))
            target = [True, True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli IX'):
            value = list(pauli.commutes('IX'))
            target = [True, True, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli XI'):
            value = list(pauli.commutes('XI'))
            target = [True, True, False, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli YI'):
            value = list(pauli.commutes('YI'))
            target = [True, True, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli IY'):
            value = list(pauli.commutes('IY'))
            target = [True, False, True, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli XY'):
            value = list(pauli.commutes('XY'))
            target = [True, False, False, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli YX'):
            value = list(pauli.commutes('YX'))
            target = [True, True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes single-Pauli ZZ'):
            value = list(pauli.commutes('ZZ'))
            target = [True, False, False, True, True]
            self.assertEqual(value, target)

    def test_commutes_with_all(self):
        """Test commutes_with_all method."""
        # 1-qubit
        pauli = PauliTable.from_labels(['I', 'X', 'Y', 'Z'])
        with self.subTest(msg='commutes_with_all [I]'):
            value = list(pauli.commutes_with_all('I'))
            target = [0, 1, 2, 3]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [X]'):
            value = list(pauli.commutes_with_all('X'))
            target = [0, 1]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [Y]'):
            value = list(pauli.commutes_with_all('Y'))
            target = [0, 2]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [Z]'):
            value = list(pauli.commutes_with_all('Z'))
            target = [0, 3]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliTable.from_labels(['II', 'IX', 'YI', 'XY', 'ZZ'])

        with self.subTest(msg='commutes_with_all [IX, YI]'):
            other = PauliTable.from_labels(['IX', 'YI'])
            value = list(pauli.commutes_with_all(other))
            target = [0, 1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [XY, ZZ]'):
            other = PauliTable.from_labels(['XY', 'ZZ'])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [YX, ZZ]'):
            other = PauliTable.from_labels(['YX', 'ZZ'])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [XY, YX]'):
            other = PauliTable.from_labels(['XY', 'YX'])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [XY, IX]'):
            other = PauliTable.from_labels(['XY', 'IX'])
            value = list(pauli.commutes_with_all(other))
            target = [0]
            self.assertEqual(value, target)

        with self.subTest(msg='commutes_with_all [YX, IX]'):
            other = PauliTable.from_labels(['YX', 'IX'])
            value = list(pauli.commutes_with_all(other))
            target = [0, 1, 2]
            self.assertEqual(value, target)

    def test_anticommutes_with_all(self):
        """Test anticommutes_with_all method."""
        # 1-qubit
        pauli = PauliTable.from_labels(['I', 'X', 'Y', 'Z'])
        with self.subTest(msg='anticommutes_with_all [I]'):
            value = list(pauli.anticommutes_with_all('I'))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg='antianticommutes_with_all [X]'):
            value = list(pauli.anticommutes_with_all('X'))
            target = [2, 3]
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [Y]'):
            value = list(pauli.anticommutes_with_all('Y'))
            target = [1, 3]
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [Z]'):
            value = list(pauli.anticommutes_with_all('Z'))
            target = [1, 2]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliTable.from_labels(['II', 'IX', 'YI', 'XY', 'ZZ'])

        with self.subTest(msg='anticommutes_with_all [IX, YI]'):
            other = PauliTable.from_labels(['IX', 'YI'])
            value = list(pauli.anticommutes_with_all(other))
            target = [3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [XY, ZZ]'):
            other = PauliTable.from_labels(['XY', 'ZZ'])
            value = list(pauli.anticommutes_with_all(other))
            target = [1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [YX, ZZ]'):
            other = PauliTable.from_labels(['YX', 'ZZ'])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [XY, YX]'):
            other = PauliTable.from_labels(['XY', 'YX'])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [XY, IX]'):
            other = PauliTable.from_labels(['XY', 'IX'])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg='anticommutes_with_all [YX, IX]'):
            other = PauliTable.from_labels(['YX', 'IX'])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)


if __name__ == '__main__':
    unittest.main()
