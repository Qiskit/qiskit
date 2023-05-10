# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PauliTable class."""

import unittest
from test import combine

import numpy as np
from ddt import ddt
from scipy.sparse import csr_matrix

from qiskit import QiskitError
from qiskit.quantum_info.operators.symplectic import PauliTable
from qiskit.test import QiskitTestCase


def pauli_mat(label):
    """Return Pauli matrix from a Pauli label"""
    mat = np.eye(1, dtype=complex)
    for i in label:
        if i == "I":
            mat = np.kron(mat, np.eye(2, dtype=complex))
        elif i == "X":
            mat = np.kron(mat, np.array([[0, 1], [1, 0]], dtype=complex))
        elif i == "Y":
            mat = np.kron(mat, np.array([[0, -1j], [1j, 0]], dtype=complex))
        elif i == "Z":
            mat = np.kron(mat, np.array([[1, 0], [0, -1]], dtype=complex))
        else:
            raise QiskitError(f"Invalid Pauli string {i}")
    return mat


class TestPauliTableInit(QiskitTestCase):
    """Tests for PauliTable initialization."""

    def test_array_init(self):
        """Test array initialization."""
        # Matrix array initialization
        with self.subTest(msg="bool array"):
            target = np.array([[False, False], [True, True]])
            with self.assertWarns(DeprecationWarning):
                value = PauliTable(target)._array
            self.assertTrue(np.all(value == target))

        with self.subTest(msg="bool array no copy"):
            target = np.array([[False, True], [True, True]])
            with self.assertWarns(DeprecationWarning):
                value = PauliTable(target)._array
            value[0, 0] = not value[0, 0]
            self.assertTrue(np.all(value == target))

        with self.subTest(msg="bool array raises"):
            array = np.array([[False, False, False], [True, True, True]])
            with self.assertWarns(DeprecationWarning):
                self.assertRaises(QiskitError, PauliTable, array)

    def test_vector_init(self):
        """Test vector initialization."""
        # Vector array initialization
        with self.subTest(msg="bool vector"):
            target = np.array([False, False, False, False])
            with self.assertWarns(DeprecationWarning):
                value = PauliTable(target)._array
            self.assertTrue(np.all(value == target))

        with self.subTest(msg="bool vector no copy"):
            target = np.array([False, True, True, False])
            with self.assertWarns(DeprecationWarning):
                value = PauliTable(target)._array
            value[0, 0] = not value[0, 0]
            self.assertTrue(np.all(value == target))

    def test_string_init(self):
        """Test string initialization."""
        # String initialization
        with self.subTest(msg='str init "I"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("I")._array
            target = np.array([[False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "X"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("X")._array
            target = np.array([[True, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "Y"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("Y")._array
            target = np.array([[True, True]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "Z"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("Z")._array
            target = np.array([[False, True]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "IX"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("IX")._array
            target = np.array([[True, False, False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "XI"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("XI")._array
            target = np.array([[False, True, False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "YZ"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("YZ")._array
            target = np.array([[False, True, True, True]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "XIZ"'):
            with self.assertWarns(DeprecationWarning):
                value = PauliTable("XIZ")._array
            target = np.array([[False, False, True, True, False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

    def test_table_init(self):
        """Test table initialization."""
        # Pauli Table initialization
        with self.subTest(msg="PauliTable"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XI", "IX", "IZ"])
                value = PauliTable(target)
            self.assertEqual(value, target)

        with self.subTest(msg="PauliTable no copy"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XI", "IX", "IZ"])
                value = PauliTable(target)
            value[0] = "II"
            self.assertEqual(value, target)


class TestPauliTableProperties(QiskitTestCase):
    """Tests for PauliTable properties."""

    def test_array_propertiy(self):
        """Test array property"""

        with self.subTest(msg="array"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable("II")
            array = np.zeros([2, 4], dtype=bool)
            self.assertTrue(np.all(pauli.array == array))

        with self.subTest(msg="set array"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable("XX")
            array = np.zeros([1, 4], dtype=bool)
            pauli.array = array
            self.assertTrue(np.all(pauli.array == array))

        with self.subTest(msg="set array raises"):

            def set_array_raise():
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable("XXX")
                pauli.array = np.eye(4)
                return pauli

            self.assertRaises(ValueError, set_array_raise)

    def test_x_propertiy(self):
        """Test X property"""
        with self.subTest(msg="X"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["XI", "IZ", "YY"])
            array = np.array([[False, True], [False, False], [True, True]], dtype=bool)
            self.assertTrue(np.all(pauli.X == array))

        with self.subTest(msg="set X"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["XI", "IZ"])
            val = np.array([[False, False], [True, True]], dtype=bool)
            pauli.X = val
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli, PauliTable.from_labels(["II", "XY"]))

        with self.subTest(msg="set X raises"):

            def set_x():
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable.from_labels(["XI", "IZ"])
                val = np.array([[False, False, False], [True, True, True]], dtype=bool)
                pauli.X = val
                return pauli

            self.assertRaises(Exception, set_x)

    def test_z_propertiy(self):
        """Test Z property"""
        with self.subTest(msg="Z"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["XI", "IZ", "YY"])
            array = np.array([[False, False], [True, False], [True, True]], dtype=bool)
            self.assertTrue(np.all(pauli.Z == array))

        with self.subTest(msg="set Z"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["XI", "IZ"])
            val = np.array([[False, False], [True, True]], dtype=bool)
            pauli.Z = val
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli, PauliTable.from_labels(["XI", "ZZ"]))

        with self.subTest(msg="set Z raises"):

            def set_z():
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable.from_labels(["XI", "IZ"])
                val = np.array([[False, False, False], [True, True, True]], dtype=bool)
                pauli.Z = val
                return pauli

            self.assertRaises(Exception, set_z)

    def test_shape_propertiy(self):
        """Test shape property"""
        shape = (3, 8)
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable(np.zeros(shape))
        self.assertEqual(pauli.shape, shape)

    def test_size_propertiy(self):
        """Test size property"""
        with self.subTest(msg="size"):
            for j in range(1, 10):
                shape = (j, 8)
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable(np.zeros(shape))
                self.assertEqual(pauli.size, j)

    def test_n_qubit_propertiy(self):
        """Test n_qubit property"""
        with self.subTest(msg="num_qubits"):
            for j in range(1, 10):
                shape = (5, 2 * j)
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable(np.zeros(shape))
                self.assertEqual(pauli.num_qubits, j)

    def test_eq(self):
        """Test __eq__ method."""
        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["II", "XI"])
            pauli2 = PauliTable.from_labels(["XI", "II"])
        self.assertEqual(pauli1, pauli1)
        self.assertNotEqual(pauli1, pauli2)

    def test_len_methods(self):
        """Test __len__ method."""
        for j in range(1, 10):
            labels = j * ["XX"]
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(labels)
            self.assertEqual(len(pauli), j)

    def test_add_methods(self):
        """Test __add__ method."""
        labels1 = ["XXI", "IXX"]
        labels2 = ["XXI", "ZZI", "ZYZ"]
        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(labels1)
            pauli2 = PauliTable.from_labels(labels2)
            target = PauliTable.from_labels(labels1 + labels2)
        self.assertEqual(target, pauli1 + pauli2)

    def test_add_qargs(self):
        """Test add method with qargs."""
        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["IIII", "YYYY"])
            pauli2 = PauliTable.from_labels(["XY", "YZ"])

        with self.subTest(msg="qargs=[0, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IIII", "YYYY", "IIXY", "IIYZ"])
            self.assertEqual(pauli1 + pauli2([0, 1]), target)

        with self.subTest(msg="qargs=[0, 3]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IIII", "YYYY", "XIIY", "YIIZ"])
            self.assertEqual(pauli1 + pauli2([0, 3]), target)

        with self.subTest(msg="qargs=[2, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IIII", "YYYY", "IYXI", "IZYI"])
            self.assertEqual(pauli1 + pauli2([2, 1]), target)

        with self.subTest(msg="qargs=[3, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IIII", "YYYY", "YIXI", "ZIYI"])
            self.assertEqual(pauli1 + pauli2([3, 1]), target)

    def test_getitem_methods(self):
        """Test __getitem__ method."""
        with self.subTest(msg="__getitem__ single"):
            labels = ["XI", "IY"]
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(labels)
                self.assertEqual(pauli[0], PauliTable(labels[0]))
                self.assertEqual(pauli[1], PauliTable(labels[1]))

        with self.subTest(msg="__getitem__ array"):
            labels = np.array(["XI", "IY", "IZ", "XY", "ZX"])
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(labels)
            inds = [0, 3]
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli[inds], PauliTable.from_labels(labels[inds]))
            inds = np.array([4, 1])
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli[inds], PauliTable.from_labels(labels[inds]))

        with self.subTest(msg="__getitem__ slice"):
            labels = np.array(["XI", "IY", "IZ", "XY", "ZX"])
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(labels)
            self.assertEqual(pauli[:], pauli)
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli[1:3], PauliTable.from_labels(labels[1:3]))

    def test_setitem_methods(self):
        """Test __setitem__ method."""
        with self.subTest(msg="__setitem__ single"):
            labels = ["XI", "IY"]
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["XI", "IY"])
            pauli[0] = "II"
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli[0], PauliTable("II"))
            pauli[1] = "XX"
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pauli[1], PauliTable("XX"))

            def raises_single():
                # Wrong size Pauli
                pauli[0] = "XXX"

            self.assertRaises(Exception, raises_single)

        with self.subTest(msg="__setitem__ array"):
            labels = np.array(["XI", "IY", "IZ"])
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(labels)
                target = PauliTable.from_labels(["II", "ZZ"])
            inds = [2, 0]
            pauli[inds] = target
            self.assertEqual(pauli[inds], target)

            def raises_array():
                with self.assertWarns(DeprecationWarning):
                    pauli[inds] = PauliTable.from_labels(["YY", "ZZ", "XX"])

            self.assertRaises(Exception, raises_array)

        with self.subTest(msg="__setitem__ slice"):
            labels = np.array(5 * ["III"])
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(labels)
                target = PauliTable.from_labels(5 * ["XXX"])
            pauli[:] = target
            self.assertEqual(pauli[:], target)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(2 * ["ZZZ"])
            pauli[1:3] = target
            self.assertEqual(pauli[1:3], target)


class TestPauliTableLabels(QiskitTestCase):
    """Tests PauliTable label representation conversions."""

    def test_from_labels_1q(self):
        """Test 1-qubit from_labels method."""
        labels = ["I", "Z", "Z", "X", "Y"]
        array = np.array(
            [[False, False], [False, True], [False, True], [True, False], [True, True]], dtype=bool
        )
        with self.assertWarns(DeprecationWarning):
            target = PauliTable(array)
            value = PauliTable.from_labels(labels)
        self.assertEqual(target, value)

    def test_from_labels_2q(self):
        """Test 2-qubit from_labels method."""
        labels = ["II", "YY", "XZ"]
        array = np.array(
            [[False, False, False, False], [True, True, True, True], [False, True, True, False]],
            dtype=bool,
        )
        with self.assertWarns(DeprecationWarning):
            target = PauliTable(array)
            value = PauliTable.from_labels(labels)
        self.assertEqual(target, value)

    def test_from_labels_5q(self):
        """Test 5-qubit from_labels method."""
        labels = [5 * "I", 5 * "X", 5 * "Y", 5 * "Z"]
        array = np.array(
            [10 * [False], 5 * [True] + 5 * [False], 10 * [True], 5 * [False] + 5 * [True]],
            dtype=bool,
        )
        with self.assertWarns(DeprecationWarning):
            target = PauliTable(array)
            value = PauliTable.from_labels(labels)
        self.assertEqual(target, value)

    def test_to_labels_1q(self):
        """Test 1-qubit to_labels method."""
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable(
                np.array(
                    [[False, False], [False, True], [False, True], [True, False], [True, True]],
                    dtype=bool,
                )
            )
        target = ["I", "Z", "Z", "X", "Y"]
        value = pauli.to_labels()
        self.assertEqual(value, target)

    def test_to_labels_1q_array(self):
        """Test 1-qubit to_labels method w/ array=True."""
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable(
                np.array(
                    [[False, False], [False, True], [False, True], [True, False], [True, True]],
                    dtype=bool,
                )
            )
        target = np.array(["I", "Z", "Z", "X", "Y"])
        value = pauli.to_labels(array=True)
        self.assertTrue(np.all(value == target))

    def test_labels_round_trip(self):
        """Test from_labels and to_labels round trip."""
        target = ["III", "IXZ", "XYI", "ZZZ"]
        with self.assertWarns(DeprecationWarning):
            value = PauliTable.from_labels(target).to_labels()
        self.assertEqual(value, target)

    def test_labels_round_trip_array(self):
        """Test from_labels and to_labels round trip w/ array=True."""
        labels = ["III", "IXZ", "XYI", "ZZZ"]
        target = np.array(labels)
        with self.assertWarns(DeprecationWarning):
            value = PauliTable.from_labels(labels).to_labels(array=True)
        self.assertTrue(np.all(value == target))


class TestPauliTableMatrix(QiskitTestCase):
    """Tests PauliTable matrix representation conversions."""

    def test_to_matrix_1q(self):
        """Test 1-qubit to_matrix method."""
        labels = ["X", "I", "Z", "Y"]
        targets = [pauli_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = PauliTable.from_labels(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_1q_array(self):
        """Test 1-qubit to_matrix method w/ array=True."""
        labels = ["Z", "I", "Y", "X"]
        target = np.array([pauli_mat(i) for i in labels])
        with self.assertWarns(DeprecationWarning):
            value = PauliTable.from_labels(labels).to_matrix(array=True)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertTrue(np.all(value == target))

    def test_to_matrix_1q_sparse(self):
        """Test 1-qubit to_matrix method w/ sparse=True."""
        labels = ["X", "I", "Z", "Y"]
        targets = [pauli_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = PauliTable.from_labels(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_2q(self):
        """Test 2-qubit to_matrix method."""
        labels = ["IX", "YI", "II", "ZZ"]
        targets = [pauli_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = PauliTable.from_labels(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_2q_array(self):
        """Test 2-qubit to_matrix method w/ array=True."""
        labels = ["ZZ", "XY", "YX", "IZ"]
        target = np.array([pauli_mat(i) for i in labels])
        with self.assertWarns(DeprecationWarning):
            value = PauliTable.from_labels(labels).to_matrix(array=True)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertTrue(np.all(value == target))

    def test_to_matrix_2q_sparse(self):
        """Test 2-qubit to_matrix method w/ sparse=True."""
        labels = ["IX", "II", "ZY", "YZ"]
        targets = [pauli_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = PauliTable.from_labels(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_5q(self):
        """Test 5-qubit to_matrix method."""
        labels = ["IXIXI", "YZIXI", "IIXYZ"]
        targets = [pauli_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = PauliTable.from_labels(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_5q_sparse(self):
        """Test 5-qubit to_matrix method w/ sparse=True."""
        labels = ["XXXYY", "IXIZY", "ZYXIX"]
        targets = [pauli_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = PauliTable.from_labels(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))


class TestPauliTableIteration(QiskitTestCase):
    """Tests for PauliTable iterators class."""

    def test_enumerate(self):
        """Test enumerate with PauliTable."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(labels)
        for idx, i in enumerate(pauli):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(i, PauliTable(labels[idx]))

    def test_iter(self):
        """Test iter with PauliTable."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(labels)
        for idx, i in enumerate(iter(pauli)):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(i, PauliTable(labels[idx]))

    def test_zip(self):
        """Test zip with PauliTable."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(labels)
        for label, i in zip(labels, pauli):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(i, PauliTable(label))

    def test_label_iter(self):
        """Test PauliTable label_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(labels)
        for idx, i in enumerate(pauli.label_iter()):
            self.assertEqual(i, labels[idx])

    def test_matrix_iter(self):
        """Test PauliTable dense matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(labels)
        for idx, i in enumerate(pauli.matrix_iter()):
            self.assertTrue(np.all(i == pauli_mat(labels[idx])))

    def test_matrix_iter_sparse(self):
        """Test PauliTable sparse matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(labels)
        for idx, i in enumerate(pauli.matrix_iter(sparse=True)):
            self.assertTrue(isinstance(i, csr_matrix))
            self.assertTrue(np.all(i.toarray() == pauli_mat(labels[idx])))


@ddt
class TestPauliTableOperator(QiskitTestCase):
    """Tests for PauliTable base operator methods."""

    @combine(j=range(1, 10))
    def test_tensor(self, j):
        """Test tensor method j={j}."""
        labels1 = ["XX", "YY"]
        labels2 = [j * "I", j * "Z"]
        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(labels1)
            pauli2 = PauliTable.from_labels(labels2)

        value = pauli1.tensor(pauli2)
        with self.assertWarns(DeprecationWarning):
            target = PauliTable.from_labels([i + j for i in labels1 for j in labels2])
        self.assertEqual(value, target)

    @combine(j=range(1, 10))
    def test_expand(self, j):
        """Test expand method j={j}."""
        labels1 = ["XX", "YY"]
        labels2 = [j * "I", j * "Z"]
        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(labels1)
            pauli2 = PauliTable.from_labels(labels2)

        value = pauli1.expand(pauli2)
        with self.assertWarns(DeprecationWarning):
            target = PauliTable.from_labels([j + i for j in labels2 for i in labels1])
        self.assertEqual(value, target)

    def test_compose_1q(self):
        """Test 1-qubit compose methods."""
        # Test single qubit Pauli dot products
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["I", "X", "Y", "Z"])

        with self.subTest(msg="compose single I"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["I", "X", "Y", "Z"])
            value = pauli.compose("I")
            self.assertEqual(target, value)

        with self.subTest(msg="compose single X"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["X", "I", "Z", "Y"])
            value = pauli.compose("X")
            self.assertEqual(target, value)

        with self.subTest(msg="compose single Y"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["Y", "Z", "I", "X"])
            value = pauli.compose("Y")
            self.assertEqual(target, value)

        with self.subTest(msg="compose single Z"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["Z", "Y", "X", "I"])
            value = pauli.compose("Z")
            self.assertEqual(target, value)

    def test_dot_1q(self):
        """Test 1-qubit dot method."""
        # Test single qubit Pauli dot products
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["I", "X", "Y", "Z"])

        with self.subTest(msg="dot single I"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["I", "X", "Y", "Z"])
            value = pauli.dot("I")
            self.assertEqual(target, value)

        with self.subTest(msg="dot single X"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["X", "I", "Z", "Y"])
            value = pauli.dot("X")
            self.assertEqual(target, value)

        with self.subTest(msg="dot single Y"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["Y", "Z", "I", "X"])
            value = pauli.dot("Y")
            self.assertEqual(target, value)

        with self.subTest(msg="dot single Z"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["Z", "Y", "X", "I"])
            value = pauli.dot("Z")
            self.assertEqual(target, value)

    def test_qargs_compose_1q(self):
        """Test 1-qubit compose method with qargs."""

        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["III", "XXX"])
            pauli2 = PauliTable("Z")

        with self.subTest(msg="compose 1-qubit qargs=[0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IIZ", "XXY"])
            value = pauli1.compose(pauli2, qargs=[0])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IZI", "XYX"])
            value = pauli1.compose(pauli2, qargs=[1])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["ZII", "YXX"])
            value = pauli1.compose(pauli2, qargs=[2])
            self.assertEqual(value, target)

    def test_qargs_dot_1q(self):
        """Test 1-qubit dot method with qargs."""

        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["III", "XXX"])
            pauli2 = PauliTable("Z")

        with self.subTest(msg="dot 1-qubit qargs=[0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IIZ", "XXY"])
            value = pauli1.dot(pauli2, qargs=[0])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 1-qubit qargs=[1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IZI", "XYX"])
            value = pauli1.dot(pauli2, qargs=[1])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 1-qubit qargs=[2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["ZII", "YXX"])
            value = pauli1.dot(pauli2, qargs=[2])
            self.assertEqual(value, target)

    def test_qargs_compose_2q(self):
        """Test 2-qubit compose method with qargs."""

        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["III", "XXX"])
            pauli2 = PauliTable("ZY")

        with self.subTest(msg="compose 2-qubit qargs=[0, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IZY", "XYZ"])
            value = pauli1.compose(pauli2, qargs=[0, 1])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[1, 0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IYZ", "XZY"])
            value = pauli1.compose(pauli2, qargs=[1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[0, 2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["ZIY", "YXZ"])
            value = pauli1.compose(pauli2, qargs=[0, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[2, 0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["YIZ", "ZXY"])
            value = pauli1.compose(pauli2, qargs=[2, 0])
            self.assertEqual(value, target)

    def test_qargs_dot_2q(self):
        """Test 2-qubit dot method with qargs."""

        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["III", "XXX"])
            pauli2 = PauliTable("ZY")

        with self.subTest(msg="dot 2-qubit qargs=[0, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IZY", "XYZ"])
            value = pauli1.dot(pauli2, qargs=[0, 1])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 2-qubit qargs=[1, 0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IYZ", "XZY"])
            value = pauli1.dot(pauli2, qargs=[1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 2-qubit qargs=[0, 2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["ZIY", "YXZ"])
            value = pauli1.dot(pauli2, qargs=[0, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 2-qubit qargs=[2, 0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["YIZ", "ZXY"])
            value = pauli1.dot(pauli2, qargs=[2, 0])
            self.assertEqual(value, target)

    def test_qargs_compose_3q(self):
        """Test 3-qubit compose method with qargs."""

        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["III", "XXX"])
            pauli2 = PauliTable("XYZ")

        with self.subTest(msg="compose 3-qubit qargs=None"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XYZ", "IZY"])
            value = pauli1.compose(pauli2)
            self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[0, 1, 2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XYZ", "IZY"])
            value = pauli1.compose(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[2, 1, 0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["ZYX", "YZI"])
            value = pauli1.compose(pauli2, qargs=[2, 1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[1, 0, 2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XZY", "IYZ"])
            value = pauli1.compose(pauli2, qargs=[1, 0, 2])
            self.assertEqual(value, target)

    def test_qargs_dot_3q(self):
        """Test 3-qubit dot method with qargs."""

        with self.assertWarns(DeprecationWarning):
            pauli1 = PauliTable.from_labels(["III", "XXX"])
            pauli2 = PauliTable("XYZ")

        with self.subTest(msg="dot 3-qubit qargs=None"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XYZ", "IZY"])
            value = pauli1.dot(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[0, 1, 2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XYZ", "IZY"])
            value = pauli1.dot(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[2, 1, 0]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["ZYX", "YZI"])
            value = pauli1.dot(pauli2, qargs=[2, 1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[1, 0, 2]"):
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["XZY", "IYZ"])
            value = pauli1.dot(pauli2, qargs=[1, 0, 2])
            self.assertEqual(value, target)


class TestPauliTableMethods(QiskitTestCase):
    """Tests for PauliTable utility methods class."""

    def test_sort(self):
        """Test sort method."""
        with self.subTest(msg="1 qubit standard order"):
            unsrt = ["X", "Z", "I", "Y", "X", "Z"]
            srt = ["I", "X", "X", "Y", "Z", "Z"]
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(srt)
                value = PauliTable.from_labels(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit weight order"):
            unsrt = ["X", "Z", "I", "Y", "X", "Z"]
            srt = ["I", "X", "X", "Y", "Z", "Z"]
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(srt)
                value = PauliTable.from_labels(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit standard order"):
            srt = [
                "II",
                "IX",
                "IY",
                "IY",
                "XI",
                "XX",
                "XY",
                "XZ",
                "YI",
                "YX",
                "YY",
                "YZ",
                "ZI",
                "ZI",
                "ZX",
                "ZY",
                "ZZ",
                "ZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(srt)
                value = PauliTable.from_labels(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit weight order"):
            srt = [
                "II",
                "IX",
                "IX",
                "IY",
                "IZ",
                "XI",
                "YI",
                "YI",
                "ZI",
                "XX",
                "XX",
                "XY",
                "XZ",
                "YX",
                "YY",
                "YY",
                "YZ",
                "ZX",
                "ZX",
                "ZY",
                "ZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(srt)
                value = PauliTable.from_labels(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg="3 qubit standard order"):
            srt = [
                "III",
                "III",
                "IIX",
                "IIY",
                "IIZ",
                "IXI",
                "IXX",
                "IXY",
                "IXZ",
                "IYI",
                "IYX",
                "IYY",
                "IYZ",
                "IZI",
                "IZX",
                "IZY",
                "IZY",
                "IZZ",
                "XII",
                "XII",
                "XIX",
                "XIY",
                "XIZ",
                "XXI",
                "XXX",
                "XXY",
                "XXZ",
                "XYI",
                "XYX",
                "XYY",
                "XYZ",
                "XYZ",
                "XZI",
                "XZX",
                "XZY",
                "XZZ",
                "YII",
                "YIX",
                "YIY",
                "YIZ",
                "YXI",
                "YXX",
                "YXY",
                "YXZ",
                "YXZ",
                "YYI",
                "YYX",
                "YYX",
                "YYY",
                "YYZ",
                "YZI",
                "YZX",
                "YZY",
                "YZZ",
                "ZII",
                "ZIX",
                "ZIY",
                "ZIZ",
                "ZXI",
                "ZXX",
                "ZXX",
                "ZXY",
                "ZXZ",
                "ZYI",
                "ZYI",
                "ZYX",
                "ZYY",
                "ZYZ",
                "ZZI",
                "ZZX",
                "ZZY",
                "ZZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(srt)
                value = PauliTable.from_labels(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="3 qubit weight order"):
            srt = [
                "III",
                "IIX",
                "IIY",
                "IIZ",
                "IXI",
                "IYI",
                "IZI",
                "XII",
                "YII",
                "ZII",
                "IXX",
                "IXY",
                "IXZ",
                "IYX",
                "IYY",
                "IYZ",
                "IZX",
                "IZY",
                "IZZ",
                "XIX",
                "XIY",
                "XIZ",
                "XXI",
                "XYI",
                "XZI",
                "XZI",
                "YIX",
                "YIY",
                "YIZ",
                "YXI",
                "YYI",
                "YZI",
                "YZI",
                "ZIX",
                "ZIY",
                "ZIZ",
                "ZXI",
                "ZYI",
                "ZZI",
                "ZZI",
                "XXX",
                "XXY",
                "XXZ",
                "XYX",
                "XYY",
                "XYZ",
                "XZX",
                "XZY",
                "XZZ",
                "YXX",
                "YXY",
                "YXZ",
                "YYX",
                "YYY",
                "YYZ",
                "YZX",
                "YZY",
                "YZZ",
                "ZXX",
                "ZXY",
                "ZXZ",
                "ZYX",
                "ZYY",
                "ZYZ",
                "ZZX",
                "ZZY",
                "ZZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(srt)
                value = PauliTable.from_labels(unsrt).sort(weight=True)
            self.assertEqual(target, value)

    def test_unique(self):
        """Test unique method."""
        with self.subTest(msg="1 qubit"):
            labels = ["X", "Z", "X", "X", "I", "Y", "I", "X", "Z", "Z", "X", "I"]
            unique = ["X", "Z", "I", "Y"]
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(unique)
                value = PauliTable.from_labels(labels).unique()
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit"):
            labels = ["XX", "IX", "XX", "II", "IZ", "ZI", "YX", "YX", "ZZ", "IX", "XI"]
            unique = ["XX", "IX", "II", "IZ", "ZI", "YX", "ZZ", "XI"]
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(unique)
                value = PauliTable.from_labels(labels).unique()
            self.assertEqual(target, value)

        with self.subTest(msg="10 qubit"):
            labels = [10 * "X", 10 * "I", 10 * "X"]
            unique = [10 * "X", 10 * "I"]
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(unique)
                value = PauliTable.from_labels(labels).unique()
            self.assertEqual(target, value)

    def test_delete(self):
        """Test delete method."""
        with self.subTest(msg="single row"):
            for j in range(1, 6):
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable.from_labels([j * "X", j * "Y"])
                    self.assertEqual(pauli.delete(0), PauliTable(j * "Y"))
                    self.assertEqual(pauli.delete(1), PauliTable(j * "X"))

        with self.subTest(msg="multiple rows"):
            for j in range(1, 6):
                with self.assertWarns(DeprecationWarning):
                    pauli = PauliTable.from_labels([j * "X", j * "Y", j * "Z"])
                    self.assertEqual(pauli.delete([0, 2]), PauliTable(j * "Y"))
                    self.assertEqual(pauli.delete([1, 2]), PauliTable(j * "X"))
                    self.assertEqual(pauli.delete([0, 1]), PauliTable(j * "Z"))

        with self.subTest(msg="single qubit"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["IIX", "IYI", "ZII"])
            value = pauli.delete(0, qubit=True)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["II", "IY", "ZI"])
            self.assertEqual(value, target)
            value = pauli.delete(1, qubit=True)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IX", "II", "ZI"])
            self.assertEqual(value, target)
            value = pauli.delete(2, qubit=True)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["IX", "YI", "II"])
            self.assertEqual(value, target)

        with self.subTest(msg="multiple qubits"):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable.from_labels(["IIX", "IYI", "ZII"])
            value = pauli.delete([0, 1], qubit=True)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["I", "I", "Z"])
            self.assertEqual(value, target)
            value = pauli.delete([1, 2], qubit=True)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["X", "I", "I"])
            self.assertEqual(value, target)
            value = pauli.delete([0, 2], qubit=True)
            with self.assertWarns(DeprecationWarning):
                target = PauliTable.from_labels(["I", "Y", "I"])
            self.assertEqual(value, target)

    def test_insert(self):
        """Test insert method."""
        # Insert single row
        for j in range(1, 10):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable(j * "X")
                target0 = PauliTable.from_labels([j * "I", j * "X"])
                target1 = PauliTable.from_labels([j * "X", j * "I"])

            with self.subTest(msg=f"single row from str ({j})"):
                value0 = pauli.insert(0, j * "I")
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, j * "I")
                self.assertEqual(value1, target1)

            with self.subTest(msg=f"single row from PauliTable ({j})"):
                with self.assertWarns(DeprecationWarning):
                    value0 = pauli.insert(0, PauliTable(j * "I"))
                self.assertEqual(value0, target0)
                with self.assertWarns(DeprecationWarning):
                    value1 = pauli.insert(1, PauliTable(j * "I"))
                self.assertEqual(value1, target1)

            with self.subTest(msg=f"single row from array ({j})"):
                with self.assertWarns(DeprecationWarning):
                    value0 = pauli.insert(0, PauliTable(j * "I").array)
                self.assertEqual(value0, target0)
                with self.assertWarns(DeprecationWarning):
                    value1 = pauli.insert(1, PauliTable(j * "I").array)
                self.assertEqual(value1, target1)

        # Insert multiple rows
        for j in range(1, 10):
            with self.assertWarns(DeprecationWarning):
                pauli = PauliTable(j * "X")
                insert = PauliTable.from_labels([j * "I", j * "Y", j * "Z"])
            target0 = insert + pauli
            target1 = pauli + insert

            with self.subTest(msg=f"multiple-rows from PauliTable ({j})"):
                value0 = pauli.insert(0, insert)
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, insert)
                self.assertEqual(value1, target1)

            with self.subTest(msg=f"multiple-rows from array ({j})"):
                value0 = pauli.insert(0, insert.array)
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, insert.array)
                self.assertEqual(value1, target1)

        # Insert single column
        pauli = PauliTable.from_labels(["X", "Y", "Z"])
        for i in ["I", "X", "Y", "Z"]:
            with self.assertWarns(DeprecationWarning):
                target0 = PauliTable.from_labels(["X" + i, "Y" + i, "Z" + i])
                target1 = PauliTable.from_labels([i + "X", i + "Y", i + "Z"])

            with self.subTest(msg="single-column single-val from str"):
                value = pauli.insert(0, i, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, i, qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg="single-column single-val from PauliTable"):
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(0, PauliTable(i), qubit=True)
                self.assertEqual(value, target0)
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(1, PauliTable(i), qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg="single-column single-val from array"):
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(0, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target0)
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(1, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target1)

        # Insert single column with multiple values
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["X", "Y", "Z"])
        for i in [("I", "X", "Y"), ("X", "Y", "Z"), ("Y", "Z", "I")]:
            with self.assertWarns(DeprecationWarning):
                target0 = PauliTable.from_labels(["X" + i[0], "Y" + i[1], "Z" + i[2]])
                target1 = PauliTable.from_labels([i[0] + "X", i[1] + "Y", i[2] + "Z"])

            with self.subTest(msg="single-column multiple-vals from PauliTable"):
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(0, PauliTable.from_labels(i), qubit=True)
                self.assertEqual(value, target0)
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(1, PauliTable.from_labels(i), qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg="single-column multiple-vals from array"):
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(0, PauliTable.from_labels(i).array, qubit=True)
                self.assertEqual(value, target0)
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(1, PauliTable.from_labels(i).array, qubit=True)
                self.assertEqual(value, target1)

        # Insert multiple columns from single
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["X", "Y", "Z"])
        for j in range(1, 5):
            for i in [j * "I", j * "X", j * "Y", j * "Z"]:
                with self.assertWarns(DeprecationWarning):
                    target0 = PauliTable.from_labels(["X" + i, "Y" + i, "Z" + i])
                    target1 = PauliTable.from_labels([i + "X", i + "Y", i + "Z"])

            with self.subTest(msg="multiple-columns single-val from str"):
                value = pauli.insert(0, i, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, i, qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg="multiple-columns single-val from PauliTable"):
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(0, PauliTable(i), qubit=True)
                self.assertEqual(value, target0)
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(1, PauliTable(i), qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg="multiple-columns single-val from array"):
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(0, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target0)
                with self.assertWarns(DeprecationWarning):
                    value = pauli.insert(1, PauliTable(i).array, qubit=True)
                self.assertEqual(value, target1)

        # Insert multiple columns multiple row values
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["X", "Y", "Z"])
        for j in range(1, 5):
            for i in [
                (j * "I", j * "X", j * "Y"),
                (j * "X", j * "Z", j * "Y"),
                (j * "Y", j * "Z", j * "I"),
            ]:
                with self.assertWarns(DeprecationWarning):
                    target0 = PauliTable.from_labels(["X" + i[0], "Y" + i[1], "Z" + i[2]])
                    target1 = PauliTable.from_labels([i[0] + "X", i[1] + "Y", i[2] + "Z"])

                with self.subTest(msg="multiple-column multiple-vals from PauliTable"):
                    with self.assertWarns(DeprecationWarning):
                        value = pauli.insert(0, PauliTable.from_labels(i), qubit=True)
                    self.assertEqual(value, target0)
                    with self.assertWarns(DeprecationWarning):
                        value = pauli.insert(1, PauliTable.from_labels(i), qubit=True)
                    self.assertEqual(value, target1)

                with self.subTest(msg="multiple-column multiple-vals from array"):
                    with self.assertWarns(DeprecationWarning):
                        value = pauli.insert(0, PauliTable.from_labels(i).array, qubit=True)
                    self.assertEqual(value, target0)
                    with self.assertWarns(DeprecationWarning):
                        value = pauli.insert(1, PauliTable.from_labels(i).array, qubit=True)
                    self.assertEqual(value, target1)

    def test_commutes(self):
        """Test commutes method."""
        # Single qubit Pauli
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["I", "X", "Y", "Z"])
        with self.subTest(msg="commutes single-Pauli I"):
            value = list(pauli.commutes("I"))
            target = [True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli X"):
            value = list(pauli.commutes("X"))
            target = [True, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli Y"):
            value = list(pauli.commutes("Y"))
            target = [True, False, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli Z"):
            value = list(pauli.commutes("Z"))
            target = [True, False, False, True]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["II", "IX", "YI", "XY", "ZZ"])
        with self.subTest(msg="commutes single-Pauli II"):
            value = list(pauli.commutes("II"))
            target = [True, True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli IX"):
            value = list(pauli.commutes("IX"))
            target = [True, True, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli XI"):
            value = list(pauli.commutes("XI"))
            target = [True, True, False, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli YI"):
            value = list(pauli.commutes("YI"))
            target = [True, True, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli IY"):
            value = list(pauli.commutes("IY"))
            target = [True, False, True, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli XY"):
            value = list(pauli.commutes("XY"))
            target = [True, False, False, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli YX"):
            value = list(pauli.commutes("YX"))
            target = [True, True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli ZZ"):
            value = list(pauli.commutes("ZZ"))
            target = [True, False, False, True, True]
            self.assertEqual(value, target)

    def test_commutes_with_all(self):
        """Test commutes_with_all method."""
        # 1-qubit
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["I", "X", "Y", "Z"])
        with self.subTest(msg="commutes_with_all [I]"):
            value = list(pauli.commutes_with_all("I"))
            target = [0, 1, 2, 3]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [X]"):
            value = list(pauli.commutes_with_all("X"))
            target = [0, 1]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [Y]"):
            value = list(pauli.commutes_with_all("Y"))
            target = [0, 2]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [Z]"):
            value = list(pauli.commutes_with_all("Z"))
            target = [0, 3]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["II", "IX", "YI", "XY", "ZZ"])

        with self.subTest(msg="commutes_with_all [IX, YI]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["IX", "YI"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [XY, ZZ]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["XY", "ZZ"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [YX, ZZ]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["YX", "ZZ"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [XY, YX]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["XY", "YX"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [XY, IX]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["XY", "IX"])
            value = list(pauli.commutes_with_all(other))
            target = [0]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [YX, IX]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["YX", "IX"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 1, 2]
            self.assertEqual(value, target)

    def test_anticommutes_with_all(self):
        """Test anticommutes_with_all method."""
        # 1-qubit
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["I", "X", "Y", "Z"])
        with self.subTest(msg="anticommutes_with_all [I]"):
            value = list(pauli.anticommutes_with_all("I"))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="antianticommutes_with_all [X]"):
            value = list(pauli.anticommutes_with_all("X"))
            target = [2, 3]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [Y]"):
            value = list(pauli.anticommutes_with_all("Y"))
            target = [1, 3]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [Z]"):
            value = list(pauli.anticommutes_with_all("Z"))
            target = [1, 2]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        with self.assertWarns(DeprecationWarning):
            pauli = PauliTable.from_labels(["II", "IX", "YI", "XY", "ZZ"])

        with self.subTest(msg="anticommutes_with_all [IX, YI]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["IX", "YI"])
            value = list(pauli.anticommutes_with_all(other))
            target = [3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [XY, ZZ]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["XY", "ZZ"])
            value = list(pauli.anticommutes_with_all(other))
            target = [1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [YX, ZZ]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["YX", "ZZ"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [XY, YX]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["XY", "YX"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [XY, IX]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["XY", "IX"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [YX, IX]"):
            with self.assertWarns(DeprecationWarning):
                other = PauliTable.from_labels(["YX", "IX"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)


if __name__ == "__main__":
    unittest.main()
