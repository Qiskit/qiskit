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

"""Tests for StabilizerTable class."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from qiskit import QiskitError
from qiskit.quantum_info.operators.symplectic import PauliTable, StabilizerTable
from qiskit.test import QiskitTestCase


def stab_mat(label):
    """Return stabilizer matrix from a stabilizer label"""
    mat = np.eye(1, dtype=complex)
    if label[0] == "-":
        mat *= -1
    if label[0] in ["-", "+"]:
        label = label[1:]
    for i in label:
        if i == "I":
            mat = np.kron(mat, np.eye(2))
        elif i == "X":
            mat = np.kron(mat, np.array([[0, 1], [1, 0]]))
        elif i == "Y":
            mat = np.kron(mat, np.array([[0, 1], [-1, 0]]))
        elif i == "Z":
            mat = np.kron(mat, np.array([[1, 0], [0, -1]]))
        else:
            raise QiskitError(f"Invalid stabilizer string {i}")
    return mat


class TestStabilizerTableInit(QiskitTestCase):
    """Tests for StabilizerTable initialization."""

    def test_array_init(self):
        """Test array initialization."""

        with self.subTest(msg="bool array"):
            target = np.array([[False, False], [True, True]])
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable(target)._array
            self.assertTrue(np.all(value == target))

        with self.subTest(msg="bool array no copy"):
            target = np.array([[False, True], [True, True]])
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable(target)._array
            value[0, 0] = not value[0, 0]
            self.assertTrue(np.all(value == target))

        with self.subTest(msg="bool array raises"):
            array = np.array([[False, False, False], [True, True, True]])
            with self.assertWarns(DeprecationWarning):
                self.assertRaises(QiskitError, StabilizerTable, array)

    def test_vector_init(self):
        """Test vector initialization."""

        with self.subTest(msg="bool vector"):
            target = np.array([False, False, False, False])
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable(target)._array
            self.assertTrue(np.all(value == target))

        with self.subTest(msg="bool vector no copy"):
            target = np.array([False, True, True, False])
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable(target)._array
            value[0, 0] = not value[0, 0]
            self.assertTrue(np.all(value == target))

    def test_string_init(self):
        """Test string initialization."""

        with self.subTest(msg='str init "I"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("I")._array
            target = np.array([[False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "X"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("X")._array
            target = np.array([[True, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "Y"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("Y")._array
            target = np.array([[True, True]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "Z"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("Z")._array
            target = np.array([[False, True]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "IX"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("IX")._array
            target = np.array([[True, False, False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "XI"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("XI")._array
            target = np.array([[False, True, False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "YZ"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("YZ")._array
            target = np.array([[False, True, True, True]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

        with self.subTest(msg='str init "XIZ"'):
            with self.assertWarns(DeprecationWarning):
                value = StabilizerTable("XIZ")._array
            target = np.array([[False, False, True, True, False, False]], dtype=bool)
            self.assertTrue(np.all(np.array(value == target)))

    def test_table_init(self):
        """Test StabilizerTable initialization."""

        with self.subTest(msg="StabilizerTable"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["XI", "IX", "IZ"])
                value = StabilizerTable(target)
                self.assertEqual(value, target)

        with self.subTest(msg="StabilizerTable no copy"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["XI", "IX", "IZ"])
                value = StabilizerTable(target)
                value[0] = "II"
                self.assertEqual(value, target)


class TestStabilizerTableProperties(QiskitTestCase):
    """Tests for StabilizerTable properties."""

    def test_array_property(self):
        """Test array property"""

        with self.subTest(msg="array"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable("II")
            array = np.zeros([2, 4], dtype=bool)
            self.assertTrue(np.all(stab.array == array))

        with self.subTest(msg="set array"):

            def set_array():
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable("XXX")
                stab.array = np.eye(4)
                return stab

            self.assertRaises(Exception, set_array)

    def test_x_property(self):
        """Test X property"""

        with self.subTest(msg="X"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["XI", "IZ", "YY"])
            array = np.array([[False, True], [False, False], [True, True]], dtype=bool)
            self.assertTrue(np.all(stab.X == array))

        with self.subTest(msg="set X"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["XI", "IZ"])
            val = np.array([[False, False], [True, True]], dtype=bool)
            stab.X = val
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(stab, StabilizerTable.from_labels(["II", "XY"]))

        with self.subTest(msg="set X raises"):

            def set_x():
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable.from_labels(["XI", "IZ"])
                val = np.array([[False, False, False], [True, True, True]], dtype=bool)
                stab.X = val
                return stab

            self.assertRaises(Exception, set_x)

    def test_z_property(self):
        """Test Z property"""
        with self.subTest(msg="Z"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["XI", "IZ", "YY"])
            array = np.array([[False, False], [True, False], [True, True]], dtype=bool)
            self.assertTrue(np.all(stab.Z == array))

        with self.subTest(msg="set Z"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["XI", "IZ"])
            val = np.array([[False, False], [True, True]], dtype=bool)
            stab.Z = val
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(stab, StabilizerTable.from_labels(["XI", "ZZ"]))

        with self.subTest(msg="set Z raises"):

            def set_z():
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable.from_labels(["XI", "IZ"])
                val = np.array([[False, False, False], [True, True, True]], dtype=bool)
                stab.Z = val
                return stab

            self.assertRaises(Exception, set_z)

    def test_shape_property(self):
        """Test shape property"""
        shape = (3, 8)
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable(np.zeros(shape))
        self.assertEqual(stab.shape, shape)

    def test_size_property(self):
        """Test size property"""
        with self.subTest(msg="size"):
            for j in range(1, 10):
                shape = (j, 8)
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable(np.zeros(shape))
                self.assertEqual(stab.size, j)

    def test_num_qubits_property(self):
        """Test num_qubits property"""
        with self.subTest(msg="num_qubits"):
            for j in range(1, 10):
                shape = (5, 2 * j)
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable(np.zeros(shape))
                self.assertEqual(stab.num_qubits, j)

    def test_phase_property(self):
        """Test phase property"""
        with self.subTest(msg="phase"):
            phase = np.array([False, True, True, False])
            array = np.eye(4, dtype=bool)
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(array, phase)
            self.assertTrue(np.all(stab.phase == phase))

        with self.subTest(msg="set phase"):
            phase = np.array([False, True, True, False])
            array = np.eye(4, dtype=bool)
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(array)
            stab.phase = phase
            self.assertTrue(np.all(stab.phase == phase))

        with self.subTest(msg="set phase raises"):
            phase = np.array([False, True, False])
            array = np.eye(4, dtype=bool)
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(array)

            def set_phase_raise():
                """Raise exception"""
                stab.phase = phase

            self.assertRaises(ValueError, set_phase_raise)

    def test_pauli_property(self):
        """Test pauli property"""
        with self.subTest(msg="pauli"):
            phase = np.array([False, True, True, False])
            array = np.eye(4, dtype=bool)
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(array, phase)
                pauli = PauliTable(array)
                self.assertEqual(stab.pauli, pauli)

        with self.subTest(msg="set pauli"):
            phase = np.array([False, True, True, False])
            array = np.zeros((4, 4), dtype=bool)
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(array, phase)
                pauli = PauliTable(np.eye(4, dtype=bool))
            stab.pauli = pauli
            self.assertTrue(np.all(stab.array == pauli.array))
            self.assertTrue(np.all(stab.phase == phase))

        with self.subTest(msg="set pauli"):
            phase = np.array([False, True, True, False])
            array = np.zeros((4, 4), dtype=bool)
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(array, phase)
                pauli = PauliTable(np.eye(4, dtype=bool)[1:])

            def set_pauli_raise():
                """Raise exception"""
                stab.pauli = pauli

            self.assertRaises(ValueError, set_pauli_raise)

    def test_eq(self):
        """Test __eq__ method."""
        with self.assertWarns(DeprecationWarning):
            stab1 = StabilizerTable.from_labels(["II", "XI"])
            stab2 = StabilizerTable.from_labels(["XI", "II"])
            self.assertEqual(stab1, stab1)
            self.assertNotEqual(stab1, stab2)

    def test_len_methods(self):
        """Test __len__ method."""
        for j in range(1, 10):
            labels = j * ["XX"]
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(labels)
            self.assertEqual(len(stab), j)

    def test_add_methods(self):
        """Test __add__ method."""
        labels1 = ["+XXI", "-IXX"]
        labels2 = ["+XXI", "-ZZI", "+ZYZ"]
        with self.assertWarns(DeprecationWarning):
            stab1 = StabilizerTable.from_labels(labels1)
            stab2 = StabilizerTable.from_labels(labels2)
            target = StabilizerTable.from_labels(labels1 + labels2)
            self.assertEqual(target, stab1 + stab2)

    def test_add_qargs(self):
        """Test add method with qargs."""
        with self.assertWarns(DeprecationWarning):
            stab1 = StabilizerTable.from_labels(["+IIII", "-YYYY"])
            stab2 = StabilizerTable.from_labels(["-XY", "+YZ"])

        with self.subTest(msg="qargs=[0, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["+IIII", "-YYYY", "-IIXY", "+IIYZ"])
                self.assertEqual(stab1 + stab2([0, 1]), target)

        with self.subTest(msg="qargs=[0, 3]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["+IIII", "-YYYY", "-XIIY", "+YIIZ"])
                self.assertEqual(stab1 + stab2([0, 3]), target)

        with self.subTest(msg="qargs=[2, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["+IIII", "-YYYY", "-IYXI", "+IZYI"])
                self.assertEqual(stab1 + stab2([2, 1]), target)

        with self.subTest(msg="qargs=[3, 1]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["+IIII", "-YYYY", "-YIXI", "+ZIYI"])
                self.assertEqual(stab1 + stab2([3, 1]), target)

    def test_getitem_methods(self):
        """Test __getitem__ method."""
        with self.subTest(msg="__getitem__ single"):
            labels = ["+XI", "-IY"]
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(labels)
                self.assertEqual(stab[0], StabilizerTable(labels[0]))
                self.assertEqual(stab[1], StabilizerTable(labels[1]))

        with self.subTest(msg="__getitem__ array"):
            labels = np.array(["+XI", "-IY", "+IZ", "-XY", "+ZX"])
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(labels)
            inds = [0, 3]
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(stab[inds], StabilizerTable.from_labels(labels[inds]))
            inds = np.array([4, 1])
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(stab[inds], StabilizerTable.from_labels(labels[inds]))

        with self.subTest(msg="__getitem__ slice"):
            labels = np.array(["+XI", "-IY", "+IZ", "-XY", "+ZX"])
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(labels)
                self.assertEqual(stab[:], stab)
                self.assertEqual(stab[1:3], StabilizerTable.from_labels(labels[1:3]))

    def test_setitem_methods(self):
        """Test __setitem__ method."""
        with self.subTest(msg="__setitem__ single"):
            labels = ["+XI", "IY"]
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["+XI", "IY"])
                stab[0] = "+II"
                self.assertEqual(stab[0], StabilizerTable("+II"))
                stab[1] = "-XX"
                self.assertEqual(stab[1], StabilizerTable("-XX"))

            def raises_single():
                # Wrong size Pauli
                stab[0] = "+XXX"

            self.assertRaises(Exception, raises_single)

        with self.subTest(msg="__setitem__ array"):
            labels = np.array(["+XI", "-IY", "+IZ"])
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(labels)
                target = StabilizerTable.from_labels(["+II", "-ZZ"])
            inds = [2, 0]
            stab[inds] = target
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(stab[inds], target)

            def raises_array():
                with self.assertWarns(DeprecationWarning):
                    stab[inds] = StabilizerTable.from_labels(["+YY", "-ZZ", "+XX"])

            self.assertRaises(Exception, raises_array)

        with self.subTest(msg="__setitem__ slice"):
            labels = np.array(5 * ["+III"])
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(labels)
                target = StabilizerTable.from_labels(5 * ["-XXX"])
            stab[:] = target
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(stab[:], target)
                target = StabilizerTable.from_labels(2 * ["+ZZZ"])
                stab[1:3] = target
                self.assertEqual(stab[1:3], target)


class TestStabilizerTableLabels(QiskitTestCase):
    """Tests for StabilizerTable label converions."""

    def test_from_labels_1q(self):
        """Test 1-qubit from_labels method."""
        labels = ["I", "X", "Y", "Z", "+I", "+X", "+Y", "+Z", "-I", "-X", "-Y", "-Z"]
        array = np.vstack(
            3 * [np.array([[False, False], [True, False], [True, True], [False, True]], dtype=bool)]
        )
        phase = np.array(8 * [False] + 4 * [True], dtype=bool)
        with self.assertWarns(DeprecationWarning):
            target = StabilizerTable(array, phase)
            value = StabilizerTable.from_labels(labels)
            self.assertEqual(target, value)

    def test_from_labels_2q(self):
        """Test 2-qubit from_labels method."""
        labels = ["II", "-YY", "+XZ"]
        array = np.array(
            [[False, False, False, False], [True, True, True, True], [False, True, True, False]],
            dtype=bool,
        )
        phase = np.array([False, True, False])
        with self.assertWarns(DeprecationWarning):
            target = StabilizerTable(array, phase)
            value = StabilizerTable.from_labels(labels)
            self.assertEqual(target, value)

    def test_from_labels_5q(self):
        """Test 5-qubit from_labels method."""
        labels = ["IIIII", "-XXXXX", "YYYYY", "ZZZZZ"]
        array = np.array(
            [10 * [False], 5 * [True] + 5 * [False], 10 * [True], 5 * [False] + 5 * [True]],
            dtype=bool,
        )
        phase = np.array([False, True, False, False])
        with self.assertWarns(DeprecationWarning):
            target = StabilizerTable(array, phase)
            value = StabilizerTable.from_labels(labels)
            self.assertEqual(target, value)

    def test_to_labels_1q(self):
        """Test 1-qubit to_labels method."""
        array = np.vstack(
            2 * [np.array([[False, False], [True, False], [True, True], [False, True]], dtype=bool)]
        )
        phase = np.array(4 * [False] + 4 * [True], dtype=bool)
        with self.assertWarns(DeprecationWarning):
            value = StabilizerTable(array, phase).to_labels()
        target = ["+I", "+X", "+Y", "+Z", "-I", "-X", "-Y", "-Z"]
        self.assertEqual(value, target)

    def test_to_labels_1q_array(self):
        """Test 1-qubit to_labels method w/ array=True."""
        array = np.vstack(
            2 * [np.array([[False, False], [True, False], [True, True], [False, True]], dtype=bool)]
        )
        phase = np.array(4 * [False] + 4 * [True], dtype=bool)
        with self.assertWarns(DeprecationWarning):
            value = StabilizerTable(array, phase).to_labels(array=True)
        target = np.array(["+I", "+X", "+Y", "+Z", "-I", "-X", "-Y", "-Z"])
        self.assertTrue(np.all(value == target))

    def test_labels_round_trip(self):
        """Test from_labels and to_labels round trip."""
        target = ["+III", "-IXZ", "-XYI", "+ZZZ"]
        with self.assertWarns(DeprecationWarning):
            value = StabilizerTable.from_labels(target).to_labels()
        self.assertEqual(value, target)

    def test_labels_round_trip_array(self):
        """Test from_labels and to_labels round trip w/ array=True."""
        labels = ["+III", "-IXZ", "-XYI", "+ZZZ"]
        target = np.array(labels)
        with self.assertWarns(DeprecationWarning):
            value = StabilizerTable.from_labels(labels).to_labels(array=True)
        self.assertTrue(np.all(value == target))


class TestStabilizerTableMatrix(QiskitTestCase):
    """Tests for StabilizerTable matrix converions."""

    def test_to_matrix_1q(self):
        """Test 1-qubit to_matrix method."""
        labels = ["+I", "+X", "+Y", "+Z", "-I", "-X", "-Y", "-Z"]
        targets = [stab_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = StabilizerTable.from_labels(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_1q_array(self):
        """Test 1-qubit to_matrix method w/ array=True."""
        labels = ["+I", "+X", "+Y", "+Z", "-I", "-X", "-Y", "-Z"]
        target = np.array([stab_mat(i) for i in labels])
        with self.assertWarns(DeprecationWarning):
            value = StabilizerTable.from_labels(labels).to_matrix(array=True)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertTrue(np.all(value == target))

    def test_to_matrix_1q_sparse(self):
        """Test 1-qubit to_matrix method w/ sparse=True."""
        labels = ["+I", "+X", "+Y", "+Z", "-I", "-X", "-Y", "-Z"]
        targets = [stab_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = StabilizerTable.from_labels(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_2q(self):
        """Test 2-qubit to_matrix method."""
        labels = ["+IX", "-YI", "-II", "+ZZ"]
        targets = [stab_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = StabilizerTable.from_labels(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_2q_array(self):
        """Test 2-qubit to_matrix method w/ array=True."""
        labels = ["-ZZ", "-XY", "+YX", "-IZ"]
        target = np.array([stab_mat(i) for i in labels])
        with self.assertWarns(DeprecationWarning):
            value = StabilizerTable.from_labels(labels).to_matrix(array=True)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertTrue(np.all(value == target))

    def test_to_matrix_2q_sparse(self):
        """Test 2-qubit to_matrix method w/ sparse=True."""
        labels = ["+IX", "+II", "-ZY", "-YZ"]
        targets = [stab_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = StabilizerTable.from_labels(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_5q(self):
        """Test 5-qubit to_matrix method."""
        labels = ["IXIXI", "YZIXI", "IIXYZ"]
        targets = [stab_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = StabilizerTable.from_labels(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_5q_sparse(self):
        """Test 5-qubit to_matrix method w/ sparse=True."""
        labels = ["-XXXYY", "IXIZY", "-ZYXIX", "+ZXIYZ"]
        targets = [stab_mat(i) for i in labels]
        with self.assertWarns(DeprecationWarning):
            values = StabilizerTable.from_labels(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))


class TestStabilizerTableMethods(QiskitTestCase):
    """Tests for StabilizerTable methods."""

    def test_sort(self):
        """Test sort method."""
        with self.subTest(msg="1 qubit"):
            unsrt = ["X", "-Z", "I", "Y", "-X", "Z"]
            srt = ["I", "X", "-X", "Y", "-Z", "Z"]
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(srt)
                value = StabilizerTable.from_labels(unsrt).sort()
                self.assertEqual(target, value)

        with self.subTest(msg="1 qubit weight order"):
            unsrt = ["X", "-Z", "I", "Y", "-X", "Z"]
            srt = ["I", "X", "-X", "Y", "-Z", "Z"]
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(srt)
                value = StabilizerTable.from_labels(unsrt).sort(weight=True)
                self.assertEqual(target, value)

        with self.subTest(msg="2 qubit standard order"):
            srt_p = [
                "II",
                "IX",
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
                "ZX",
                "ZY",
                "ZZ",
            ]
            srt_m = ["-" + i for i in srt_p]

            unsrt_p = srt_p.copy()
            np.random.shuffle(unsrt_p)
            unsrt_m = srt_m.copy()
            np.random.shuffle(unsrt_m)

            # Sort with + cases all first in shuffled list
            srt = [val for pair in zip(srt_p, srt_m) for val in pair]
            unsrt = unsrt_p + unsrt_m
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(srt)
                value = StabilizerTable.from_labels(unsrt).sort()
                self.assertEqual(target, value)

            # Sort with - cases all first in shuffled list
            srt = [val for pair in zip(srt_m, srt_p) for val in pair]
            unsrt = unsrt_m + unsrt_p
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(srt)
                value = StabilizerTable.from_labels(unsrt).sort()
                self.assertEqual(target, value)

        with self.subTest(msg="2 qubit weight order"):
            srt_p = [
                "II",
                "IX",
                "IY",
                "IZ",
                "XI",
                "YI",
                "ZI",
                "XX",
                "XY",
                "XZ",
                "YX",
                "YY",
                "YZ",
                "ZX",
                "ZY",
                "ZZ",
            ]
            srt_m = ["-" + i for i in srt_p]

            unsrt_p = srt_p.copy()
            np.random.shuffle(unsrt_p)
            unsrt_m = srt_m.copy()
            np.random.shuffle(unsrt_m)

            # Sort with + cases all first in shuffled list
            srt = [val for pair in zip(srt_p, srt_m) for val in pair]
            unsrt = unsrt_p + unsrt_m
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(srt)
                value = StabilizerTable.from_labels(unsrt).sort(weight=True)
                self.assertEqual(target, value)

            # Sort with - cases all first in shuffled list
            srt = [val for pair in zip(srt_m, srt_p) for val in pair]
            unsrt = unsrt_m + unsrt_p
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(srt)
                value = StabilizerTable.from_labels(unsrt).sort(weight=True)
                self.assertEqual(target, value)

    def test_unique(self):
        """Test unique method."""
        with self.subTest(msg="1 qubit"):
            labels = ["X", "Z", "-I", "-X", "X", "I", "Y", "-I", "-X", "-Z", "Z", "X", "I"]
            unique = ["X", "Z", "-I", "-X", "I", "Y", "-Z"]
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(unique)
                value = StabilizerTable.from_labels(labels).unique()
                self.assertEqual(target, value)

        with self.subTest(msg="2 qubit"):
            labels = [
                "XX",
                "IX",
                "-XX",
                "XX",
                "-IZ",
                "II",
                "IZ",
                "ZI",
                "YX",
                "YX",
                "ZZ",
                "IX",
                "XI",
            ]
            unique = ["XX", "IX", "-XX", "-IZ", "II", "IZ", "ZI", "YX", "ZZ", "XI"]
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(unique)
                value = StabilizerTable.from_labels(labels).unique()
                self.assertEqual(target, value)

        with self.subTest(msg="10 qubit"):
            labels = [10 * "X", "-" + 10 * "X", "-" + 10 * "X", 10 * "I", 10 * "X"]
            unique = [10 * "X", "-" + 10 * "X", 10 * "I"]
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(unique)
                value = StabilizerTable.from_labels(labels).unique()
                self.assertEqual(target, value)

    def test_delete(self):
        """Test delete method."""
        with self.subTest(msg="single row"):
            for j in range(1, 6):
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable.from_labels([j * "X", "-" + j * "Y"])
                    self.assertEqual(stab.delete(0), StabilizerTable("-" + j * "Y"))
                    self.assertEqual(stab.delete(1), StabilizerTable(j * "X"))

        with self.subTest(msg="multiple rows"):
            for j in range(1, 6):
                with self.assertWarns(DeprecationWarning):
                    stab = StabilizerTable.from_labels([j * "X", "-" + j * "Y", j * "Z"])
                    self.assertEqual(stab.delete([0, 2]), StabilizerTable("-" + j * "Y"))
                    self.assertEqual(stab.delete([1, 2]), StabilizerTable(j * "X"))
                    self.assertEqual(stab.delete([0, 1]), StabilizerTable(j * "Z"))

        with self.subTest(msg="single qubit"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["IIX", "IYI", "ZII"])
                value = stab.delete(0, qubit=True)
                target = StabilizerTable.from_labels(["II", "IY", "ZI"])
                self.assertEqual(value, target)
                value = stab.delete(1, qubit=True)
                target = StabilizerTable.from_labels(["IX", "II", "ZI"])
                self.assertEqual(value, target)
                value = stab.delete(2, qubit=True)
                target = StabilizerTable.from_labels(["IX", "YI", "II"])
                self.assertEqual(value, target)

        with self.subTest(msg="multiple qubits"):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable.from_labels(["IIX", "IYI", "ZII"])
                value = stab.delete([0, 1], qubit=True)
                target = StabilizerTable.from_labels(["I", "I", "Z"])
                self.assertEqual(value, target)
                value = stab.delete([1, 2], qubit=True)
                target = StabilizerTable.from_labels(["X", "I", "I"])
                self.assertEqual(value, target)
                value = stab.delete([0, 2], qubit=True)
                target = StabilizerTable.from_labels(["I", "Y", "I"])
                self.assertEqual(value, target)

    def test_insert(self):
        """Test insert method."""
        # Insert single row
        for j in range(1, 10):
            l_px = j * "X"
            l_mi = "-" + j * "I"
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(l_px)
                target0 = StabilizerTable.from_labels([l_mi, l_px])
                target1 = StabilizerTable.from_labels([l_px, l_mi])

            with self.subTest(msg=f"single row from str ({j})"):
                with self.assertWarns(DeprecationWarning):
                    value0 = stab.insert(0, l_mi)
                    self.assertEqual(value0, target0)
                    value1 = stab.insert(1, l_mi)
                    self.assertEqual(value1, target1)

            with self.subTest(msg=f"single row from StabilizerTable ({j})"):
                with self.assertWarns(DeprecationWarning):
                    value0 = stab.insert(0, StabilizerTable(l_mi))
                    self.assertEqual(value0, target0)
                    value1 = stab.insert(1, StabilizerTable(l_mi))
                    self.assertEqual(value1, target1)

        # Insert multiple rows
        for j in range(1, 10):
            with self.assertWarns(DeprecationWarning):
                stab = StabilizerTable(j * "X")
                insert = StabilizerTable.from_labels(["-" + j * "I", j * "Y", "-" + j * "Z"])
                target0 = insert + stab
                target1 = stab + insert

            with self.subTest(msg=f"multiple-rows from StabilizerTable ({j})"):
                with self.assertWarns(DeprecationWarning):
                    value0 = stab.insert(0, insert)
                    self.assertEqual(value0, target0)
                    value1 = stab.insert(1, insert)
                    self.assertEqual(value1, target1)

        # Insert single column
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(["X", "Y", "Z"])
        for sgn in ["+", "-"]:
            for i in ["I", "X", "Y", "Z"]:
                with self.assertWarns(DeprecationWarning):
                    target0 = StabilizerTable.from_labels(
                        [sgn + "X" + i, sgn + "Y" + i, sgn + "Z" + i]
                    )
                    target1 = StabilizerTable.from_labels(
                        [sgn + i + "X", sgn + i + "Y", sgn + i + "Z"]
                    )

                with self.subTest(msg=f"single-column single-val from str {sgn + i}"):
                    with self.assertWarns(DeprecationWarning):
                        value = stab.insert(0, sgn + i, qubit=True)
                        self.assertEqual(value, target0)
                        value = stab.insert(1, sgn + i, qubit=True)
                        self.assertEqual(value, target1)

                with self.subTest(msg=f"single-column single-val from StabilizerTable {sgn + i}"):
                    with self.assertWarns(DeprecationWarning):
                        value = stab.insert(0, StabilizerTable(sgn + i), qubit=True)
                        self.assertEqual(value, target0)
                        value = stab.insert(1, StabilizerTable(sgn + i), qubit=True)
                        self.assertEqual(value, target1)

        # Insert single column with multiple values
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(["X", "Y", "Z"])
        for i in [("I", "X", "Y"), ("X", "Y", "Z"), ("Y", "Z", "I")]:
            with self.assertWarns(DeprecationWarning):
                target0 = StabilizerTable.from_labels(["X" + i[0], "Y" + i[1], "Z" + i[2]])
                target1 = StabilizerTable.from_labels([i[0] + "X", i[1] + "Y", i[2] + "Z"])

            with self.subTest(msg="single-column multiple-vals from StabilizerTable"):
                with self.assertWarns(DeprecationWarning):
                    value = stab.insert(0, StabilizerTable.from_labels(i), qubit=True)
                    self.assertEqual(value, target0)
                    value = stab.insert(1, StabilizerTable.from_labels(i), qubit=True)
                    self.assertEqual(value, target1)

            with self.subTest(msg="single-column multiple-vals from array"):
                with self.assertWarns(DeprecationWarning):
                    value = stab.insert(0, StabilizerTable.from_labels(i).array, qubit=True)
                    self.assertEqual(value, target0)
                    value = stab.insert(1, StabilizerTable.from_labels(i).array, qubit=True)
                    self.assertEqual(value, target1)

        # Insert multiple columns from single
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(["X", "Y", "Z"])
        for j in range(1, 5):
            for i in [j * "I", j * "X", j * "Y", j * "Z"]:
                with self.assertWarns(DeprecationWarning):
                    target0 = StabilizerTable.from_labels(["X" + i, "Y" + i, "Z" + i])
                    target1 = StabilizerTable.from_labels([i + "X", i + "Y", i + "Z"])

            with self.subTest(msg="multiple-columns single-val from str"):
                with self.assertWarns(DeprecationWarning):
                    value = stab.insert(0, i, qubit=True)
                    self.assertEqual(value, target0)
                    value = stab.insert(1, i, qubit=True)
                    self.assertEqual(value, target1)

            with self.subTest(msg="multiple-columns single-val from StabilizerTable"):
                with self.assertWarns(DeprecationWarning):
                    value = stab.insert(0, StabilizerTable(i), qubit=True)
                    self.assertEqual(value, target0)
                    value = stab.insert(1, StabilizerTable(i), qubit=True)
                    self.assertEqual(value, target1)

            with self.subTest(msg="multiple-columns single-val from array"):
                with self.assertWarns(DeprecationWarning):
                    value = stab.insert(0, StabilizerTable(i).array, qubit=True)
                    self.assertEqual(value, target0)
                    value = stab.insert(1, StabilizerTable(i).array, qubit=True)
                    self.assertEqual(value, target1)

        # Insert multiple columns multiple row values
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(["X", "Y", "Z"])
        for j in range(1, 5):
            for i in [
                (j * "I", j * "X", j * "Y"),
                (j * "X", j * "Z", j * "Y"),
                (j * "Y", j * "Z", j * "I"),
            ]:
                with self.assertWarns(DeprecationWarning):
                    target0 = StabilizerTable.from_labels(["X" + i[0], "Y" + i[1], "Z" + i[2]])
                    target1 = StabilizerTable.from_labels([i[0] + "X", i[1] + "Y", i[2] + "Z"])

                with self.subTest(msg="multiple-column multiple-vals from StabilizerTable"):
                    with self.assertWarns(DeprecationWarning):
                        value = stab.insert(0, StabilizerTable.from_labels(i), qubit=True)
                        self.assertEqual(value, target0)
                        value = stab.insert(1, StabilizerTable.from_labels(i), qubit=True)
                        self.assertEqual(value, target1)

                with self.subTest(msg="multiple-column multiple-vals from array"):
                    with self.assertWarns(DeprecationWarning):
                        value = stab.insert(0, StabilizerTable.from_labels(i).array, qubit=True)
                        self.assertEqual(value, target0)
                        value = stab.insert(1, StabilizerTable.from_labels(i).array, qubit=True)
                        self.assertEqual(value, target1)

    def test_iteration(self):
        """Test iteration methods."""

        labels = ["+III", "+IXI", "-IYY", "+YIZ", "-ZIZ", "+XYZ", "-III"]
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(labels)

        with self.subTest(msg="enumerate"):
            with self.assertWarns(DeprecationWarning):
                for idx, i in enumerate(stab):
                    self.assertEqual(i, StabilizerTable(labels[idx]))

        with self.subTest(msg="iter"):
            with self.assertWarns(DeprecationWarning):
                for idx, i in enumerate(iter(stab)):
                    self.assertEqual(i, StabilizerTable(labels[idx]))

        with self.subTest(msg="zip"):
            with self.assertWarns(DeprecationWarning):
                for label, i in zip(labels, stab):
                    self.assertEqual(i, StabilizerTable(label))

        with self.subTest(msg="label_iter"):
            for idx, i in enumerate(stab.label_iter()):
                self.assertEqual(i, labels[idx])

        with self.subTest(msg="matrix_iter (dense)"):
            for idx, i in enumerate(stab.matrix_iter()):
                self.assertTrue(np.all(i == stab_mat(labels[idx])))

        with self.subTest(msg="matrix_iter (sparse)"):
            for idx, i in enumerate(stab.matrix_iter(sparse=True)):
                self.assertTrue(isinstance(i, csr_matrix))
                self.assertTrue(np.all(i.toarray() == stab_mat(labels[idx])))

    def test_tensor(self):
        """Test tensor method."""
        labels1 = ["-XX", "YY"]
        labels2 = ["III", "-ZZZ"]
        with self.assertWarns(DeprecationWarning):
            stab1 = StabilizerTable.from_labels(labels1)
            stab2 = StabilizerTable.from_labels(labels2)

            target = StabilizerTable.from_labels(["-XXIII", "XXZZZ", "YYIII", "-YYZZZ"])
            value = stab1.tensor(stab2)
            self.assertEqual(value, target)

    def test_expand(self):
        """Test expand method."""
        labels1 = ["-XX", "YY"]
        labels2 = ["III", "-ZZZ"]
        with self.assertWarns(DeprecationWarning):
            stab1 = StabilizerTable.from_labels(labels1)
            stab2 = StabilizerTable.from_labels(labels2)

            target = StabilizerTable.from_labels(["-IIIXX", "IIIYY", "ZZZXX", "-ZZZYY"])
            value = stab1.expand(stab2)
            self.assertEqual(value, target)

    def test_compose(self):
        """Test compose and dot methods."""

        # Test single qubit Pauli dot products
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(["I", "X", "Y", "Z"])

        # Test single qubit Pauli dot products
        with self.assertWarns(DeprecationWarning):
            stab = StabilizerTable.from_labels(["I", "X", "Y", "Z", "-I", "-X", "-Y", "-Z"])

        with self.subTest(msg="dot single I"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("I")
                target = StabilizerTable.from_labels(["I", "X", "Y", "Z", "-I", "-X", "-Y", "-Z"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single -I"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("-I")
                target = StabilizerTable.from_labels(["-I", "-X", "-Y", "-Z", "I", "X", "Y", "Z"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single I"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("I")
                target = StabilizerTable.from_labels(["I", "X", "Y", "Z", "-I", "-X", "-Y", "-Z"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single -I"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("-I")
                target = StabilizerTable.from_labels(["-I", "-X", "-Y", "-Z", "I", "X", "Y", "Z"])
                self.assertEqual(target, value)

        with self.subTest(msg="compose single X"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("X")
                target = StabilizerTable.from_labels(["X", "I", "-Z", "Y", "-X", "-I", "Z", "-Y"])
                self.assertEqual(target, value)

        with self.subTest(msg="compose single -X"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("-X")
                target = StabilizerTable.from_labels(["-X", "-I", "Z", "-Y", "X", "I", "-Z", "Y"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single X"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("X")
                target = StabilizerTable.from_labels(["X", "I", "Z", "-Y", "-X", "-I", "-Z", "Y"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single -X"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("-X")
                target = StabilizerTable.from_labels(["-X", "-I", "-Z", "Y", "X", "I", "Z", "-Y"])
                self.assertEqual(target, value)

        with self.subTest(msg="compose single Y"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("Y")
                target = StabilizerTable.from_labels(["Y", "Z", "-I", "-X", "-Y", "-Z", "I", "X"])
                self.assertEqual(target, value)

        with self.subTest(msg="compose single -Y"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("-Y")
                target = StabilizerTable.from_labels(["-Y", "-Z", "I", "X", "Y", "Z", "-I", "-X"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single Y"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("Y")
                target = StabilizerTable.from_labels(["Y", "-Z", "-I", "X", "-Y", "Z", "I", "-X"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single -Y"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("-Y")
                target = StabilizerTable.from_labels(["-Y", "Z", "I", "-X", "Y", "-Z", "-I", "X"])
                self.assertEqual(target, value)

        with self.subTest(msg="compose single Z"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("Z")
                target = StabilizerTable.from_labels(["Z", "-Y", "X", "I", "-Z", "Y", "-X", "-I"])
                self.assertEqual(target, value)

        with self.subTest(msg="compose single -Z"):
            with self.assertWarns(DeprecationWarning):
                value = stab.compose("-Z")
                target = StabilizerTable.from_labels(["-Z", "Y", "-X", "-I", "Z", "-Y", "X", "I"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single Z"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("Z")
                target = StabilizerTable.from_labels(["Z", "Y", "-X", "I", "-Z", "-Y", "X", "-I"])
                self.assertEqual(target, value)

        with self.subTest(msg="dot single -Z"):
            with self.assertWarns(DeprecationWarning):
                value = stab.dot("-Z")
                target = StabilizerTable.from_labels(["-Z", "-Y", "X", "-I", "Z", "Y", "-X", "I"])
                self.assertEqual(target, value)

    def test_compose_qargs(self):
        """Test compose and dot methods with qargs."""

        # Dot product with qargs
        with self.assertWarns(DeprecationWarning):
            stab1 = StabilizerTable.from_labels(["III", "-XXX", "YYY", "-ZZZ"])

        # 1-qubit qargs
        with self.assertWarns(DeprecationWarning):
            stab2 = StabilizerTable("-Z")

        with self.subTest(msg="dot 1-qubit qargs=[0]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["-IIZ", "XXY", "YYX", "ZZI"])
                value = stab1.dot(stab2, qargs=[0])
                self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[0]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["-IIZ", "-XXY", "-YYX", "ZZI"])
                value = stab1.compose(stab2, qargs=[0])
                self.assertEqual(value, target)

        with self.subTest(msg="dot 1-qubit qargs=[1]"):
            with self.assertWarns(DeprecationWarning):
                target = StabilizerTable.from_labels(["-IZI", "XYX", "YXY", "ZIZ"])
                value = stab1.dot(stab2, qargs=[1])
                self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[1]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.compose(stab2, qargs=[1])
                target = StabilizerTable.from_labels(["-IZI", "-XYX", "-YXY", "ZIZ"])
                self.assertEqual(value, target)

        with self.assertWarns(DeprecationWarning):
            target = StabilizerTable.from_labels(["ZII", "YXX"])
        with self.subTest(msg="dot 1-qubit qargs=[2]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.dot(stab2, qargs=[2])
                target = StabilizerTable.from_labels(["-ZII", "YXX", "XYY", "IZZ"])
                self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[2]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.compose(stab2, qargs=[2])
                target = StabilizerTable.from_labels(["-ZII", "-YXX", "-XYY", "IZZ"])
                self.assertEqual(value, target)

        # 2-qubit qargs
        with self.assertWarns(DeprecationWarning):
            stab2 = StabilizerTable("-ZY")
        with self.subTest(msg="dot 2-qubit qargs=[0, 1]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.dot(stab2, qargs=[0, 1])
                target = StabilizerTable.from_labels(["-IZY", "-XYZ", "-YXI", "ZIX"])
                self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[0, 1]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.compose(stab2, qargs=[0, 1])
                target = StabilizerTable.from_labels(["-IZY", "-XYZ", "YXI", "-ZIX"])
                self.assertEqual(value, target)

        with self.assertWarns(DeprecationWarning):
            target = StabilizerTable.from_labels(["YIZ", "ZXY"])
        with self.subTest(msg="dot 2-qubit qargs=[2, 0]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.dot(stab2, qargs=[2, 0])
                target = StabilizerTable.from_labels(["-YIZ", "-ZXY", "-IYX", "XZI"])
                self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[2, 0]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.compose(stab2, qargs=[2, 0])
                target = StabilizerTable.from_labels(["-YIZ", "-ZXY", "IYX", "-XZI"])
                self.assertEqual(value, target)

        # 3-qubit qargs
        with self.assertWarns(DeprecationWarning):
            stab2 = StabilizerTable("-XYZ")

        with self.assertWarns(DeprecationWarning):
            target = StabilizerTable.from_labels(["XYZ", "IZY"])
        with self.subTest(msg="dot 3-qubit qargs=None"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.dot(stab2, qargs=[0, 1, 2])
                target = StabilizerTable.from_labels(["-XYZ", "-IZY", "-ZIX", "-YXI"])
                self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[0, 1, 2]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.dot(stab2, qargs=[0, 1, 2])
                target = StabilizerTable.from_labels(["-XYZ", "-IZY", "-ZIX", "-YXI"])
                self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[2, 1, 0]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.dot(stab2, qargs=[2, 1, 0])
                target = StabilizerTable.from_labels(["-ZYX", "-YZI", "-XIZ", "-IXY"])
                self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[2, 1, 0]"):
            with self.assertWarns(DeprecationWarning):
                value = stab1.compose(stab2, qargs=[2, 1, 0])
                target = StabilizerTable.from_labels(["-ZYX", "-YZI", "-XIZ", "-IXY"])
                self.assertEqual(value, target)


if __name__ == "__main__":
    unittest.main()
