# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Quick program to test the qi tools modules."""

import unittest
from copy import deepcopy
import numpy as np

from qiskit.quantum_info import Pauli, pauli_group
from qiskit.test import QiskitTestCase


class TestPauliAPI(QiskitTestCase):
    """Tests for Pauli class API."""

    def check(self, result):
        """checks for result to be a Pauli 'IY' """
        self.assertIsInstance(result, Pauli)
        self.assertEqual(result.num_qubits, 2)
        self.assertEqual(result.to_label(), 'IY')

    def test_ndarray_bool(self):
        """Test creation from np.bool."""
        x = np.asarray([1, 0]).astype(np.bool)
        z = np.asarray([1, 0]).astype(np.bool)
        pauli = Pauli(x=x, z=z)
        self.check(pauli)

    def test_ndarray_int(self):
        """Test creation from np.int."""
        x = np.asarray([2, 0]).astype(np.int)
        z = np.asarray([2, 0]).astype(np.int)
        pauli = Pauli(x=x, z=z)
        self.check(pauli)

    def test_list(self):
        """Test creation from lists."""
        pauli = Pauli(x=[1, 0], z=[1, 0])
        self.check(pauli)

    def test_tuple(self):
        """Test creation from tuples."""
        pauli = Pauli(x=(1, 0), z=(1, 0))
        self.check(pauli)

    def test_mix(self):
        """Test creation from tuples and list."""
        pauli = Pauli(x=(1, 0), z=[1, 0])
        self.check(pauli)


class TestPauli(QiskitTestCase):
    """Tests for Pauli class."""

    def setUp(self):
        """Setup."""
        z = np.asarray([1, 0, 1, 0]).astype(np.bool)
        x = np.asarray([1, 1, 0, 0]).astype(np.bool)
        self.ref_p = Pauli(z, x)
        self.ref_label = 'IZXY'
        self.ref_matrix = np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                                    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]])

    def test_create_from_label(self):
        """Test creation from pauli label."""
        label = 'IZXY'
        pauli = Pauli(label=label)

        self.assertEqual(pauli, self.ref_p)
        self.assertEqual(pauli.to_label(), self.ref_label)
        self.assertEqual(len(pauli), 4)

    def test_create_from_z_x(self):
        """Test creation for boolean vector."""
        self.assertEqual(self.ref_p.to_label(), 'IZXY')
        self.assertEqual(len(self.ref_p), 4)

    def test_repr(self):
        """Test __repr__."""
        p = repr(self.ref_p)
        self.assertEqual(p, "Pauli(z=[True, False, True, False], x=[True, True, False, False])")

    def test_random_pauli(self):
        """Test random pauli creation."""
        length = 4
        q = Pauli.random(length, seed=42)
        self.log.info(q)
        self.assertEqual(q.num_qubits, length)
        self.assertEqual(len(q.z), length)
        self.assertEqual(len(q.x), length)
        self.assertEqual(len(q.to_label()), length)
        self.assertEqual(len(q.to_matrix()), 2 ** length)

    def test_mul(self):
        """Test multiplication."""
        p1 = self.ref_p
        p2 = Pauli.from_label('ZXXI')
        p3 = p1 * p2
        self.assertEqual(len(p3), 4)
        self.assertEqual(p3.to_label(), 'ZYIY')

    def test_imul(self):
        """Test in-place multiplication."""
        p1 = self.ref_p
        p2 = Pauli.from_label('ZXXI')
        p3 = deepcopy(p2)
        p2 *= p1
        self.assertTrue(p2 != p3)
        self.assertEqual(p2.to_label(), 'ZYIY')

    def test_equality_equal(self):
        """Test equality operator: equal Paulis."""
        p1 = self.ref_p
        p2 = deepcopy(p1)
        self.assertTrue(p1 == p2)
        self.assertEqual(p1.to_label(), self.ref_label)
        self.assertEqual(p2.to_label(), self.ref_label)

    def test_equality_different(self):
        """Test equality operator: different Paulis."""
        p1 = self.ref_p
        p2 = deepcopy(p1)

        p2.update_z(True, 1)
        self.assertFalse(p1 == p2)
        self.assertEqual(p1.to_label(), self.ref_label)
        self.assertEqual(p2.to_label(), 'IZYY')

    def test_inequality_equal(self):
        """Test inequality operator: equal Paulis."""
        p1 = self.ref_p
        p2 = deepcopy(p1)

        self.assertFalse(p1 != p2)

    def test_inequality_different(self):
        """Test inequality operator: different Paulis."""
        p1 = self.ref_p
        p2 = deepcopy(p1)
        p2.update_x(False, 1)
        self.assertTrue(p1 != p2)
        self.assertEqual(p2.to_label(), 'IZIY')

    def test_update_z(self):
        """Test update_z method."""
        updated_z = np.asarray([0, 0, 0, 0]).astype(np.bool)
        self.ref_p.update_z(updated_z)
        np.testing.assert_equal(self.ref_p.z, np.asarray([False, False, False, False]))
        self.assertEqual(self.ref_p.to_label(), 'IIXX')

    def test_update_z_2(self):
        """Test update_z method, update partial z."""
        updated_z = np.asarray([0, 1]).astype(np.bool)
        self.ref_p.update_z(updated_z, [0, 1])
        np.testing.assert_equal(self.ref_p.z, np.asarray([False, True, True, False]))
        self.assertEqual(self.ref_p.to_label(), 'IZYX')

    def test_update_x(self):
        """Test update_x method."""
        updated_x = np.asarray([0, 1, 0, 1]).astype(np.bool)
        self.ref_p.update_x(updated_x)
        np.testing.assert_equal(self.ref_p.x, np.asarray([False, True, False, True]))
        self.assertEqual(self.ref_p.to_label(), 'XZXZ')

    def test_update_x_2(self):
        """Test update_x method, update partial x."""
        updated_x = np.asarray([0, 1]).astype(np.bool)
        self.ref_p.update_x(updated_x, [1, 2])
        np.testing.assert_equal(self.ref_p.x, np.asarray([True, False, True, False]))
        self.assertEqual(self.ref_p.to_label(), 'IYIY')

    def test_to_matrix(self):
        """Test pauli to matrix."""
        np.testing.assert_allclose(self.ref_p.to_matrix(), self.ref_matrix)

    def test_delete_qubit(self):
        """Test deleting single qubit."""
        p1 = self.ref_p
        p2 = deepcopy(p1)

        p2.delete_qubits(0)
        self.assertTrue(p1 != p2)
        self.assertEqual(len(p2), 3)
        self.assertEqual(p2.to_label(), 'IZX')

    def test_delete_qubits(self):
        """Test deleting multiple qubits."""
        p1 = self.ref_p
        p2 = deepcopy(p1)

        p2.delete_qubits([0, 2])
        self.assertTrue(p1 != p2)
        self.assertEqual(len(p2), 2)
        self.assertEqual(p2.to_label(), 'IX')

    def test_append_pauli_labels(self):
        """Test appending paulis via labels."""
        p1 = self.ref_p
        p2 = deepcopy(p1)

        p2.append_paulis(pauli_labels=['Z', 'Y', 'I'])
        self.assertTrue(p1 != p2)
        self.assertEqual(len(p2), 7)
        self.assertEqual(p2.to_label(), 'IYZ' + self.ref_label)

    def test_append_paulis(self):
        """Test appending paulis via pauli object."""
        p1 = self.ref_p
        p2 = deepcopy(p1)

        p2.append_paulis(paulis=p1)
        self.assertTrue(p1 != p2)
        self.assertEqual(len(p2), 8)
        self.assertEqual(p2.to_label(), self.ref_label + self.ref_label)

    def test_insert_pauli_labels_1(self):
        """Test inserting paulis via labels."""
        p2 = deepcopy(self.ref_p)

        p2.insert_paulis(indices=[1, 2], pauli_labels=['Y', 'I'])
        self.assertTrue(self.ref_p != p2)
        self.assertEqual(len(p2), 6)
        self.assertEqual(p2.to_label(), 'IZIXYY')

    def test_insert_pauli_labels_2(self):
        """Test inserting paulis via labels."""
        p2 = deepcopy(self.ref_p)

        p2.insert_paulis(indices=[3, 2], pauli_labels=['Y', 'I'])
        self.assertTrue(self.ref_p != p2)
        self.assertEqual(len(p2), 6)
        self.assertEqual(p2.to_label(), 'IYZIXY')

    def test_insert_paulis(self):
        """Test inserting paulis via pauli object."""
        p1 = deepcopy(self.ref_p)

        new_p = Pauli.from_label('XY')

        p1.insert_paulis(indices=[0], paulis=new_p)

        self.assertTrue(p1 != self.ref_p)
        self.assertEqual(len(p1), 6)
        self.assertEqual(p1.to_label(), self.ref_label + 'XY')

    def test_kron(self):
        """Test kron production."""
        p1 = deepcopy(self.ref_p)
        p2 = self.ref_p
        p2.kron(p1)
        self.assertTrue(p1 != p2)
        self.assertEqual(len(p2), 8)
        self.assertEqual(p2.to_label(), self.ref_label + self.ref_label)

    def test_pauli_single(self):
        """Test pauli single."""
        num_qubits = 5
        pz = Pauli.pauli_single(num_qubits, 2, 'Z')
        self.assertTrue(pz.to_label(), 'IIIZI')

        py = Pauli.pauli_single(num_qubits, 4, 'Y')
        self.assertTrue(py.to_label(), 'IYIII')

        px = Pauli.pauli_single(num_qubits, 3, 'X')
        self.assertTrue(px.to_label(), 'IIXII')

    def test_pauli_group(self):
        """Test pauli group."""
        self.log.info("Group in tensor order:")
        expected = ['III', 'XII', 'YII', 'ZII', 'IXI', 'XXI', 'YXI', 'ZXI', 'IYI', 'XYI', 'YYI',
                    'ZYI', 'IZI', 'XZI', 'YZI', 'ZZI', 'IIX', 'XIX', 'YIX', 'ZIX', 'IXX', 'XXX',
                    'YXX', 'ZXX', 'IYX', 'XYX', 'YYX', 'ZYX', 'IZX', 'XZX', 'YZX', 'ZZX', 'IIY',
                    'XIY', 'YIY', 'ZIY', 'IXY', 'XXY', 'YXY', 'ZXY', 'IYY', 'XYY', 'YYY', 'ZYY',
                    'IZY', 'XZY', 'YZY', 'ZZY', 'IIZ', 'XIZ', 'YIZ', 'ZIZ', 'IXZ', 'XXZ', 'YXZ',
                    'ZXZ', 'IYZ', 'XYZ', 'YYZ', 'ZYZ', 'IZZ', 'XZZ', 'YZZ', 'ZZZ']
        grp = pauli_group(3, case='tensor')
        for j in grp:
            self.log.info('==== j (tensor order) ====')
            self.log.info(j.to_label())
            self.assertEqual(expected.pop(0)[::-1], j.to_label())

        self.log.info("Group in weight order:")
        expected = ['III', 'XII', 'YII', 'ZII', 'IXI', 'IYI', 'IZI', 'IIX', 'IIY', 'IIZ', 'XXI',
                    'YXI', 'ZXI', 'XYI', 'YYI', 'ZYI', 'XZI', 'YZI', 'ZZI', 'XIX', 'YIX', 'ZIX',
                    'IXX', 'IYX', 'IZX', 'XIY', 'YIY', 'ZIY', 'IXY', 'IYY', 'IZY', 'XIZ', 'YIZ',
                    'ZIZ', 'IXZ', 'IYZ', 'IZZ', 'XXX', 'YXX', 'ZXX', 'XYX', 'YYX', 'ZYX', 'XZX',
                    'YZX', 'ZZX', 'XXY', 'YXY', 'ZXY', 'XYY', 'YYY', 'ZYY', 'XZY', 'YZY', 'ZZY',
                    'XXZ', 'YXZ', 'ZXZ', 'XYZ', 'YYZ', 'ZYZ', 'XZZ', 'YZZ', 'ZZZ']
        grp = pauli_group(3, case='weight')
        for j in grp:
            self.log.info('==== j (weight order) ====')
            self.log.info(j.to_label())
            self.assertEqual(expected.pop(0)[::-1], j.to_label())

    def test_sgn_prod(self):
        """Test sgn prod."""
        p1 = Pauli(np.array([False]), np.array([True]))
        p2 = Pauli(np.array([True]), np.array([True]))

        self.log.info("sign product:")
        p3, sgn = Pauli.sgn_prod(p1, p2)
        self.log.info("p1: %s", p1.to_label())
        self.log.info("p2: %s", p2.to_label())
        self.log.info("p3: %s", p3.to_label())
        self.log.info("sgn_prod(p1, p2): %s", str(sgn))
        self.assertEqual(p1.to_label(), 'X')
        self.assertEqual(p2.to_label(), 'Y')
        self.assertEqual(p3.to_label(), 'Z')
        self.assertEqual(sgn, 1j)

        self.log.info("sign product reverse:")
        p3, sgn = Pauli.sgn_prod(p2, p1)  # pylint: disable=arguments-out-of-order
        self.log.info("p2: %s", p2.to_label())
        self.log.info("p1: %s", p1.to_label())
        self.log.info("p3: %s", p3.to_label())
        self.log.info("sgn_prod(p2, p1): %s", str(sgn))
        self.assertEqual(p1.to_label(), 'X')
        self.assertEqual(p2.to_label(), 'Y')
        self.assertEqual(p3.to_label(), 'Z')
        self.assertEqual(sgn, -1j)


if __name__ == '__main__':
    unittest.main()
