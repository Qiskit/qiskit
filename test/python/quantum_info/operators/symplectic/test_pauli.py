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

"""Tests for Pauli operator class."""

import unittest
import itertools as it
from functools import lru_cache
import logging

import numpy as np
from ddt import ddt, data, unpack

from qiskit.exceptions import QiskitError
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.test import QiskitTestCase

from qiskit.quantum_info.random import random_clifford, random_pauli
from qiskit.quantum_info.operators import Pauli, Operator
from qiskit.quantum_info.operators.symplectic.pauli_tools import (
    split_pauli_label, coeff_phase_from_label)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def pauli_group_labels(nq, full_group=True):
    """Generate list of the N-qubit pauli group string labels"""
    labels = [''.join(i) for i in it.product(('I', 'X', 'Y', 'Z'), repeat=nq)]
    if full_group:
        labels = [''.join(i) for i in it.product(('', '-i', '-', 'i'), labels)]
    return labels


def operator_from_label(label):
    """Construct operator from full Pauli group label"""
    pauli, coeff = split_pauli_label(label)
    coeff = (-1j) ** coeff_phase_from_label(coeff)
    return coeff * Operator.from_label(pauli)


@ddt
class TestPauliConversions(QiskitTestCase):
    """Test representation conversions of Pauli"""

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_labels(self, label):
        """Test round trip label conversion"""
        pauli = Pauli(label)
        self.assertEqual(Pauli(str(pauli)), pauli)

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_to_operator(self, label):
        """Test Pauli operator conversion"""
        value = Pauli(label).to_operator()
        target = operator_from_label(label)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_to_matrix_sparse(self, label):
        """Test Pauli operator conversion"""
        spmat = Pauli(label).to_matrix(sparse=True)
        value = Operator(spmat.todense())
        target = operator_from_label(label)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_to_instruction(self, label):
        """Test Pauli to instruction"""
        pauli = Pauli(label)
        value = Operator(pauli.to_instruction())
        target = pauli.to_operator()
        self.assertEqual(value, target)

    @data((IGate(), 'I'), (XGate(), 'X'), (YGate(), 'Y'), (ZGate(), 'Z'))
    @unpack
    def test_init_single_pauli_gate(self, gate, label):
        """Test initialization from Pauli basis gates"""
        self.assertEqual(str(Pauli(gate)), label)

    @data('IXYZ', 'XXY', 'ZYX', 'ZI', 'Y')
    def test_init_pauli_gate(self, label):
        """Test initialization from Pauli basis gates"""
        pauli = Pauli(PauliGate(label))
        self.assertEqual(str(pauli), label)


@ddt
class TestPauliProperties(QiskitTestCase):
    """Test Pauli properties"""

    @data('I', 'XY', 'XYZ', 'IXYZ', 'IXYZX')
    def test_len(self, label):
        """Test __len__ method"""
        self.assertEqual(len(Pauli(label)), len(label))

    @data(*it.product(pauli_group_labels(1, full_group=False),
                      pauli_group_labels(1)))
    @unpack
    def test_equal(self, label1, label2):
        """Test __eq__ method"""
        pauli1 = Pauli(label1)
        pauli2 = Pauli(label2)
        target = np.all(pauli1.array == pauli2.array) and pauli1.phase == pauli2.phase
        self.assertEqual(pauli1 == pauli2, target)

    @data(*it.product(pauli_group_labels(1, full_group=False),
                      pauli_group_labels(1)))
    @unpack
    def test_equiv(self, label1, label2):
        """Test equiv method"""
        pauli1 = Pauli(label1)
        pauli2 = Pauli(label2)
        target = np.all(pauli1.array == pauli2.array)
        self.assertEqual(pauli1.equiv(pauli2), target)

    @data(*pauli_group_labels(1))
    def test_phase(self, label):
        """Test phase attribute"""
        pauli = Pauli(label)
        _, coeff = split_pauli_label(str(pauli))
        target = coeff_phase_from_label(coeff)
        self.assertEqual(pauli.phase, target)

    @data(*[(p, q) for p in ['I', 'X', 'Y', 'Z'] for q in range(4)])
    @unpack
    def test_phase_setter(self, pauli, phase):
        """Test phase setter"""
        pauli = Pauli(pauli)
        pauli.phase = phase
        _, coeff = split_pauli_label(str(pauli))
        value = coeff_phase_from_label(coeff)
        self.assertEqual(value, phase)

    def test_array_setter_raises(self):
        """Test changing array size raises"""
        pauli = Pauli('IX')

        def bad_assign():
            pauli.array = np.array([False, True])

        self.assertRaises(ValueError, bad_assign)

    def test_x_setter(self):
        """Test phase attribute"""
        pauli = Pauli('II')
        pauli.x = True
        self.assertEqual(pauli, Pauli('XX'))

    def test_z_setter(self):
        """Test phase attribute"""
        pauli = Pauli('II')
        pauli.z = True
        self.assertEqual(pauli, Pauli('ZZ'))

    @data(*[('IXYZ', i) for i in [0, 1, 2, 3, slice(None, None, None),
                                  slice(None, 2, None), [0, 3], [2, 1, 3]]])
    @unpack
    def test_getitem(self, label, qubits):
        """Test __getitem__"""
        pauli = Pauli(label)
        value = str(pauli[qubits])
        val_array = np.array(list(reversed(label)))[qubits]
        target = ''.join(reversed(val_array.tolist()))
        self.assertEqual(value, target, msg='indices = {}'.format(qubits))

    @data((0, 'iY', 'iIIY'), ([1, 0], 'XZ', 'IZX'),
          (slice(None, None, None), 'XYZ', 'XYZ'),
          (slice(None, None, -1), 'XYZ', 'ZYX'))
    @unpack
    def test_setitem(self, qubits, value, target):
        """Test __setitem__"""
        pauli = Pauli('III')
        pauli[qubits] = value
        self.assertEqual(str(pauli), target)

    def test_insert(self):
        """Test insert method"""
        pauli = Pauli('III')
        pauli = pauli.insert([2, 0, 4], 'XYZ')
        self.assertEqual(str(pauli), 'IXIZIY')

    def test_delete(self):
        """Test delete method"""
        pauli = Pauli('IXYZ')
        pauli = pauli.delete([0, 2])
        self.assertEqual(str(pauli), 'IY')


@ddt
class TestPauli(QiskitTestCase):
    """Tests for Pauli operator class."""

    @data(*pauli_group_labels(2))
    def test_conjugate(self, label):
        """Test conjugate method."""
        value = Pauli(label).conjugate()
        target = operator_from_label(label).conjugate()
        self.assertEqual(Operator(value), target)

    @data(*pauli_group_labels(2))
    def test_transpose(self, label):
        """Test transpose method."""
        value = Pauli(label).transpose()
        target = operator_from_label(label).transpose()
        self.assertEqual(Operator(value), target)

    @data(*pauli_group_labels(2))
    def test_adjoint(self, label):
        """Test adjoint method."""
        value = Pauli(label).adjoint()
        target = operator_from_label(label).adjoint()
        self.assertEqual(Operator(value), target)

    @data(*pauli_group_labels(2))
    def test_inverse(self, label):
        """Test inverse method."""
        pauli = Pauli(label)
        value = pauli.inverse()
        target = pauli.adjoint()
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(2, full_group=False), repeat=2))
    @unpack
    def test_dot(self, label1, label2):
        """Test dot method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.dot(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.dot(op2)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1))
    def test_dot_qargs(self, label2):
        """Test dot method with qargs."""
        label1 = '-iXYZ'
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        qargs = [0]
        value = Operator(p1.dot(p2, qargs=qargs))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.dot(op2, qargs=qargs)
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(2, full_group=False), repeat=2))
    @unpack
    def test_compose(self, label1, label2):
        """Test compose method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.compose(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.compose(op2)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1))
    def test_compose_qargs(self, label2):
        """Test compose method with qargs."""
        label1 = '-XYZ'
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        qargs = [0]
        value = Operator(p1.compose(p2, qargs=qargs))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.compose(op2, qargs=qargs)
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(1, full_group=False), repeat=2))
    @unpack
    def test_tensor(self, label1, label2):
        """Test tensor method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.tensor(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.tensor(op2)
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(1, full_group=False), repeat=2))
    @unpack
    def test_expand(self, label1, label2):
        """Test expand method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.expand(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.expand(op2)
        self.assertEqual(value, target)

    @data('II', 'XI', 'YX', 'ZZ', 'YZ')
    def test_power(self, label):
        """Test power method."""
        iden = Pauli('II')
        op = Pauli(label)
        self.assertTrue(op ** 2, iden)

    def test_power_except(self):
        """Test power method raises exceptions."""
        op = Pauli('YXZ')
        # Non-integer power raises error
        self.assertRaises(QiskitError, op.power, 0.5)

    def test_add_except(self):
        """Test add method raises exceptions."""
        P1 = Pauli('X')
        P2 = Pauli('Z')
        self.assertRaises(NotImplementedError, P1._add, P2)

    @data(1, 1.0, -1, -1.0, 1j, -1j)
    def test_multiply(self, val):
        """Test multiply method."""
        op = val * Pauli([True, True], [False, False], phase=0)
        phase = (-1j) ** op.phase
        self.assertEqual(phase, val)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        op = Pauli('XYZ')
        self.assertRaises(QiskitError, op._multiply, 2)

    @data(0, 1, 2, 3)
    def test_negate(self, phase):
        """Test negate method"""
        op = Pauli([False], [True], phase)
        neg = -op
        self.assertTrue(op.equiv(neg))
        self.assertEqual(neg.phase, (op.phase + 2) % 4)

    @data(*it.product(pauli_group_labels(1, False), repeat=2))
    @unpack
    def test_commutes(self, p1, p2):
        """Test commutes method"""
        P1 = Pauli(p1)
        P2 = Pauli(p2)
        self.assertEqual(P1.commutes(P2), P1.dot(P2) == P2.dot(P1))

    @data(*it.product(pauli_group_labels(1, False), repeat=2))
    @unpack
    def test_anticommutes(self, p1, p2):
        """Test anticommutes method"""
        P1 = Pauli(p1)
        P2 = Pauli(p2)
        self.assertEqual(P1.anticommutes(P2), P1.dot(P2) == -P2.dot(P1))

    def test_evolve_clifford(self):
        """Test evolve method"""
        cliff = random_clifford(3, seed=10)
        op = Operator(cliff)
        pauli = random_pauli(3, seed=10)
        value = Operator(pauli.evolve(cliff))
        target = op.dot(pauli).dot(op.adjoint())
        self.assertEqual(value, target)


if __name__ == '__main__':
    unittest.main()
