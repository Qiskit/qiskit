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

"""Tests for IdentityOp class."""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators import IdentityOp, Operator


class TestIdentityOp(QiskitTestCase):
    """Tests for IdentityOp class."""

    def test_init(self):
        """Test initialization."""

        for j in range(5):
            with self.subTest(msg='{}-qubit automatic dims'.format(j)):
                dim = 2 ** j
                op = IdentityOp(dim)
                self.assertEqual(op.dim, (dim, dim))
                self.assertEqual(op.input_dims(), j * (2, ))
                self.assertEqual(op.output_dims(), j * (2, ))
                self.assertEqual(op.coeff, None)

        with self.subTest(msg='custom dims'):
            dims = (2, 3, 4, 5)
            dim = np.product(dims)
            op = IdentityOp(dims)
            self.assertEqual(op.dim, (dim, dim))
            self.assertEqual(op.input_dims(), dims)
            self.assertEqual(op.output_dims(), dims)
            self.assertEqual(op.coeff, None)

        with self.subTest(msg='real coeff'):
            op = IdentityOp(5, coeff=5.2)
            self.assertEqual(op.coeff, 5.2)

        with self.subTest(msg='complex coeff'):
            op = IdentityOp(5, coeff=3j)
            self.assertEqual(op.coeff, 3j)

    def test_to_operator(self):
        """Test to_matrix and to_operator methods."""
        for dims in [2, 4, 5, (2, 3), (3, 2)]:
            for coeff in [0, 1, 2.1 - 3.1j]:
                dim = np.product(dims)
                iden = IdentityOp(dims, coeff=coeff)
                target = Operator(coeff * np.eye(dim),
                                  input_dims=dims,
                                  output_dims=dims)
                with self.subTest(msg='to_operator, dims = {}, coeff ='
                                      ' {}'.format(dims, coeff)):
                    self.assertEqual(iden.to_operator(), target)

                with self.subTest(msg='to_matrix, dims = {}, coeff ='
                                      ' {}'.format(dims, coeff)):
                    self.assertTrue(np.allclose(iden.to_matrix(), target.data))

    def test_base_operator_methods(self):
        """Test basic class methods"""
        for coeff in [None, 1, 5, -1j, 2.3-5.2j]:
            with self.subTest(msg='conjugate (coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).conjugate()
                target = None if coeff is None else np.conjugate(coeff)
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertEqual(op.coeff, target)

            with self.subTest(msg='transpose (coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).transpose()
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertEqual(op.coeff, coeff)

            with self.subTest(msg='adjoint (coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).adjoint()
                target = None if coeff is None else np.conjugate(coeff)
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertEqual(op.coeff, target)

            with self.subTest(msg='power (** 0, coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).power(0)
                target = None
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertAlmostEqual(op.coeff, target)

            with self.subTest(msg='power (** 1, coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).power(1)
                target = coeff
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertAlmostEqual(op.coeff, target)

            with self.subTest(msg='power (** 3.3, coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).power(3.3)
                target = None if coeff is None else coeff ** 3.3
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertAlmostEqual(op.coeff, target)

            with self.subTest(msg='power (** -2, coeff={})'.format(coeff)):
                op = IdentityOp(4, coeff=coeff).power(-2)
                target = None if coeff is None else coeff ** (-2)
                self.assertTrue(isinstance(op, IdentityOp))
                self.assertAlmostEqual(op.coeff, target)

    def test_multiply(self):
        """Test scalar multiplication."""

        for coeff1 in [None, 1, -3.1, 1 + 3j]:
            dims = (3, 2)
            op = IdentityOp(dims, coeff=coeff1)
            with self.subTest(msg='negate (coeff={})'.format(coeff1)):
                val = -op
                target = -1 if coeff1 is None else -1 * coeff1
                self.assertTrue(isinstance(val, IdentityOp))
                self.assertEqual(val.input_dims(), dims)
                self.assertEqual(val.output_dims(), dims)
                self.assertAlmostEqual(val.coeff, target)

            for coeff2 in [1, -1, -5.1 - 2j]:
                with self.subTest(msg='multiply ({}, coeff={})'.format(coeff2, coeff1)):
                    val = coeff2 * op
                    if coeff2 == 1:
                        target = coeff1
                    elif coeff1 is None:
                        target = coeff2
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

    def test_add(self):
        """Test add and subtract operations with IdentityOp."""

        # Add two IdentityOps
        for coeff1 in [None, 1, -3.1, 1 + 3j]:
            for coeff2 in [None, -1, -5.1 - 2j]:
                dims = (3, 2)
                op1 = IdentityOp(dims, coeff=coeff1)
                op2 = IdentityOp(6, coeff=coeff2)

                with self.subTest(msg='{} + {}'.format(op1, op2)):
                    val = op1 + op2
                    if coeff1 is None and coeff2 is None:
                        target = 2
                    elif coeff1 is None:
                        target = 1 + coeff2
                    elif coeff2 is None:
                        target = 1 + coeff1
                    else:
                        target = coeff1 + coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} - {}'.format(op1, op2)):
                    val = op1 - op2
                    if coeff1 is None and coeff2 is None:
                        target = 0
                    elif coeff1 is None:
                        target = 1 - coeff2
                    elif coeff2 is None:
                        target = coeff1 - 1
                    else:
                        target = coeff1 - coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

    def test_add_operator(self):
        """Test add and subtract operations with Operator."""

        for coeff in [None, 1, -3.1, 1 + 3j]:
            for label in ['II', 'XX', 'YY', 'ZZ']:
                dims = (2, 2)
                iden = IdentityOp(4, coeff=coeff)
                op = Operator.from_label(label)
                with self.subTest(msg='{} + Operator({})'.format(iden, label)):
                    val = iden + op
                    if coeff is None:
                        target = Operator.from_label('II') + op
                    else:
                        target = coeff * Operator.from_label('II') + op
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}) + {}'.format(label, iden)):
                    val = op + iden
                    if coeff is None:
                        target = Operator.from_label('II') + op
                    else:
                        target = coeff * Operator.from_label('II') + op
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertEqual(val, target)

                with self.subTest(msg='{} - Operator({})'.format(iden, label)):
                    val = iden - op
                    if coeff is None:
                        target = Operator.from_label('II') - op
                    else:
                        target = coeff * Operator.from_label('II') - op
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}) - {}'.format(label, iden)):
                    val = op - iden
                    if coeff is None:
                        target = op - Operator.from_label('II')
                    else:
                        target = op - coeff * Operator.from_label('II')
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertEqual(val, target)

    def test_tensor_identity(self):
        """Test tensor and expand methods with two IdentityOp."""

        for coeff1 in [None, 1, -3.1, 1 + 3j]:
            for coeff2 in [None, -1, -5.1 - 2j]:
                dims1 = (3, 2)
                dims2 = (2, 4)
                op1 = IdentityOp(dims1, coeff=coeff1)
                op2 = IdentityOp(dims2, coeff=coeff2)

                with self.subTest(msg='{}.expand({})'.format(op1, op2)):
                    val = op1.expand(op2)
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims1 + dims2)
                    self.assertEqual(val.output_dims(), dims1 + dims2)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{}.tensor({})'.format(op1, op2)):
                    val = op1.tensor(op2)
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims2 + dims1)
                    self.assertEqual(val.output_dims(), dims2 + dims1)
                    self.assertAlmostEqual(val.coeff, target)

    def test_tensor_operator(self):
        """Test tensor and expand methods with IdentityOp and Operator."""

        for coeff in [None, 1, -3.1, 1 + 3j]:
            for label in ['I', 'X', 'Y', 'Z']:
                dim = 3
                iden = IdentityOp(dim, coeff=coeff)
                op = Operator.from_label(label)

                with self.subTest(msg='{}.expand(Operator({}))'.format(iden, label)):
                    val = iden.expand(op)
                    target = iden.to_operator().expand(op)
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (3, 2))
                    self.assertEqual(val.output_dims(), (3, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}).expand({})'.format(label, iden)):
                    val = op.expand(iden)
                    target = op.expand(iden.to_operator())
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 3))
                    self.assertEqual(val.output_dims(), (2, 3))
                    self.assertEqual(val, target)

                with self.subTest(msg='{}.tensor(Operator({}))'.format(iden, label)):
                    val = iden.tensor(op)
                    target = iden.to_operator().tensor(op)
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 3))
                    self.assertEqual(val.output_dims(), (2, 3))
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}).tensor({})'.format(label, iden)):
                    val = op.tensor(iden)
                    target = op.tensor(iden.to_operator())
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (3, 2))
                    self.assertEqual(val.output_dims(), (3, 2))
                    self.assertEqual(val, target)

    def test_compose_identity(self):
        """Test compose and dot methods with two IdentityOp."""
        # Add two IdentityOps
        for coeff1 in [None, 1, -3.1, 1 + 3j]:
            for coeff2 in [None, -1, -5.1 - 2j]:
                dims = (3, 2)
                op1 = IdentityOp(dims, coeff=coeff1)
                op2 = IdentityOp(dims, coeff=coeff2)

                with self.subTest(msg='{}.compose({})'.format(op1, op2)):
                    val = op1.compose(op2)
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{}.dot({})'.format(op1, op2)):
                    val = op1.dot(op2)
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} @ {}'.format(op1, op2)):
                    val = op1 @ op2
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} * {}'.format(op1, op2)):
                    val = op1 * op2
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

    def test_compose_qargs_identity(self):
        """Test qargs compose and dot methods with two IdentityOp."""
        # Add two IdentityOps
        for coeff1 in [None, 1, -3.1, 1 + 3j]:
            for coeff2 in [None, -1, -5.1 - 2j]:
                dims = (2, 3)
                op1 = IdentityOp(dims, coeff=coeff1)
                op2 = IdentityOp(2, coeff=coeff2)
                op3 = IdentityOp(3, coeff=coeff2)

                with self.subTest(msg='{}.compose({}, qargs=[0])'.format(op1, op2)):
                    val = op1.compose(op2, qargs=[0])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{}.compose({}, qargs=[1])'.format(op1, op3)):
                    val = op1.compose(op3, qargs=[1])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{}.dot({}, qargs=[0])'.format(op1, op2)):
                    val = op1.dot(op2, qargs=[0])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{}.dot({}, qargs=[1])'.format(op1, op3)):
                    val = op1.dot(op3, qargs=[1])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} @ {}([0])'.format(op1, op2)):
                    val = op1 @ op2([0])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} @ {}([1])'.format(op1, op3)):
                    val = op1 @ op3([1])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} * {}([0])'.format(op1, op2)):
                    val = op1 * op2([0])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

                with self.subTest(msg='{} * {}([1])'.format(op1, op3)):
                    val = op1 * op3([1])
                    if coeff1 is None:
                        target = coeff2
                    elif coeff2 is None:
                        target = coeff1
                    else:
                        target = coeff1 * coeff2
                    self.assertTrue(isinstance(val, IdentityOp))
                    self.assertEqual(val.input_dims(), dims)
                    self.assertEqual(val.output_dims(), dims)
                    self.assertAlmostEqual(val.coeff, target)

    def test_compose_operator(self):
        """Test compose and dot methods with IdentityOp and Operator."""
        for coeff in [None, 1, -3.1, 1 + 3j]:
            for label in ['II', 'XX', 'YY', 'ZZ']:
                dim = 4
                iden = IdentityOp(dim, coeff=coeff)
                op = Operator.from_label(label)

                with self.subTest(msg='{}.compose(Operator({}))'.format(iden, label)):
                    val = iden.compose(op)
                    target = iden.to_operator().compose(op)
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}).compose({})'.format(label, iden)):
                    val = op.compose(iden)
                    target = op.compose(iden.to_operator())
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{}.dot(Operator({}))'.format(iden, label)):
                    val = iden.dot(op)
                    target = iden.to_operator().dot(op)
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}).dot({})'.format(label, iden)):
                    val = op.dot(iden)
                    target = op.dot(iden.to_operator())
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{} @ Operator({})'.format(iden, label)):
                    val = iden @ op
                    target = iden.to_operator().compose(op)
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}) @ {}'.format(label, iden)):
                    val = op @ iden
                    target = op.compose(iden.to_operator())
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{} * Operator({})'.format(iden, label)):
                    val = iden * op
                    target = iden.to_operator().dot(op)
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='Operator({}) * {}'.format(label, iden)):
                    val = op * iden
                    target = op.dot(iden.to_operator())
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

    def test_compose_qargs_operator(self):
        """Test qargs compose and dot methods with IdentityOp and Operator."""
        for coeff in [None, 1, -3.1, 1 + 3j]:
            for label in ['I', 'X', 'Y', 'Z']:
                iden = IdentityOp((2, 2), coeff=coeff)
                op = Operator.from_label(label)

                with self.subTest(msg='{}.compose(Operator({}), qargs=[0])'.format(iden, label)):
                    val = iden.compose(op, qargs=[0])
                    target = iden.to_operator().compose(op, qargs=[0])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{}.compose(Operator({}), qargs=[1])'.format(iden, label)):
                    val = iden.compose(op, qargs=[1])
                    target = iden.to_operator().compose(op, qargs=[1])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{}.dot(Operator({}), qargs=[0])'.format(iden, label)):
                    val = iden.dot(op, qargs=[0])
                    target = iden.to_operator().dot(op, qargs=[0])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{}.dot(Operator({}), qargs=[1])'.format(iden, label)):
                    val = iden.dot(op, qargs=[1])
                    target = iden.to_operator().dot(op, qargs=[1])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{} @ Operator({})([0])'.format(iden, label)):
                    val = iden @ op([0])
                    target = iden.to_operator().compose(op, qargs=[0])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{} @ Operator({})([1])'.format(iden, label)):
                    val = iden @ op([1])
                    target = iden.to_operator().compose(op, qargs=[1])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{} * Operator({})([0])'.format(iden, label)):
                    val = iden * op([0])
                    target = iden.to_operator().dot(op, qargs=[0])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)

                with self.subTest(msg='{} * Operator({})([1])'.format(iden, label)):
                    val = iden * op([1])
                    target = iden.to_operator().dot(op, qargs=[1])
                    self.assertTrue(isinstance(val, Operator))
                    self.assertEqual(val.input_dims(), (2, 2))
                    self.assertEqual(val.output_dims(), (2, 2))
                    self.assertEqual(val, target)


if __name__ == '__main__':
    unittest.main()
