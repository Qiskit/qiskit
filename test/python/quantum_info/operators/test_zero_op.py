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

"""Tests for ZeroOp class."""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators import ZeroOp, IdentityOp, Operator


class TestZeroOp(QiskitTestCase):
    """Tests for ZeroOp class."""

    def test_init(self):
        """Test initialization."""

        for j in range(5):
            with self.subTest(msg='{}-qubit automatic dims'.format(j)):
                dim = 2 ** j
                op = ZeroOp(dim)
                self.assertEqual(op.dim, (dim, dim))
                self.assertEqual(op.input_dims(), j * (2, ))
                self.assertEqual(op.output_dims(), j * (2, ))

        with self.subTest(msg='input_dims only'):
            dims = (2, 3, 4, 5)
            dim = np.product(dims)
            op = ZeroOp(input_dims=dims)
            self.assertEqual(op.dim, (dim, dim))
            self.assertEqual(op.input_dims(), dims)
            self.assertEqual(op.output_dims(), dims)

        with self.subTest(msg='output_dims only'):
            dims = (2, 3, 4, 5)
            dim = np.product(dims)
            op = ZeroOp(output_dims=dims)
            self.assertEqual(op.dim, (dim, dim))
            self.assertEqual(op.input_dims(), dims)
            self.assertEqual(op.output_dims(), dims)

        with self.subTest(msg='custom dims only'):
            input_dims = (2, 3)
            output_dims = (4, 5)
            dim_in = np.product(input_dims)
            dim_out = np.product(output_dims)
            op = ZeroOp(input_dims=input_dims,
                        output_dims=output_dims)
            self.assertEqual(op.dim, (dim_in, dim_out))
            self.assertEqual(op.input_dims(), input_dims)
            self.assertEqual(op.output_dims(), output_dims)

    def test_to_operator(self):
        """Test to_matrix and to_operator methods."""
        for input_dims in [2, 4, 5, (2, 3), (3, 2)]:
            for output_dims in [2, 4, 5, (2, 3), (3, 2)]:
                dim_in = np.product(input_dims)
                dim_out = np.product(output_dims)
                op = ZeroOp(output_dims=output_dims,
                            input_dims=input_dims)
                target = Operator(np.zeros((dim_out, dim_in)),
                                  input_dims=input_dims,
                                  output_dims=output_dims)
                with self.subTest(msg='{}.to_operator()'.format(op)):
                    self.assertEqual(op.to_operator(), target)

                with self.subTest(msg='{}.to_matrix()'.format(op)):
                    self.assertTrue(np.allclose(op.to_matrix(), target.data))

    def test_base_operator_methods(self):
        """Test basic class methods"""

        input_dims = (2, 3)
        output_dims = (4, 5)
        op = ZeroOp(input_dims=input_dims,
                    output_dims=output_dims)

        with self.subTest(msg='conjugate'):
            value = op.conjugate()
            target = ZeroOp(input_dims=input_dims,
                            output_dims=output_dims)
            self.assertEqual(value, target)

        with self.subTest(msg='transpose'):
            value = op.transpose()
            target = ZeroOp(input_dims=output_dims,
                            output_dims=input_dims)
            self.assertEqual(value, target)

        with self.subTest(msg='adjoint'):
            value = op.adjoint()
            target = ZeroOp(input_dims=output_dims,
                            output_dims=input_dims)
            self.assertEqual(value, target)

        op = ZeroOp(input_dims=input_dims)
        for p in [0, 1, 3.3, -2]:
            with self.subTest(msg='power ** {}'.format(p)):
                value = op.power(p)
                target = op
                self.assertEqual(value, target)

    def test_multiply(self):
        """Test scalar multiplication."""

        target = ZeroOp((2, 3), (4, 5))
        with self.subTest(msg='negate'):
            val = -target
            self.assertAlmostEqual(val, target)

        for scalar in [1, -1, -5.1 - 2j]:
            with self.subTest(msg='multiply ({})'.format(scalar)):
                val = scalar * target
                self.assertAlmostEqual(val, target)

    def test_add_zero(self):
        """Test add and subtract operations with two ZeroOp."""
        for input_dims in [2, 4, 5, (2, 3), (3, 2)]:
            for output_dims in [2, 4, 5, (2, 3), (3, 2)]:
                op = ZeroOp(output_dims=output_dims,
                            input_dims=input_dims)
                with self.subTest(msg='{} + {}'.format(op, op)):
                    value = op + op
                    self.assertEqual(value, op)

                with self.subTest(msg='{} - {}'.format(op, op)):
                    value = op + op
                    self.assertEqual(value, op)

    def test_add_identity(self):
        """Test add and subtract operations with IdentityOp."""

        for dims in [2, 3, (2, 3), (3, 2)]:
            for coeff in [None, 1, -3.3, 4.5j]:
                zero = ZeroOp(dims)
                iden = IdentityOp(dims, coeff=coeff)

                with self.subTest(msg='{} + {}'.format(zero, iden)):
                    value = zero + iden
                    target = iden
                    self.assertEqual(value, target)

                with self.subTest(msg='{} + {}'.format(iden, zero)):
                    value = iden + zero
                    target = iden
                    self.assertEqual(value, target)

                with self.subTest(msg='{} - {}'.format(zero, iden)):
                    value = zero - iden
                    target = -iden
                    self.assertEqual(value, target)

                with self.subTest(msg='{} - {}'.format(iden, zero)):
                    value = iden - zero
                    target = iden
                    self.assertEqual(value, target)

    def test_add_operator(self):
        """Test add and subtract operations with Operator."""

        for input_dims in [2, 3, (2, 3), (3, 2)]:
            for output_dims in [2, 3, (2, 3), (3, 2)]:
                shape = (np.product(output_dims), np.product(input_dims))
                zero = ZeroOp(output_dims=output_dims,
                              input_dims=input_dims)
                ones = Operator(np.ones(shape),
                                output_dims=output_dims,
                                input_dims=input_dims)

                with self.subTest(msg='{} + {}'.format(zero, ones)):
                    value = zero + ones
                    target = ones
                    self.assertEqual(value, target)

                with self.subTest(msg='{} + {}'.format(ones, zero)):
                    value = ones + zero
                    target = ones
                    self.assertEqual(value, target)

                with self.subTest(msg='{} - {}'.format(zero, ones)):
                    value = zero - ones
                    target = -1 * ones
                    self.assertEqual(value, target)

                with self.subTest(msg='{} - {}'.format(ones, zero)):
                    value = ones - zero
                    target = ones
                    self.assertEqual(value, target)

    def test_tensor_zero(self):
        """Test tensor and expand methods with two ZeroOp."""

        dims1 = [(2,), (3,)]
        dims2 = [(2, 3), (3, 2)]
        for input_dims1 in dims1:
            for output_dims1 in dims1:
                for input_dims2 in dims2:
                    for output_dims2 in dims2:

                        op1 = ZeroOp(output_dims=output_dims1,
                                     input_dims=input_dims1)
                        op2 = ZeroOp(output_dims=output_dims2,
                                     input_dims=input_dims2)

                        with self.subTest(msg='{}.expand({})'.format(op1, op2)):
                            value = op1.expand(op2)
                            target = ZeroOp(output_dims=output_dims1 + output_dims2,
                                            input_dims=input_dims1 + input_dims2)
                            self.assertEqual(value, target)

                        with self.subTest(msg='{}.tensor({})'.format(op1, op2)):
                            value = op1.tensor(op2)
                            target = ZeroOp(output_dims=output_dims2 + output_dims1,
                                            input_dims=input_dims2 + input_dims1)
                            self.assertEqual(value, target)

    def test_tensor_identity(self):
        """Test tensor and expand methods with ZeroOp and IdentityOp."""

        dims = [(2, 3), (3, 2)]
        coeffs = [None, 1, -3.1, 1 + 3j]
        for input_dims in dims:
            for output_dims in dims:
                for iden_dims in dims:
                    for coeff in coeffs:
                        op1 = ZeroOp(output_dims=output_dims,
                                     input_dims=input_dims)
                        op2 = IdentityOp(iden_dims, coeff=coeff)

                        with self.subTest(msg='{}.expand({})'.format(op1, op2)):
                            value = op1.expand(op2)
                            target = ZeroOp(output_dims=output_dims + iden_dims,
                                            input_dims=input_dims + iden_dims)
                            self.assertEqual(value, target)

                        with self.subTest(msg='{}.tensor({})'.format(op1, op2)):
                            value = op1.tensor(op2)
                            target = ZeroOp(output_dims=iden_dims + output_dims,
                                            input_dims=iden_dims + input_dims)
                            self.assertEqual(value, target)

    def test_tensor_operator(self):
        """Test tensor and expand methods with ZeroOp and Operator."""

        dims1 = [(2, 4), (4, 2)]
        dims2 = [(2,), (3,)]
        for input_dims1 in dims1:
            for output_dims1 in dims1:
                for input_dims2 in dims2:
                    for output_dims2 in dims2:

                        op1 = ZeroOp(output_dims=output_dims1,
                                     input_dims=input_dims1)
                        shape = (np.product(output_dims2),
                                 np.product(input_dims2))
                        op2 = Operator(np.ones(shape),
                                       output_dims=output_dims2,
                                       input_dims=input_dims2)

                        with self.subTest(msg='{}.expand({})'.format(op1, op2)):
                            value = op1.expand(op2)
                            target = ZeroOp(output_dims=output_dims1 + output_dims2,
                                            input_dims=input_dims1 + input_dims2)
                            self.assertEqual(value, target)

                        with self.subTest(msg='{}.tensor({})'.format(op1, op2)):
                            value = op1.tensor(op2)
                            target = ZeroOp(output_dims=output_dims2 + output_dims1,
                                            input_dims=input_dims2 + input_dims1)
                            self.assertEqual(value, target)

    def test_compose_zero(self):
        """Test compose and dot methods with two ZeroOp."""
        dims = [2, 3, (2, 3), (3, 2)]
        for input_dims in dims:
            for output_dims in dims:
                op1 = ZeroOp(output_dims=output_dims,
                             input_dims=input_dims)
                op2 = ZeroOp(output_dims=input_dims,
                             input_dims=output_dims)

                with self.subTest(msg='{}.compose({})'.format(op1, op2)):
                    value = op1.compose(op2)
                    target = ZeroOp(input_dims)
                    self.assertEqual(value, target)

                with self.subTest(msg='{}.dot({})'.format(op1, op2)):
                    value = op1.dot(op2)
                    target = ZeroOp(output_dims)
                    self.assertEqual(value, target)

                with self.subTest(msg='{} @ {}'.format(op1, op2)):
                    value = op1 @ op2
                    target = ZeroOp(input_dims)
                    self.assertEqual(value, target)

                with self.subTest(msg='{} * {}'.format(op1, op2)):
                    value = op1 * op2
                    target = ZeroOp(output_dims)
                    self.assertEqual(value, target)

    def test_compose_operator(self):
        """Test compose and dot methods with ZeroOp and Operator."""
        dims = [2, 3, (2, 3), (3, 2)]
        for input_dims in dims:
            for output_dims in dims:
                op1 = ZeroOp(output_dims=output_dims,
                             input_dims=input_dims)
                shape = (np.product(input_dims),
                         np.product(output_dims))
                op2 = Operator(np.ones(shape),
                               output_dims=input_dims,
                               input_dims=output_dims)

                with self.subTest(msg='{}.compose({})'.format(op1, op2)):
                    value = op1.compose(op2)
                    target = ZeroOp(input_dims)
                    self.assertEqual(value, target)

                with self.subTest(msg='{}.dot({})'.format(op1, op2)):
                    value = op1.dot(op2)
                    target = ZeroOp(output_dims)
                    self.assertEqual(value, target)

                with self.subTest(msg='{} @ {}'.format(op1, op2)):
                    value = op1 @ op2
                    target = ZeroOp(input_dims)
                    self.assertEqual(value, target)

                with self.subTest(msg='{} * {}'.format(op1, op2)):
                    value = op1 * op2
                    target = ZeroOp(output_dims)
                    self.assertEqual(value, target)

    def test_compose_identity(self):
        """Test compose and dot methods with ZeroOp and IdentityOp."""
        dims = [2, 3, (2, 3), (3, 2)]
        for input_dims in dims:
            for output_dims in dims:
                op1 = ZeroOp(output_dims=output_dims,
                             input_dims=input_dims)

                op2 = IdentityOp(output_dims)
                with self.subTest(msg='{}.compose({})'.format(op1, op2)):
                    value = op1.compose(op2)
                    target = op1
                    self.assertEqual(value, target)

                with self.subTest(msg='{} @ {}'.format(op1, op2)):
                    value = op1 @ op2
                    target = op1
                    self.assertEqual(value, target)         

                op2 = IdentityOp(input_dims)
                with self.subTest(msg='{}.dot({})'.format(op1, op2)):
                    value = op1.dot(op2)
                    target = op1
                    self.assertEqual(value, target)

                with self.subTest(msg='{} * {}'.format(op1, op2)):
                    value = op1 * op2
                    target = op1
                    self.assertEqual(value, target)

    def test_compose_qargs_zero(self):
        """Test qargs compose and dot methods with two ZeroOp."""
        dims = [(2, 3), (3, 2)]
        for input_dims in dims:
            for output_dims in dims:
                for i in range(len(dims)):
                    op1 = ZeroOp(output_dims=output_dims,
                                 input_dims=input_dims)
                    op2 = ZeroOp(output_dims=input_dims[i],
                                 input_dims=output_dims[i])

                    with self.subTest(msg='{}.compose({}, qargs=[{}])'
                                          ''.format(op1, op2, i)):
                        value = op1.compose(op2, qargs=[i])
                        target_output_dims = list(output_dims)
                        target_output_dims[i] = input_dims[i]
                        target = ZeroOp(output_dims=target_output_dims,
                                        input_dims=input_dims)
                        self.assertEqual(value, target)

                    with self.subTest(msg='{}.dot({}, qargs=[{}])'
                                          ''.format(op1, op2, i)):
                        value = op1.dot(op2, qargs=[i])
                        target_input_dims = list(input_dims)
                        target_input_dims[i] = output_dims[i]
                        target = ZeroOp(output_dims=output_dims,
                                        input_dims=target_input_dims)
                        self.assertEqual(value, target)

                    with self.subTest(msg='{} @ {}([{}])'.format(op1, op2, i)):
                        value = op1 @ op2([i])
                        target_output_dims = list(output_dims)
                        target_output_dims[i] = input_dims[i]
                        target = ZeroOp(output_dims=target_output_dims,
                                        input_dims=input_dims)
                        self.assertEqual(value, target)

                    with self.subTest(msg='{} * {}([{}])'.format(op1, op2, i)):
                        value = op1 * op2([i])
                        target_input_dims = list(input_dims)
                        target_input_dims[i] = output_dims[i]
                        target = ZeroOp(output_dims=output_dims,
                                        input_dims=target_input_dims)
                        self.assertEqual(value, target)

    def test_compose_qargs_operator(self):
        """Test qargs compose and dot methods with ZeroOp and Operator."""
        dims = [(2, 3), (3, 2)]
        for input_dims in dims:
            for output_dims in dims:
                for i in range(len(dims)):
                    op1 = ZeroOp(output_dims=output_dims,
                                 input_dims=input_dims)
                    op2 = Operator(np.ones((input_dims[i], output_dims[i])),
                                   output_dims=input_dims[i],
                                   input_dims=output_dims[i])

                    with self.subTest(msg='{}.compose({}, qargs=[{}])'
                                          ''.format(op1, op2, i)):
                        value = op1.compose(op2, qargs=[i])
                        target_output_dims = list(output_dims)
                        target_output_dims[i] = input_dims[i]
                        target = ZeroOp(output_dims=target_output_dims,
                                        input_dims=input_dims)
                        self.assertEqual(value, target)

                    with self.subTest(msg='{}.dot({}, qargs=[{}])'
                                          ''.format(op1, op2, i)):
                        value = op1.dot(op2, qargs=[i])
                        target_input_dims = list(input_dims)
                        target_input_dims[i] = output_dims[i]
                        target = ZeroOp(output_dims=output_dims,
                                        input_dims=target_input_dims)
                        self.assertEqual(value, target)

                    with self.subTest(msg='{} @ {}([{}])'.format(op1, op2, i)):
                        value = op1 @ op2([i])
                        target_output_dims = list(output_dims)
                        target_output_dims[i] = input_dims[i]
                        target = ZeroOp(output_dims=target_output_dims,
                                        input_dims=input_dims)
                        self.assertEqual(value, target)

                    with self.subTest(msg='{} * {}([{}])'.format(op1, op2, i)):
                        value = op1 * op2([i])
                        target_input_dims = list(input_dims)
                        target_input_dims[i] = output_dims[i]
                        target = ZeroOp(output_dims=output_dims,
                                        input_dims=target_input_dims)
                        self.assertEqual(value, target)

    def test_compose_qargs_identity(self):
        """Test qargs compose and dot methods with ZeroOp and IdentityOp."""
        dims = [(2, 3), (3, 2)]
        for input_dims in dims:
            for output_dims in dims:
                for i in range(len(dims)):
                    op1 = ZeroOp(output_dims=output_dims,
                                 input_dims=input_dims)

                    op2 = IdentityOp(output_dims[i])
                    with self.subTest(msg='{}.compose({}, qargs=[{}])'
                                          ''.format(op1, op2, i)):
                        value = op1.compose(op2, qargs=[i])
                        target = op1
                        self.assertEqual(value, target)

                    with self.subTest(msg='{} @ {}([{}])'.format(op1, op2, i)):
                        value = op1 @ op2([i])
                        target = op1
                        self.assertEqual(value, target)

                    op2 = IdentityOp(input_dims[i])
                    with self.subTest(msg='{}.dot({}, qargs=[{}])'
                                          ''.format(op1, op2, i)):
                        value = op1.dot(op2, qargs=[i])
                        target = op1
                        self.assertEqual(value, target)

                    with self.subTest(msg='{} * {}([{}])'.format(op1, op2, i)):
                        value = op1 * op2([i])
                        target = op1
                        self.assertEqual(value, target)


if __name__ == '__main__':
    unittest.main()
