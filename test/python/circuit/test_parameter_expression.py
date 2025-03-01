# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test parameter expression."""

import math

from test import combine
from test import QiskitTestCase

import ddt

from qiskit.circuit import Parameter, ParameterVector, ParameterExpression


param_x = Parameter("x")
param_y = Parameter("y")
nested_expr = param_x + param_y - param_x
nested_expr = nested_expr.subs({param_y: param_x})

vector = ParameterVector("vec", 1000)
nested_vector_expr = vector[500] + vector[256] - vector[500]
for i in range(1000):
    nested_vector_expr += vector[i] - vector[i]


operands = [
    Parameter("a"),
    Parameter("大"),
    ParameterVector("a", 100)[42],
    complex(3.14, -3.14),
    2.3,
    int(5),
    1.0,
    -1.0,
    Parameter("ab") + 2 - 2,
    Parameter("abc") ** 1.0,
    Parameter("abcd") / 1,
    Parameter("abcd_complex") / complex(1, 0),
    ParameterVector("b", 1)[0] + (0 * 1) * Parameter("ZERO"),
    nested_expr,
    nested_vector_expr,
]


@ddt.ddt
class TestParameterExpression(QiskitTestCase):
    """Test parameter expression."""

    @combine(
        left=operands,
        right=operands,
    )
    def test_addition_simple(self, left, right):
        """Test expression addition."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left + right
            res = expr.bind({x: 1.0 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, 2.0)
            elif not isinstance(left, ParameterExpression):
                self.assertEqual(res, left + 1.0)
            elif not isinstance(right, ParameterExpression):
                self.assertEqual(res, right + 1.0)

    @combine(
        left=operands,
        right=operands,
    )
    def test_subtraction_simple(self, left, right):
        """Test expression subtraction."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left - right
            res = expr.bind({x: 1.0 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, 0.0)
            elif not isinstance(left, ParameterExpression):
                self.assertEqual(res, left - 1.0)
            elif not isinstance(right, ParameterExpression):
                self.assertEqual(res, 1.0 - right)

    @combine(
        left=operands,
        right=operands,
    )
    def test_multiplication_simple(self, left, right):
        """Test expression multiplication."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left * right
            res = expr.bind({x: 1.0 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, 1.0)
            elif not isinstance(left, ParameterExpression):
                self.assertEqual(res, left)
            elif not isinstance(right, ParameterExpression):
                self.assertEqual(res, right)

    @combine(
        left=operands,
        right=operands,
    )
    def test_division_simple(self, left, right):
        """Test expression division."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left / right
            res = expr.bind({x: 1.0 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, 1.0)
            elif not isinstance(left, ParameterExpression):
                self.assertEqual(res, left)
            elif not isinstance(right, ParameterExpression):
                self.assertEqual(res, 1.0 / right)

    @combine(
        left=operands,
        right=operands,
    )
    def test_pow_simple(self, left, right):
        """Test expression pow."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left**right
            res = expr.bind({x: 1.0 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, 1.0)
            elif not isinstance(left, ParameterExpression):
                if isinstance(left, complex):
                    self.assertAlmostEqual(complex(res), left)
                else:
                    self.assertEqual(res, left)
            elif not isinstance(right, ParameterExpression):
                self.assertEqual(res, 1.0**right)

    @combine(expression=operands)
    def test_abs_simple(self, expression):
        """Test expression abs."""
        if isinstance(expression, ParameterExpression):
            expr = abs(expression)
            res = expr.bind({x: 1.0 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, 1.0)
            # Test negative
            expr = abs(expression)
            res = expr.bind({x: -2.4 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, 2.4)

    @combine(expression=operands)
    def test_acos_simple(self, expression):
        """Test expression arccos."""
        if isinstance(expression, ParameterExpression):
            expr = expression.arccos()
            res = expr.bind({x: 0.2 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.acos(0.2))
            # Test negative
            expr = expression.arccos()
            res = expr.bind({x: -0.3 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.acos(-0.3))

    @combine(expression=operands)
    def test_asin_simple(self, expression):
        """Test expression arcsin."""
        if isinstance(expression, ParameterExpression):
            expr = expression.arcsin()
            res = expr.bind({x: 0.2 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.asin(0.2))
            # Test negative
            expr = expression.arcsin()
            res = expr.bind({x: -0.3 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.asin(-0.3))

    @combine(expression=operands)
    def test_atan_simple(self, expression):
        """Test expression arctan."""
        if isinstance(expression, ParameterExpression):
            expr = expression.arctan()
            res = expr.bind({x: 0.2 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.atan(0.2))
            # Test negative
            expr = expression.arctan()
            res = expr.bind({x: -0.3 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.atan(-0.3))

    @combine(expression=operands)
    def test_conjugate_simple(self, expression):
        """Test expression conjugate."""
        if isinstance(expression, ParameterExpression):
            expr = expression.conjugate()
            res = expr.bind({x: complex(1.4, 0.2) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, complex(1.4, -0.2))

    @combine(expression=operands)
    def test_conjugate_float_bind(self, expression):
        """Test expression conjugate with float binding."""
        if isinstance(expression, ParameterExpression):
            expr = expression.conjugate()
            res = expr.bind({x: 0.2 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, 0.2)

    @combine(expression=operands)
    def test_conjugate_int_bind(self, expression):
        """Test expression conjugate with int binding."""
        if isinstance(expression, ParameterExpression):
            expr = expression.conjugate()
            res = expr.bind({x: int(2) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, 2)

    @combine(expression=operands)
    def test_cos_simple(self, expression):
        """Test expression cos."""
        if isinstance(expression, ParameterExpression):
            expr = expression.cos()
            res = expr.bind({x: math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.cos(math.pi))
            # Test negative
            expr = expression.cos()
            res = expr.bind({x: -math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.cos(-math.pi))

    @combine(expression=operands)
    def test_sin_simple(self, expression):
        """Test expression sin."""
        if isinstance(expression, ParameterExpression):
            expr = expression.sin()
            res = expr.bind({x: math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.sin(math.pi))
            # Test negative
            expr = expression.sin()
            res = expr.bind({x: -math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.sin(-math.pi))

    @combine(expression=operands)
    def test_tan_simple(self, expression):
        """Test expression tan."""
        if isinstance(expression, ParameterExpression):
            expr = expression.tan()
            res = expr.bind({x: math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.tan(math.pi))
            # Test negative
            expr = expression.tan()
            res = expr.bind({x: -math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.tan(-math.pi))

    @combine(expression=operands)
    def test_exp_simple(self, expression):
        """Test expression exp."""
        if isinstance(expression, ParameterExpression):
            expr = expression.exp()
            res = expr.bind({x: math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.exp(math.pi))
            # Test negative
            expr = expression.exp()
            res = expr.bind({x: -math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.exp(-math.pi))

    @combine(expression=operands)
    def test_log_simple(self, expression):
        """Test expression log."""
        if isinstance(expression, ParameterExpression):
            expr = expression.log()
            res = expr.bind({x: math.pi for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.log(math.pi))

    @combine(expression=operands)
    def test_sign_simple(self, expression):
        """Test expression sign."""
        if isinstance(expression, ParameterExpression):
            expr = expression.sign()
            res = expr.bind({x: -0.1 for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, -1)

    @combine(
        left=operands,
        right=operands,
    )
    def test_addition_simple_complex_bind(self, left, right):
        """Test expression addition with complex bindings."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left + right
            res = expr.bind({x: complex(1.2, 1.2) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(2.4, 2.4))
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(complex(res), left + complex(1.2, 1.2))
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), right + complex(1.2, 1.2))

    @combine(
        left=operands,
        right=operands,
    )
    def test_subtraction_simple_complex_bind(self, left, right):
        """Test expression subtraction with complex binding."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left - right
            res = expr.bind({x: complex(2.4, -2.3) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, complex(0.0))
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(complex(res), left - complex(2.4, -2.3))
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(2.4, -2.3) - right)

    @combine(
        left=operands,
        right=operands,
    )
    def test_multiplication_simple_complex_bind(self, left, right):
        """Test expression multiplication with complex binding."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left * right
            res = expr.bind({x: complex(2.4, -2.3) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(2.4, -2.3) ** 2)
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(complex(res), left * complex(2.4, -2.3))
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(2.4, -2.3) * right)

    @combine(
        left=operands,
        right=operands,
    )
    def test_division_simple_complex_bind(self, left, right):
        """Test expression division with complex binding."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left / right
            res = expr.bind({x: complex(2.4, -2.3) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertEqual(res, complex(1))
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(complex(res), left / complex(2.4, -2.3))
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(2.4, -2.3) / right)
