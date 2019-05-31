# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parser Test."""

from qiskit.test import QiskitTestCase
from qiskit.pulse.parser import parse_string_expr
from qiskit import QiskitError


class TestInstructionToQobjConverter(QiskitTestCase):
    """Expression parser test."""

    def test_valid_expression1(self):
        """Parsing valid expression."""

        expr = '1+1*2*3.2+8*cos(0)**2'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, [])
        self.assertEqual(parsed_expr(), 15.4+0j)

    def test_valid_expression2(self):
        """Parsing valid expression."""

        expr = 'pi*2'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, [])
        self.assertEqual(parsed_expr(), 6.283185307179586+0j)

    def test_valid_expression3(self):
        """Parsing valid expression."""

        expr = '-P1*cos(P2)'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, ['P1', 'P2'])
        self.assertEqual(parsed_expr(P1=1.0, P2=2.0), 0.4161468365471424 + 0j)

    def test_valid_expression4(self):
        """Parsing valid expression."""

        expr = '-P1*P2*P3'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, ['P1', 'P2', 'P3'])
        self.assertEqual(parsed_expr(P1=1.0, P2=2.0, P3=3.0), -6.0 + 0j)

    def test_valid_expression5(self):
        """Parsing valid expression."""

        expr = '-(P1)'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, ['P1'])
        self.assertEqual(parsed_expr(P1=1.0), -1.0 + 0j)

    def test_valid_expression6(self):
        """Parsing valid expression."""

        expr = '-1.*P1'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, ['P1'])
        self.assertEqual(parsed_expr(P1=1.0), -1.0 + 0j)

    def test_valid_expression7(self):
        """Parsing valid expression."""

        expr = '-1.*P1*P2'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, ['P1', 'P2'])
        self.assertEqual(parsed_expr(P1=1.0, P2=2.0), -2.0 + 0j)

    def test_valid_expression8(self):
        """Parsing valid expression."""

        expr = 'P3-P2*(4+P1)'
        parsed_expr, params = parse_string_expr(expr)

        self.assertEqual(params, ['P1', 'P2', 'P3'])
        self.assertEqual(parsed_expr(P1=1, P2=2, P3=3), -7.0 + 0j)

    def test_invalid_expressions1(self):
        """Parsing invalid expressions."""

        expr = '2***2'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions2(self):
        """Parsing invalid expressions."""

        expr = 'avdfd*3'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions3(self):
        """Parsing invalid expressions."""

        expr = 'Cos(1+2)'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions4(self):
        """Parsing invalid expressions."""

        expr = 'hello_world'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions5(self):
        """Parsing invalid expressions."""

        expr = '1.1.1.1'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions6(self):
        """Parsing invalid expressions."""

        expr = 'abc.1'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions1(self):
        """Parsing malicious expressions."""

        expr = '__import__("sys").stdout.write("unsafe input.")'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions2(self):
        """Parsing malicious expressions."""

        expr = 'INSERT INTO students VALUES (?,?)'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions3(self):
        """Parsing malicious expressions."""

        expr = 'import math'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions4(self):
        """Parsing malicious expressions."""

        expr = 'complex'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions5(self):
        """Parsing malicious expressions."""

        expr = 'print(1.0)'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions6(self):
        """Parsing malicious expressions."""

        expr = 'eval("()._" + "_class_" + "_._" +  "_bases_" + "_[0]")'
        with self.assertRaises(QiskitError):
            parsed_expr, _ = parse_string_expr(expr)
            parsed_expr()
