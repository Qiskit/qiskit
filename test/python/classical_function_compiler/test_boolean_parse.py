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

"""Tests the BooleanExpression parser."""

import ast
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.circuit.classicalfunction.boolean_expression_visitor import (
    BooleanExpressionEvalVisitor,
    BooleanExpressionArgsCollectorVisitor,
)
from qiskit.circuit.classicalfunction.exceptions import BooleanExpressionParseError


class TestBooleanParse(QiskitTestCase):
    """Tests the boolean experssion parsers"""

    def test_args_collection(self):
        """Tests that args are collected based of appearance in the string"""
        exp = "x2 & x1"
        args_collector = BooleanExpressionArgsCollectorVisitor()
        args_collector.visit(ast.parse(exp))
        res = args_collector.get_sorted_args()
        expected_result = ["x2", "x1"]
        self.assertEqual(res, expected_result)

        exp = "(x3 | x1) ^ x2 & (x2 | x3)"  # some vars appear twice
        args_collector = BooleanExpressionArgsCollectorVisitor()
        args_collector.visit(ast.parse(exp))
        res = args_collector.get_sorted_args()
        expected_result = ["x3", "x1", "x2"]
        self.assertEqual(res, expected_result)

    def test_unrecognized_op(self):
        """Boolean expression has an unknown op"""
        exp = "x1 + x2"  # + should not be recognized as it is not a boolean op
        parser = BooleanExpressionEvalVisitor()
        parser.arg_values = {"x1": True, "x2": True}
        with self.assertRaisesRegex(BooleanExpressionParseError, "Unknown op.*Add"):
            parser.visit(ast.parse(exp))

    def test_undefined_var(self):
        """Boolean expression has an variable with no known value"""
        exp = "x1 & z2"  # value for z2 is unknown
        parser = BooleanExpressionEvalVisitor()
        parser.arg_values = {"x1": True, "x2": True}
        with self.assertRaisesRegex(BooleanExpressionParseError, "Undefined value.*z2"):
            parser.visit(ast.parse(exp))

    def test_incorrect_format(self):
        """Parsed string is not a python expression"""
        exp = "return x5"  # statement, not expression
        parser = BooleanExpressionEvalVisitor()
        with self.assertRaisesRegex(
            BooleanExpressionParseError, "Incorrectly formatted boolean expression"
        ):
            parser.visit(ast.parse(exp))
