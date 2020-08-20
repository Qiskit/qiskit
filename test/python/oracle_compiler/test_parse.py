# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests the oracle parser."""

from qiskit.circuit.oracle import OracleParseError
from qiskit.circuit.oracle.oracle import OracleVisitor
from qiskit.circuit.oracle.compile_oracle import compile_oracle
from qiskit.test import QiskitTestCase
from . import bad_examples
from . import examples as good_examples


class TestParse(QiskitTestCase):
    """Tests good_examples with the oracle parser."""

    def test_identity(self):
        expected = {'name': 'FunctionDef',
                    'type': 'Int1',
                    'value': [{'name': 'Name', 'type': 'Int1', 'value': 'a'}]}
        ast = compile_oracle(good_examples.identity).ast
        self.assertEqual(ast._to_dict(), expected)

    def test_bit_and(self):
        expected = {'name': 'FunctionDef',
                    'type': 'Int1',
                    'value': [{'name': 'BitAnd',
                               'type': 'Int1',
                               'value': [{'name': 'Name', 'type': 'Int1', 'value': 'a'},
                                         {'name': 'Name', 'type': 'Int1', 'value': 'b'}]}]}
        ast = compile_oracle(good_examples.bit_and).ast
        self.assertEqual(ast._to_dict(), expected)

# class TestParseFail(QiskitTestCase):
#     """Tests bad_examples with the oracle parser."""
#
#     def assertExceptionMessage(self, context, message):
#         """Asserts the message of an exception context"""
#         self.assertTrue(message in context.exception.args[0])
#
#     def test_id_bad_return(self):
#         """Trying to parse examples.id_bad_return raises OracleParseError"""
#         with self.assertRaises(OracleParseError) as context:
#             compile_oracle(bad_examples.id_bad_return)
#         self.assertExceptionMessage(context, 'return type error')
#
#     def test_id_no_type_arg(self):
#         """Trying to parse examples.id_no_type_arg raises OracleParseError"""
#         with self.assertRaises(OracleParseError) as context:
#             compile_oracle(bad_examples.id_no_type_arg)
#         self.assertExceptionMessage(context, 'argument type is needed')
#
#     def test_id_no_type_return(self):
#         """Trying to parse examples.id_no_type_return raises OracleParseError"""
#         with self.assertRaises(OracleParseError) as context:
#             compile_oracle(bad_examples.id_no_type_return)
#         self.assertExceptionMessage(context, 'return type is needed')
#
#     def test_out_of_scope(self):
#         """Trying to parse examples.out_of_scope raises OracleParseError"""
#         with self.assertRaises(OracleParseError) as context:
#             compile_oracle(bad_examples.out_of_scope)
#         self.assertExceptionMessage(context, 'out of scope: c')
