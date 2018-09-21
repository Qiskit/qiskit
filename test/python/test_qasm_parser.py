# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Test for the QASM parser"""

import unittest
import ply

from qiskit.qasm import Qasm, QasmError
from qiskit.qasm._node._node import Node
from qiskit.qasm._node import Comment

from .common import QiskitTestCase


def parse(file_path, prec=15):
    """
      Simple helper
      - file_path: Path to the OpenQASM file
      - prec: Precision for the returned string
    """
    qasm = Qasm(file_path)
    return qasm.parse().qasm(prec)


class TestParser(QiskitTestCase):
    """QasmParser"""

    def setUp(self):
        self.qasm_file_path = self._get_resource_path('qasm/example.qasm')
        self.qasm_file_path_fail = self._get_resource_path(
            'qasm/example_fail.qasm')
        self.qasm_file_path_if = self._get_resource_path(
            'qasm/example_if.qasm')

    def test_parser(self):
        """should return a correct response for a valid circuit."""

        res = parse(self.qasm_file_path)
        self.log.info(res)
        # TODO: For now only some basic checks.
        self.assertEqual(len(res), 1562)
        self.assertEqual(res[:12], "OPENQASM 2.0")
        self.assertEqual(res[14:41], "gate u3(theta,phi,lambda) q")
        self.assertEqual(res[1547:], "measure r -> d;")

    def test_parser_fail(self):
        """should fail a for a  not valid circuit."""

        self.assertRaisesRegex(QasmError, "Perhaps there is a missing",
                               parse, file_path=self.qasm_file_path_fail)

    def test_all_valid_nodes(self):
        """Test that the tree contains only Node subclasses."""

        def inspect(node):
            for child in node.children:
                self.assertTrue(isinstance(child, Node))
                inspect(child)

        # Test the canonical example file.
        qasm = Qasm(self.qasm_file_path)
        res = qasm.parse()
        inspect(res)

        # Test a file containing if instructions.
        qasm_if = Qasm(self.qasm_file_path_if)
        res_if = qasm_if.parse()
        inspect(res_if)

    def test_get_tokens(self):
        """Test whether we get only valid tokens."""
        qasm = Qasm(self.qasm_file_path)
        for token in qasm.get_tokens():
            self.assertTrue(isinstance(token, ply.lex.LexToken))

    def test_single_line_program(self):
        """The unparse should keep the single line formating"""
        qasm_string = """OPENQASM 2.0;qreg q[2];creg c0[1];"""
        ast = Qasm(data=qasm_string).parse()
        self.assertEqual(len(ast.children), 3)
        self.assertEqual(ast.qasm(), qasm_string)

class TestParserWithComments(QiskitTestCase):
    """QasmParser has a with_comment option to get have comments as extra nodes"""

    def test_single_comment(self):
        """The comment to parse is a single comment"""
        qasm_string = """//this is a comment"""
        ast = Qasm(data=qasm_string).parse(with_comments=True)
        self.assertEqual(len(ast.children), 1)
        self.assertIsInstance(ast.children[0], Comment)
        self.assertEqual(ast.qasm(), qasm_string)

    def test_inline_comment(self):
        """The comment to parse is after a statement. """
        qasm_string = """OPENQASM 2.0;//another comment"""
        ast = Qasm(data=qasm_string).parse(with_comments=True)
        self.assertEqual(len(ast.children), 2)
        self.assertIsInstance(ast.children[1], Comment)
        self.assertEqual(ast.qasm(), qasm_string)

    def test_in_decl_comment(self):
        """The comment to parse is inside a gate declaration. """
        qasm_string = ("OPENQASM 2.0;\n"
                       "gate a_gate(a,b,c) d,e\n"
                       "{\n"
                       "  // a comment in a declaration\n"
                       "  U(a,b,c) d;\n"
                       "}")
        ast = Qasm(data=qasm_string).parse(with_comments=True)
        self.assertEqual(len(ast.children), 2)
        self.assertIsInstance(ast.children[1].children[3].children[0], Comment)
        self.assertEqual(ast.qasm(), qasm_string)


if __name__ == '__main__':
    unittest.main()
