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

"""Test for the QASM parser"""

import unittest
import ply

from qiskit.qasm import Qasm, QasmError
from qiskit.qasm.node.node import Node
from qiskit.test import QiskitTestCase, Path


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
        self.qasm_file_path = self._get_resource_path('example.qasm', Path.QASMS)
        self.qasm_file_path_fail = self._get_resource_path(
            'example_fail.qasm', Path.QASMS)
        self.qasm_file_path_if = self._get_resource_path(
            'example_if.qasm', Path.QASMS)

    def test_parser(self):
        """should return a correct response for a valid circuit."""

        res = parse(self.qasm_file_path)
        self.log.info(res)
        # TODO: For now only some basic checks.
        self.assertEqual(len(res), 1563)
        self.assertEqual(res[:12], "OPENQASM 2.0")
        self.assertEqual(res[14:41], "gate u3(theta,phi,lambda) q")
        self.assertEqual(res[1547:1562], "measure r -> d;")

    def test_parser_fail(self):
        """should fail a for a  not valid circuit."""

        self.assertRaisesRegex(QasmError, "Perhaps there is a missing",
                               parse, file_path=self.qasm_file_path_fail)

    def test_all_valid_nodes(self):
        """Test that the tree contains only Node subclasses."""
        def inspect(node):
            """Inspect node children."""
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


if __name__ == '__main__':
    unittest.main()
