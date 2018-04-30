# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Test for the QASM parser"""

import unittest
import ply

from qiskit.qasm import Qasm, QasmError
from qiskit.qasm._node._node import Node

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
        self.QASM_FILE_PATH = self._get_resource_path('qasm/example.qasm')
        self.QASM_FILE_PATH_FAIL = self._get_resource_path(
            'qasm/example_fail.qasm')
        self.QASM_FILE_PATH_IF = self._get_resource_path(
            'qasm/example_if.qasm')

    def test_parser(self):
        """should return a correct response for a valid circuit."""

        res = parse(self.QASM_FILE_PATH)
        self.log.info(res)
        # TODO: For now only some basic checks.
        self.assertEqual(len(res), 1563)
        self.assertEqual(res[:12], "OPENQASM 2.0")
        self.assertEqual(res[14:41], "gate u3(theta,phi,lambda) q")
        self.assertEqual(res[1547:1562], "measure r -> d;")

    def test_parser_fail(self):
        """should fail a for a  not valid circuit."""

        self.assertRaisesRegex(QasmError, "Perhaps there is a missing",
                               parse, file_path=self.QASM_FILE_PATH_FAIL)

    def test_all_valid_nodes(self):
        """Test that the tree contains only Node subclasses."""
        def inspect(node):
            for child in node.children:
                self.assertTrue(isinstance(child, Node))
                inspect(child)

        # Test the canonical example file.
        qasm = Qasm(self.QASM_FILE_PATH)
        res = qasm.parse()
        inspect(res)

        # Test a file containing if instructions.
        qasm_if = Qasm(self.QASM_FILE_PATH_IF)
        res_if = qasm_if.parse()
        inspect(res_if)

    def test_get_tokens(self):
        """Test whether we get only valid tokens."""
        qasm = Qasm(self.QASM_FILE_PATH)
        for token in qasm.get_tokens():
            self.assertTrue(isinstance(token, ply.lex.LexToken))


if __name__ == '__main__':
    unittest.main()
