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

import os
import unittest
import ply

from qiskit.qasm import Qasm, QasmError
from qiskit.qasm.node.node import Node
from qiskit.test import QiskitTestCase


def parse(file_path):
    """
    Simple helper
    - file_path: Path to the OpenQASM file
    - prec: Precision for the returned string
    """
    qasm = Qasm(file_path)
    return qasm.parse().qasm()


class TestParser(QiskitTestCase):
    """QasmParser"""

    def setUp(self):
        super().setUp()
        self.qasm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qasm")
        self.qasm_file_path = os.path.join(self.qasm_dir, "example.qasm")
        self.qasm_file_path_fail = os.path.join(self.qasm_dir, "example_fail.qasm")
        self.qasm_file_path_if = os.path.join(self.qasm_dir, "example_if.qasm")

    def test_parser(self):
        """should return a correct response for a valid circuit."""

        res = parse(self.qasm_file_path)
        self.log.info(res)
        # TODO: For now only some basic checks.
        starts_expected = "OPENQASM 2.0;\ngate "
        ends_expected = "\n".join(
            [
                "}",
                "qreg q[3];",
                "qreg r[3];",
                "h q;",
                "cx q,r;",
                "creg c[3];",
                "creg d[3];",
                "barrier q;",
                "measure q -> c;",
                "measure r -> d;",
                "",
            ]
        )

        self.assertEqual(res[: len(starts_expected)], starts_expected)
        self.assertEqual(res[-len(ends_expected) :], ends_expected)

    def test_parser_fail(self):
        """should fail a for a  not valid circuit."""

        self.assertRaisesRegex(
            QasmError, "Perhaps there is a missing", parse, file_path=self.qasm_file_path_fail
        )

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

    def test_generate_tokens(self):
        """Test whether we get only valid tokens."""
        qasm = Qasm(self.qasm_file_path)
        for token in qasm.generate_tokens():
            self.assertTrue(isinstance(token, ply.lex.LexToken))


if __name__ == "__main__":
    unittest.main()
