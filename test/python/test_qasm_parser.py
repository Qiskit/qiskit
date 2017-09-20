# -*- coding: utf-8 -*-

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
import os
import unittest

import qiskit.qasm as Qasm


# TODO: Use a library to mock the fs avoiding this files.
# Note that the "example.qasm" one is used in other tests.
QASM_FILE_PATH = os.path.join(os.path.dirname(__file__), './qasm/example.qasm')
QASM_FILE_PATH_FAIL = os.path.join(os.path.dirname(__file__), './qasm/example_fail.qasm')


def parse(file_path, prec=15):
    """
      Simple helper
      - file_path: Path to the OpenQASM file
      - prec: Precision for the returned string
    """
    qasm = Qasm.Qasm(file_path)
    return qasm.parse().qasm(prec)


class TestParser(unittest.TestCase):
    """QasmParser"""

    def test_parser(self):
        """should return a correct response for a valid circuit."""

        res = parse(QASM_FILE_PATH)
        # TODO: For now only some basic checks.
        self.assertEqual(len(res), 1660)
        self.assertEqual(res[:12], "OPENQASM 2.0")
        self.assertEqual(res[14:41], "gate u3(theta,phi,lambda) q")
        self.assertEqual(res[1644:1659], "measure r -> d;")

    def test_parser_fail(self):
        """should fail a for a  not valid circuit."""

        self.assertRaisesRegex(Qasm.QasmError, "Perhaps there is a missing",
                               parse, file_path=QASM_FILE_PATH_FAIL)

if __name__ == '__main__':
    unittest.main()
