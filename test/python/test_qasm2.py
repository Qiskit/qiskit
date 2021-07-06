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

"""Test Qasm2 functional interface to QASM loading"""

import os
import tempfile
import unittest

from qiskit.qasm2.functions import load, dump
from qiskit.test import QiskitTestCase


class TestQasm2(QiskitTestCase):
    """Test Qasm2 functional interface to QASM loading/exporting"""

    def setUp(self):
        super().setUp()

        # Files used in test are in a subdir
        _qasm2_testfiles_path = os.path.join(os.path.dirname(__file__), "qasm2")
        _qasm2_outputfiles_path = os.path.join(os.path.dirname(__file__), "qasm2", "output")

        # Source file path
        self.qasm_file_path = os.path.join(_qasm2_testfiles_path, "yiqing.qasm")
        # Captured circuit draw
        self._circ_draw_path = os.path.join(_qasm2_outputfiles_path, "yiqing_circ_draw.txt")

        with open(self._circ_draw_path, "r") as y_f:
            self._circ_draw = y_f.read()

        # The file with export output from QuantumCircuit
        # It differs in order from the original source file.
        self._circ_export_path = os.path.join(_qasm2_outputfiles_path, "yiqing_circ_export.txt")

        with open(self._circ_export_path, "r") as y_f:
            self._circ_export = y_f.read()

        with open(self.qasm_file_path, "r") as y_f:
            lines = y_f.read()

        # file was prepared on Linux, so it's '\n' not os.linesep
        lines_list = lines.split("\n")

        # qasm string loaded via qiskit qasm
        self.c_0 = load(data=lines)

        # qasm list of string loaded via qiskit qasm
        self.c_0a = load(data=lines_list)

        self.temp_file = tempfile.TemporaryFile(mode="w+t")
        self.c_0_exported = dump(self.c_0, file=self.temp_file)

        self.temp_bfile = tempfile.TemporaryFile(mode="w+b")
        self.c_0_b_exported = dump(self.c_0, file=self.temp_bfile)

        self.c_0a_exported = dump(self.c_0a)

    def tearDown(self):
        super().tearDown()
        self.temp_file.close()
        self.temp_bfile.close()

    def test_load(self):
        """Test Qasm2 load()"""

        self.assertTrue(self.c_0, "Error: Circuit c_0 was not generated.")
        self.assertTrue(self.c_0a, "Error: Circuit c_0a was not generated.")

        self.assertEqual(self.c_0, self.c_0a, "Error: Circuits c_0 and c_0a are not the same.")

        s_draw_no_match = "Error: Circuit c_0 draw doesn't match {}"
        self.assertEqual(
            str(self.c_0.draw()), self._circ_draw, s_draw_no_match.format(self._circ_draw_path)
        )

    def test_export(self):
        """Test Qasm2 export()"""

        self.assertEqual(
            self.c_0_exported,
            self.c_0a_exported,
            "Error: Circuits c_0 and c_0a don't export the same.",
        )

        s_export_unmatch = "Error:\n{}\ndoesn't match\n{}"
        self.assertEqual(
            self.c_0_exported,
            self._circ_export,
            s_export_unmatch.format(self.c_0_exported, self._circ_export),
        )

        self.temp_file.seek(0)
        lines = self.temp_file.read()

        err_string = "Error:\nCircuit c_0\n{}\nand text file\n{}\naren't the same."
        self.assertEqual(self.c_0_exported, lines, err_string.format(self.c_0_exported, lines))

        self.temp_bfile.seek(0)
        lines = self.temp_bfile.read().decode("utf-8")

        err_string = "Error:\nCircuit c_0\n{}\nand binary file\n{}\naren't export the same."
        self.assertEqual(self.c_0_b_exported, lines, err_string.format(self.c_0_b_exported, lines))


if __name__ == "__main__":
    unittest.main()
