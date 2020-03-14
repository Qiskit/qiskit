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

"""Test Qasm2 functional interface to QASM loading"""

import os
import tempfile
import unittest

from qiskit.qasm2 import load, export
from qiskit.test import QiskitTestCase, Path


class TestQasm2(QiskitTestCase):
    """Test Qasm2 functional interface to QASM loading/exporting"""

    def setUp(self):

        # Source file path
        self.qasm_file_path = self._get_resource_path('yiqing.qasm',
                                                      Path.QASM2)
        # Captured circuit draw
        self._circ_draw_path = self._get_resource_path('output/yiqing_circ_draw.txt',
                                                       Path.QASM2)
        y_f = open(self._circ_draw_path, 'r')
        self._circ_draw = y_f.read()
        y_f.close()

        # The file with export output from QuantumCircuit
        # It differs in order from the original source file.
        self._circ_export_path = self._get_resource_path('output/yiqing_circ_export.txt',
                                                         Path.QASM2)
        y_f = open(self._circ_export_path, 'r')
        self._circ_export = y_f.read()
        y_f.close()

        y_f = open(self.qasm_file_path, 'r')
        lines = y_f.read()
        y_f.close()
        lines_list = lines.split(os.linesep)

        # qasm string loaded via qiskit qasm
        self.c_0 = load(data=lines)

        # qasm list of string loaded via qiskit qasm
        self.c_0a = load(data=lines_list)

        self.temp_file = tempfile.TemporaryFile(mode='w+t')
        self.c_0_exported = export(self.c_0, file=self.temp_file)

        self.temp_bfile = tempfile.TemporaryFile(mode='w+b')
        self.c_0_b_exported = export(self.c_0, file=self.temp_bfile)


        self.c_0a_exported = export(self.c_0a)

    def tearDown(self):
        self.temp_file.close()
        self.temp_bfile.close()

    def test_load(self):
        """Test Qasm2 load()"""

        self.assertTrue(self.c_0, "Error: Circuit c_0 was not generated.")
        self.assertTrue(self.c_0a, "Error: Circuit c_0a was not generated.")

        self.assertEqual(self.c_0, self.c_0a,
                         "Error: Circuits c_0 and c_0a are not the same.")

        s_draw_no_match = "Error: Circuit c_0 draw doesn't match {}"
        self.assertEqual(str(self.c_0.draw()),
                         self._circ_draw,
                         s_draw_no_match.format(self._circ_draw_path))

    def test_export(self):
        """Test Qasm2 export()"""

        self.assertEqual(self.c_0_exported, self.c_0a_exported,
                         "Error: Circuits c_0 and c_0a don't export the same.")

        s_export_unmatch = "Error:\n{}\ndoesn't match\n{}"
        self.assertEqual(self.c_0_exported,
                         self._circ_export,
                         s_export_unmatch.format(self.c_0_exported,
                                                 self._circ_export))

        self.temp_file.seek(0)
        lines = self.temp_file.read()

        err_string = "Error:\nCircuit c_0\n{}\nand text file\n{}\naren't the same."
        self.assertEqual(self.c_0_exported, lines,
                         err_string.format(self.c_0_exported, lines))

        self.temp_bfile.seek(0)
        lines = self.temp_bfile.read().decode('utf-8')

        err_string = "Error:\nCircuit c_0\n{}\nand binary file\n{}\naren't export the same."
        self.assertEqual(self.c_0_b_exported, lines,
                         err_string.format(self.c_0_b_exported, lines))


if __name__ == '__main__':
    unittest.main()
