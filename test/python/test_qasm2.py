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
import unittest

from qiskit.qasm2 import load, unload
from qiskit.test import QiskitTestCase, Path


class TestQasm2(QiskitTestCase):
    """Test Qasm2 functional interface to QASM loading/unloading"""

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

        # The file with unload output from QuantumCircuit
        # It differs in order from the original source file.
        self._circ_unload_path = self._get_resource_path('output/yiqing_circ_unload.txt',
                                                         Path.QASM2)
        y_f = open(self._circ_unload_path, 'r')
        self._circ_unload = y_f.read()
        y_f.close()

        y_f = open(self.qasm_file_path, 'r')
        lines = y_f.read()
        y_f.close()
        lines_list = lines.split(os.linesep)

        # qasm string loaded via qiskit qasm
        self.c_0 = load(data=lines)

        # qasm list of string loaded via qiskit qasm
        self.c_0a = load(data=lines_list)

        self.c_0_unloaded = unload(self.c_0)
        self.c_0a_unloaded = unload(self.c_0a)

        # x = open('foo.txt', 'w')
        # print(self.c_0_unloaded, file=x)
        # x.close()

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

    def test_unload(self):
        """Test Qasm2 unload()"""

        self.assertEqual(self.c_0_unloaded, self.c_0a_unloaded,
                         "Error: Circuits c_0 and c_0a don't unload the same.")

        s_unload_unmatch = "Error:\n{}\ndoesn't match\n{}"
        self.assertEqual(self.c_0_unloaded,
                         self._circ_unload,
                         s_unload_unmatch.format(self.c_0_unloaded,
                                                 self._circ_unload))


if __name__ == '__main__':
    unittest.main()
