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
        self.qasm_file_path = self._get_resource_path('yiqing.qasm',
                                                      Path.QASM2)
        self.yiqing_circ_draw_path = self._get_resource_path('output/yiqing_circ_draw.txt',
                                                             Path.QASM2)
        y_f = open(self.yiqing_circ_draw_path, 'r')
        self.yiqing_circ_draw = y_f.read()
        y_f.close()

    def test_load_unload(self):
        """Test Qasm2 load() and unload()"""
        ffile = open(self.qasm_file_path, 'r')
        lines = ffile.read()
        ffile.close()
        lines_list = lines.split(os.linesep)

        # qasm string loaded via qiskit qasm
        c_0 = load(data=lines)
        # print(c0.draw())

        # qasm list of string loaded via qiskit qasm'
        c_0a = load(data=lines_list)
        # print(c0a.draw())

        c_0_unloaded = unload(c_0)
        c_0a_unloaded = unload(c_0a)

        self.assertTrue(c_0, "Error: Circuit c_0 was not generated.")
        self.assertTrue(c_0a, "Error: Circuit c_0a was not generated.")

        self.assertEqual(c_0, c_0a,
                         "Error: Circuits c_0 and c_0a are not the same.")

        self.assertEqual(str(c_0.draw()),
                         self.yiqing_circ_draw,
                         "Error: Circuit c_0 draw doesn't match {}".format(self.yiqing_circ_draw_path))  #pylint: disable=line-too-long

        self.assertEqual(c_0_unloaded, c_0a_unloaded,
                         "Error: Circuits c_0 and c_0a don't unload the same.")
        # self.assertEqual(c_0_unloaded, lines,
        #                  "Error: Circuit c_0 unloaded doesn't match{}".format(self.qasm_file_path))   #pylint: disable=line-too-long

if __name__ == '__main__':
    unittest.main()
