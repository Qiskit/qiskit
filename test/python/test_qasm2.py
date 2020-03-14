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
        self.qasm_file_path = self._get_resource_path('example.qasm',
                                                      Path.QASMS)

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

        self.assertEqual(c_0, c_0a)
        self.assertEqual(unload(c_0), unload(c_0a))

if __name__ == '__main__':
    unittest.main()
