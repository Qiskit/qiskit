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

"""
Quick test program for "one register" backend.

Author: Andrew Cross
"""
import unittest

import qiskit.unroll as unroll
from qiskit.qasm import Qasm

class TestOneRegister(unittest.TestCase):
    """
    Test "one register" backend
    """
    def test_one_registe(self):
        fname = "test/test.qasm"
        basis = []  # empty basis, defaults to U, CX
        unroller = unroll.Unroller(Qasm(filename=fname).parse(),
                                   unroll.OneRegisterBackend(basis))

        result = unroller.execute().qasm(qeflag=True)
        self.assertEqual(len(result), 124)

if __name__ == '__main__':
    unittest.main()
