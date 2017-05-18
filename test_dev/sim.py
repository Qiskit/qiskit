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


"""Quick test program for unitary simulator backend."""
import unittest


from qiskit.qasm import Qasm
from qiskit.simulators._unitarysimulator import UnitarySimulator
import qiskit.unroll as unroll

class TestQISKitSIM(unittest.TestCase):
    """
    Test UnitarySimulator
    """

    def test_UnitarySimulator(self):
        basis = []  # empty basis, defaults to U, CX
        unroller = unroll.Unroller(Qasm(filename="test/test.qasm").parse(),
                                   UnitarySimulator(basis))
        unroller.backend.set_trace(True)  # print calls as they happen
        result = unroller.execute()  # Here is where simulation happens
        print(result)
        self.assertEqual(result, 'TODO: check result')

if __name__ == '__main__':
    unittest.main()
