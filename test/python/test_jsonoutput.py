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

"""Quick program to test json backend
"""
import unittest
import logging
import os
from qiskit import QuantumProgram
import qiskit.qasm as qasm
import qiskit.unroll as unroll

class TestJsonOutput(unittest.TestCase):
    """Test Json output.

    This is mostly covered in test_quantumprogram.py but will leave
    here for convenience.
    """
    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.logFileName = cls.moduleName + '.log'
        log_fmt = 'TestJsonOutput:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)

    def setUp(self):
        self.QASM_FILE_PATH = os.path.join(os.path.dirname(__file__),
                                           '../../examples/qasm/entangled_registers.qasm')

    def test_json_output(self):
        seed = 88
        qp = QuantumProgram()
        qp.load_qasm_file(self.QASM_FILE_PATH, name="example")

        basis_gates = []  # unroll to base gates, change to test
        unroller = unroll.Unroller(qasm.Qasm(data=qp.get_qasm("example")).parse(),
                                   unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        logging.info('test_json_ouptut: {0}'.format(circuit))


if __name__ == '__main__':
    unittest.main()
