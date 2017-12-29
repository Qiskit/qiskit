# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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

from qiskit import qasm, unroll, QuantumProgram

from .common import QiskitTestCase, Path


class TestJsonOutput(QiskitTestCase):
    """Test Json output.

    This is mostly covered in test_quantumprogram.py but will leave
    here for convenience.
    """
    def setUp(self):
        self.QASM_FILE_PATH = self._get_resource_path(
            'qasm/entangled_registers.qasm', Path.EXAMPLES)

    def test_json_output(self):
        qp = QuantumProgram()
        qp.load_qasm_file(self.QASM_FILE_PATH, name="example")

        basis_gates = []  # unroll to base gates, change to test
        unroller = unroll.Unroller(qasm.Qasm(data=qp.get_qasm("example")).parse(),
                                   unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        self.log.info('test_json_ouptut: %s', circuit)


if __name__ == '__main__':
    unittest.main()
