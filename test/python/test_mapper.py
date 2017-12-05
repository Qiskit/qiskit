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

from sys import version_info
import unittest

from qiskit import QuantumProgram
from .common import QiskitTestCase


class MapperTest(QiskitTestCase):
    """Test the mapper."""

    def setUp(self):
        self.seed = 42
        self.qp = QuantumProgram()
        self.qp.enable_logs()

    def tearDown(self):
        pass

    def test_mapper_overoptimization(self):
        """
        The mapper should not change the semantics of the input. An overoptimization introduced
        the issue #81: https://github.com/QISKit/qiskit-sdk-py/issues/81 
        """
        self.qp.load_qasm_file(self._get_resource_path('qasm/overoptimization.qasm'), name='test')
        coupling_map = {0: [2], 1: [2], 2: [3], 3: []}
        result1 = self.qp.execute(["test"], backend="local_qasm_simulator", coupling_map=coupling_map)
        count1 = result1.get_counts("test")
        result2 = self.qp.execute(["test"], backend="local_qasm_simulator", coupling_map=None)
        count2 = result2.get_counts("test")
        self.assertEqual(count1.keys(), count2.keys(), )

    def test_math_domain_error(self):
        """
        The math library operates over floats and introduce floating point errors that should be avoid
        See: https://github.com/QISKit/qiskit-sdk-py/issues/111
        """
        self.qp.load_qasm_file(self._get_resource_path('qasm/math_domain_error.qasm'), name='test')
        coupling_map = {0: [2], 1: [2], 2: [3], 3: []}
        result1 = self.qp.execute(["test"], backend="local_qasm_simulator", coupling_map=coupling_map, seed=self.seed)

        # TODO: the circuit produces different results under different versions
        # of Python, which defeats the purpose of the "seed" parameter. A proper
        # fix should be issued - this is a workaround for this particular test.
        if version_info.minor == 5:  # Python 3.5
            self.assertEqual(result1.get_counts("test"), {'0001': 507, '0101': 517})
        else:  # Python 3.6 and higher
            self.assertEqual(result1.get_counts("test"), {'0001': 480, '0101': 544})

    def test_optimize_1q_gates_issue159(self):
        """
        Test change in behavior for optimize_1q_gates that removes u1(2*pi) rotations.
        See: https://github.com/QISKit/qiskit-sdk-py/issues/159
        """
        self.qp.load_qasm_file(self._get_resource_path('qasm/issue159.qasm'), name='test')
        coupling_map = {1: [0], 2: [0, 1, 4], 3: [2, 4]}
        backend = "local_qasm_simulator"
        self.log.info(self.qp.get_qasm("test"))
        qobj = self.qp.compile(["test"], backend=backend, coupling_map=coupling_map, seed=self.seed)
        out_qasm = self.qp.get_compiled_qasm(qobj, "test")
        self.log.info(out_qasm)
        self.log.info(len(out_qasm))
        self.assertEqual(len(out_qasm), 220)


if __name__ == '__main__':
    unittest.main()
