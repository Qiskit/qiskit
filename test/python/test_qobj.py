# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

"""Non-string identifiers for circuit and record identifiers test"""

from qiskit.qobj._qobj import QObj, QObjConfig, QObjExperiment, QObjInstruction
from .common import QiskitTestCase


class TestQObj(QiskitTestCase):
    """Tests for QObj."""
    def test_create_qobj(self):
        """Test creation of a QObj based on the individual elements."""
        instruction_1 = QObjInstruction(name='u1', qubits=[1], params=[0.4])
        instruction_2 = QObjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
        instructions = [instruction_1, instruction_2]
        experiment_1 = QObjExperiment(instructions)
        experiments = [experiment_1]
        config = QObjConfig(shots=1024, register_slots=3)
        headers = {}

        qobj = QObj(id='12345', config=config,
                    experiments=experiments, headers=headers)

        expected = {'id': '12345',
                    'headers': {},
                    'type': 'QASM',
                    'config': {'shots': 1024, 'register_slots': 3},
                    'experiments': [
                        {'instructions': [
                            {'name': 'u1', 'params': [0.4], 'qubits': [1]},
                            {'name': 'u2', 'params': [0.4, 0.2], 'qubits': [1]}
                        ]}
                    ]}
        self.assertEqual(qobj.as_dict(), expected)
