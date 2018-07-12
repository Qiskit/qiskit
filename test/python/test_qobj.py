# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment,
                         QobjInstruction, qobj_to_dict)
from .common import QiskitTestCase


class TestQobj(QiskitTestCase):
    """Tests for Qobj."""
    def test_create_qobj(self):
        """Test creation of a Qobj based on the individual elements."""
        config = QobjConfig(max_credits=10, shots=1024, register_slots=2)
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instruction_2 = QobjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
        instructions = [instruction_1, instruction_2]
        experiment_1 = QobjExperiment(instructions=instructions)
        experiments = [experiment_1]

        qobj = Qobj(id='12345', config=config, experiments=experiments, header={})
        import pprint
        pprint.pprint(qobj_to_dict(qobj))
        expected = {
            'id': '12345',
            'type': 'QASM',
            'header': {},
            'config': {'max_credits': 10, 'register_slots': 2, 'shots': 1024},
            'experiments': [
                {'instructions': [
                    {'name': 'u1', 'params': [0.4], 'qubits': [1]},
                    {'name': 'u2', 'params': [0.4, 0.2], 'qubits': [1]}
                ]}
            ],
            }

        self.assertEqual(qobj.as_dict(), expected)
