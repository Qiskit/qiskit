# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment,
                         QobjInstruction)
from .common import QiskitTestCase


class TestQobj(QiskitTestCase):
    """Tests for Qobj."""

    def test_create_qobj(self):
        """Test creation of a Qobj based on the individual elements."""
        qobj = Qobj(
            qobj_id='12345',
            header={},
            config=QobjConfig(shots=1024, memory_slots=2, max_credits=10),
            experiments=[
                QobjExperiment(instructions=[
                    QobjInstruction(name='u1', qubits=[1], params=[0.4]),
                    QobjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
                ])
            ]
        )

        expected = {
            'qobj_id': '12345',
            'type': 'QASM',
            'schema_version': '1.0.0',
            'header': {},
            'config': {'max_credits': 10, 'memory_slots': 2, 'shots': 1024},
            'experiments': [
                {'instructions': [
                    {'name': 'u1', 'params': [0.4], 'qubits': [1]},
                    {'name': 'u2', 'params': [0.4, 0.2], 'qubits': [1]}
                ]}
            ]
        }

        self.assertEqual(qobj.as_dict(), expected)
