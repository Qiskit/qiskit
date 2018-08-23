# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
import unittest
import json
import copy
import jsonschema
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment, QobjInstruction, QobjHeader)
from qiskit.backends.local import localjob
from qiskit.backends.ibmq import IBMQBackend
from qiskit import QISKitError
from ._mockutils import FakeBackend, DummySimulator
from .common import QiskitTestCase, Path


class TestQobj(QiskitTestCase):
    """Tests for Qobj."""

    def setUp(self):
        self.valid_qobj = Qobj(
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

        self.valid_dict = {
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
            ],
        }

        self.bad_qobj = copy.deepcopy(self.valid_qobj)
        self.bad_qobj.experiments = None  # set experiments to None to cause the qobj to be invalid

    def test_as_dict(self):
        """Test conversion to dict of a Qobj based on the individual elements."""
        self.valid_qobj._version = '67890'  # private member variables shouldn't appear in the dict
        self.assertEqual(self.valid_qobj.as_dict(), self.valid_dict)

    def test_as_dict_against_schema(self):
        """Test dictionary representation of Qobj against its schema."""
        schema_file_path = self._get_resource_path('qobj_schema.json', Path.SCHEMAS)

        with open(schema_file_path, 'r') as schema_file:
            schema = json.load(schema_file)

        try:
            jsonschema.validate(self.valid_qobj.as_dict(), schema)
        except jsonschema.ValidationError as validation_error:
            self.fail(str(validation_error))

    @unittest.expectedFailure
    # expected to fail until _qobjectify_item is updated (see TODO_ on line 77 of _qobj.py)
    def test_from_dict_per_class(self):
        """Test Qobj and its subclass representations given a dictionary."""
        test_parameters = {
            Qobj: (
                self.valid_qobj,
                self.valid_dict
            ),
            QobjConfig: (
                QobjConfig(shots=1, memory_slots=2),
                {'shots': 1, 'memory_slots': 2}
            ),
            QobjExperiment: (
                QobjExperiment(
                    instructions=[QobjInstruction(name='u1', qubits=[1], params=[0.4])]),
                {'instructions': {'name': 'u1', 'qubits': [1], 'params': [0.4]}}
            ),
            QobjInstruction: (
                QobjInstruction(name='u1', qubits=[1], params=[0.4]),
                {'name': 'u1', 'qubits': [1], 'params': [0.4]}
            )
        }

        for qobj_class, (qobj, expected_dict) in test_parameters.items():
            with self.subTest(msg=str(qobj_class)):
                self.assertEqual(qobj, qobj_class.from_dict(expected_dict))

    def test_localjob_raises_error_when_sending_bad_qobj(self):
        """Test localjob is denied resource request access when given an invalid Qobj instance."""
        local_backend = FakeBackend()
        self.bad_qobj.header = QobjHeader(backend_name=local_backend.name)
        with self.assertRaises(QISKitError):
            localjob.LocalJob(None, self.bad_qobj)

    def test_ibmqobj_raises_error_when_sending_bad_qobj(self):
        """Test IBMQobj is denied resource request access when given an invalid Qobj instance."""
        config = DummySimulator.DEFAULT_CONFIGURATION
        non_local_backend = IBMQBackend(config)
        self.bad_qobj.header = QobjHeader(backend_name=non_local_backend.name)
        with self.assertRaises(QISKitError):
            non_local_backend.run(self.bad_qobj)
