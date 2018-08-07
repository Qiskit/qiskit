# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
import unittest
import json
import jsonschema
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment, QobjInstruction)
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

    def test_as_dict(self):
        """Test conversion to dict of a Qobj based on the individual elements."""
        self.valid_qobj._version = '67890'  # private member variables shouldn't appear in the dict
        self.assertEqual(self.valid_qobj.as_dict(), self.valid_dict)

    def test_as_dict_against_schema(self):
        """Test dictionary representation of Qobj against its schema."""
        file_path = self._get_resource_path('qobj_schema.json', Path.SCHEMAS)

        with open(file_path, 'r') as schema_file:
            schema = json.load(schema_file)

        try:
            jsonschema.validate(self.valid_qobj.as_dict(), schema)
        except jsonschema.ValidationError as validation_error:
            self.fail(str(validation_error))

    @unittest.expectedFailure  # expected to fail until _qobjectify_item is updated
    def test_from_dict_per_class(self):
        """Test Qobj and its subclass representations given a dictionary."""
        qobj_config_dict = {'shots': 1, 'memory_slots': 2}
        qobj_experiment_dict = {'instructions':
                                [QobjInstruction(name='u1', qubits=[1], params=[0.4])]}
        qobj_instruction_dict = {'name': 'u1', 'qubits': [1], 'params': [0.4]}

        valid_qobj_config = QobjConfig(shots=1024, memory_slots=2, max_credits=10)
        valid_qobj_experiment = QobjExperiment(
            instructions=[QobjInstruction(name='u1', qubits=[1], params=[0.4])]
        )
        valid_qobj_instruction = QobjInstruction(name='u1', qubits=[1], params=[0.4])

        dicts = [self.valid_dict, qobj_config_dict, qobj_experiment_dict, qobj_instruction_dict]
        objects = [self.valid_qobj, valid_qobj_config,
                   valid_qobj_experiment, valid_qobj_instruction]
        classes = [Qobj, QobjConfig, QobjExperiment, QobjInstruction]

        for objekt, dikt, klass in zip(objects, dicts, classes):
            with self.subTest(msg=str(klass) + ' failed from_dict test.'):
                if objekt != klass.from_dict(dikt):
                    self.assertEqual(objekt, klass.from_dict(dikt))
