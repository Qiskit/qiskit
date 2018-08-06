# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
import json
import jsonschema
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment, QobjInstruction)
from .common import QiskitTestCase, Path


class TestQobj(QiskitTestCase):
    """Tests for Qobj."""

    def setUp(self):
        self._valid_qobj = Qobj(
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
        self._valid_qobj._version = '67890'  # private member variables shouldn't appear in the dict

    def test_init_qobj(self):
        """Test initialization of a Qobj for required arguments."""
        with self.assertRaises(TypeError):
            # pylint: disable=no-value-for-parameter
            Qobj()

    def test_as_dict(self):
        """Test conversion to dict of a Qobj based on the individual elements."""
        expected_dict = {
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

        self.assertEqual(self._valid_qobj.as_dict(), expected_dict)

    def test_as_dict_against_schema(self):
        """Test dictionary representation of Qobj against its schema."""
        file_path = self._get_resource_path('qobj_schema.json', Path.SCHEMAS)

        with open(file_path, 'r') as schema_file:
            schema = json.load(schema_file)

        passed = False
        try:
            jsonschema.validate(self._valid_qobj.as_dict(), schema)
            passed = True
        except jsonschema.ValidationError as err:
            self.log.info('Error validating Qobj to_dict with JSON schema: %s', err)

        self.assertTrue(passed)


class TestQobjConfig(QiskitTestCase):
    """Tests for QobjConfig."""
    def test_init_qobj_config(self):
        """Test initialization of a QobjConfig for required arguments."""
        with self.assertRaises(TypeError):
            # pylint: disable=no-value-for-parameter
            QobjConfig()


class TestQobjExperiment(QiskitTestCase):
    """Tests for QobjExperiment."""
    def test_init_qobj_experiment(self):
        """Test initialization of a QobjExperiment for required arguments."""
        with self.assertRaises(TypeError):
            # pylint: disable=no-value-for-parameter
            QobjExperiment()


class TestQobjInstruction(QiskitTestCase):
    """Tests for QobjInstruction."""
    def test_init_qobj_instruction(self):
        """Test initialization of a QobjInstruction for required arguments."""
        with self.assertRaises(TypeError):
            # pylint: disable=no-value-for-parameter
            QobjInstruction()
