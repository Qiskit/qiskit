# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
import json
import jsonschema as jsch
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment, QobjInstruction)
from qiskit import __path__ as qiskit_path
from .common import QiskitTestCase


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

        self.expected_dict = {
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

    def tearDown(self):
        self.valid_qobj = None
        self.expected_dict = None

    def test_create_qobj_and_req_args(self):
        """Test creation of a Qobj and check for its required args."""
        self.assertTrue(
            all(getattr(self.valid_qobj, required_arg) is not None
                for required_arg in Qobj.REQUIRED_ARGS))

        with self.assertRaises(ValueError):
            Qobj(qobj_id=None, header=None, config=None, experiments=None)

    def test_as_dict(self):
        """Test conversion to dict of a Qobj based on the individual elements."""
        self.valid_qobj._version = '67890'  # private member variables shouldn't appear in the dict
        self.assertEqual(self.valid_qobj.as_dict(), self.expected_dict)

    def test_as_dict_to_json(self):
        """Test dictionary representation of Qobj against its schema."""
        file_path = self._get_resource_path('qobj_schema.json', qiskit_path.SCHEMAS)

        with open(file_path, 'r') as schema_file:
            schema = json.load(schema_file)

        jsch.validate(self.valid_qobj.as_dict(), schema)


class TestQobjConfig(QiskitTestCase):
    """Tests for QobjConfig."""
    def test_init_qobj_config(self):
        """Test initialization of a QobjConfig for required arguments."""
        shots = 1
        memory_slots = 2

        qobj_config = QobjConfig(shots=shots, memory_slots=memory_slots)

        self.assertTrue(
            all(getattr(qobj_config, required_arg) is not None
                for required_arg in QobjConfig.REQUIRED_ARGS))

        with self.assertRaises(ValueError):
            QobjConfig(shots=None, memory_slots=None)


class TestQobjExperiment(QiskitTestCase):
    """Tests for QobjExperiment."""
    def test_init_qobj_experiment(self):
        """Test initialization of a QobjExperiment for required arguments."""
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instructions = [instruction_1]

        qobj_experiment = QobjExperiment(instructions=instructions)

        self.assertTrue(
            all(getattr(qobj_experiment, required_arg) is not None
                for required_arg in QobjExperiment.REQUIRED_ARGS))

        with self.assertRaises(ValueError):
            QobjExperiment(instructions=None)


class TestQobjInstruction(QiskitTestCase):
    """Tests for QobjInstruction."""
    def test_init_qobj_instruction(self):
        """Test initialization of a QobjInstruction for required arguments."""
        name = 'name'

        qobj_instruction = QobjInstruction(name=name)

        self.assertTrue(
            all(getattr(qobj_instruction, required_arg) is not None
                for required_arg in QobjInstruction.REQUIRED_ARGS))

        with self.assertRaises(ValueError):
            QobjInstruction(name=None)
