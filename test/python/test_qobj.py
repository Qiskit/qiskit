# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""
from qiskit.qobj import (Qobj, QobjConfig, QobjExperiment,
                         QobjInstruction)
from .common import QiskitTestCase
from qiskit import __path__ as qiskit_path
import os
import json
import jsonschema as jsch


class TestQobj(QiskitTestCase):
    """Tests for Qobj."""
    def test_init_qobj(self):
        """Test initialization of a Qobj for required arguments."""
        config = QobjConfig(max_credits=10, shots=1024, memory_slots=2)
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instructions = [instruction_1]
        experiment_1 = QobjExperiment(instructions=instructions)
        experiments = [experiment_1]

        qobj = Qobj(id='12345', config=config, experiments=experiments, header={})

        self.assertTrue(
            all(getattr(qobj, required_arg) is not None for required_arg in Qobj.REQUIRED_ARGS))

    def test_as_dict(self):
        """Test conversion to dict of a Qobj based on the individual elements."""
        config = QobjConfig(max_credits=10, shots=1024, memory_slots=2)
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instruction_2 = QobjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
        instructions = [instruction_1, instruction_2]
        experiment_1 = QobjExperiment(instructions=instructions)
        experiments = [experiment_1]

        qobj = Qobj(id='12345', config=config, experiments=experiments, header={})
        qobj._version = '67890'  # private member variables shouldn't appear in the dict, edit this to verify that

        expected = {
            'id': '12345',
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

        self.assertEqual(qobj.as_dict(), expected)

    def test_as_dict_to_json(self):

        """Test conversion to dict of a Qobj based on the individual elements."""
        config = QobjConfig(max_credits=10, shots=1024, memory_slots=2)
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instruction_2 = QobjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
        instructions = [instruction_1, instruction_2]
        experiment_1 = QobjExperiment(instructions=instructions)
        experiments = [experiment_1]

        qobj = Qobj(id='12345', config=config, experiments=experiments, header={})
        qobj._version = '67890'  # private member variables shouldn't appear in the dict, edit this to verify that

        expected = {
            'id': '12345',
            'type': 'QASM',
            'header': {},
            'config': {'max_credits': 10, 'shots': 1024},
            'schema_version': '1.0.0',
            'experiments': [
                {'instructions': [
                    {'name': 'u1', 'params': [0.4], 'qubits': [1]},
                    {'name': 'u2', 'params': [0.4, 0.2], 'qubits': [1]}
                ]}
            ],
            }

        # Main SDK path:    qiskit/
        SDK = qiskit_path[0]
        # Schemas path:     qiskit/schemas
        SCHEMAS = os.path.join(SDK, 'schemas')
        # Schema name: qobj_schema.json
        FILE = os.path.join(SCHEMAS, 'qobj_schema.json')
        f = open(FILE, 'r')

        schema = json.load(f)
        example = json.dumps(qobj.as_dict())
        try:
            jsch.validate(example, schema)
        except jsch.ValidationError as err:
            print("Error on example %s:" % 'test_as_dict_to_json')
            print(err)
        f.close()

    def test_expand_item(self):
        """Test distinct cases of _expand_item."""
        single_obj = 1
        single_list = [1, 2]
        nested_list = [[1, 2], ['a', 'b']]

        self.assertEqual(Qobj._expand_item(single_obj), single_obj)
        self.assertEqual(Qobj._expand_item(single_list), single_list)
        self.assertEqual(Qobj._expand_item(nested_list), nested_list)

        config = QobjConfig(max_credits=10, shots=1024, memory_slots=2)
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instruction_2 = QobjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
        instructions = [instruction_1, instruction_2]
        experiment_1 = QobjExperiment(instructions=instructions)
        experiments = [experiment_1]

        qobj = Qobj(id='12345', config=config, experiments=experiments, header={})

        expected_dict = {
            'id': '12345',
            'type': 'QASM',
            'header': {},
            'config': {'max_credits': 10, 'memory_slots': 2, 'shots': 1024},
            'schema_version': '1.0.0',
            'experiments': [
                {'instructions': [
                    {'name': 'u1', 'params': [0.4], 'qubits': [1]},
                    {'name': 'u2', 'params': [0.4, 0.2], 'qubits': [1]}
                ]}
            ],
        }

        self.assertEqual(Qobj._expand_item(qobj), expected_dict)


class TestQobjConfig(QiskitTestCase):
    """Tests for QobjConfig."""
    def test_init_QobjConfig(self):
        """Test initialization of a QobjConfig for required arguments."""
        shots = 1
        memory_slots = 2

        qobj_config = QobjConfig(shots=shots, memory_slots=memory_slots)

        self.assertTrue(
            all(getattr(qobj_config, required_arg) is not None for required_arg in QobjConfig.REQUIRED_ARGS))


class QobjHeader(QiskitTestCase):
    """Tests for QobjHeader."""
    pass


class TestQobjExperiment(QiskitTestCase):
    """Tests for QobjExperiment."""
    def test_init_QobjExperiment(self):
        """Test initialization of a QobjExperiment for required arguments."""
        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instructions = [instruction_1]

        qobj_experiment = QobjExperiment(instructions=instructions)

        self.assertTrue(
            all(getattr(qobj_experiment, required_arg) is not None for required_arg in QobjExperiment.REQUIRED_ARGS))


class QobjExperimentHeader(QiskitTestCase):
    """Tests for QobjExperimentHeader."""
    pass


class TestQobjInstruction(QiskitTestCase):
    """Tests for QobjInstruction."""
    def test_init_QobjInstruction(self):
        """Test initialization of a QobjInstruction for required arguments."""
        name = 'name'

        qobj_instruction = QobjInstruction(name=name)

        self.assertTrue(
            all(getattr(qobj_instruction, required_arg) is not None for required_arg in QobjInstruction.REQUIRED_ARGS))
