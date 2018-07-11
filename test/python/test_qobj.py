# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QOBj test."""

from qiskit.qobj import (Qobj, QobjCompiledCircuit, QobjConfig, QobjExperiment,
                         QobjExperimentConfig, QobjInstruction)
from .common import QiskitTestCase


class TestQobj(QiskitTestCase):
    """Tests for Qobj."""
    def test_create_qobj(self):
        """Test creation of a Qobj based on the individual elements."""
        config = QobjConfig(max_credits=10, shots=1024, backend_name='backend')
        circuit_config = QobjExperimentConfig(
            seed=1234, basis_gates='u1,u2,u3,cx,id,snapshot',
            coupling_map=None, layout=None)
        circuit_config.seed = 1234

        instruction_1 = QobjInstruction(name='u1', qubits=[1], params=[0.4])
        instruction_2 = QobjInstruction(name='u2', qubits=[1], params=[0.4, 0.2])
        instructions = [instruction_1, instruction_2]
        compiled_circuit = QobjCompiledCircuit(header=None, operations=instructions)
        experiment_1 = QobjExperiment(name='circuit1', config=circuit_config,
                                      compiled_circuit=compiled_circuit,
                                      compiled_circuit_qasm='compiled_qasm')
        experiments = [experiment_1]

        qobj = Qobj(id='12345', config=config, circuits=experiments)

        expected = {'id': '12345',
                    'type': 'QASM',
                    'config': {'max_credits': 10, 'shots': 1024,
                               'backend_name': 'backend'},
                    'circuits': [
                        {
                            'name': 'circuit1',
                            'config': {
                                'seed': 1234, 'basis_gates': 'u1,u2,u3,cx,id,snapshot',
                                'coupling_map': None, 'layout': None},
                            'compiled_circuit': {
                                'header': None,
                                'operations': [
                                    {'name': 'u1', 'params': [0.4], 'qubits': [1]},
                                    {'name': 'u2', 'params': [0.4, 0.2],
                                     'qubits': [1]}]
                            },
                            'compiled_circuit_qasm': 'compiled_qasm'
                        }
                    ]}
        self.assertEqual(qobj.as_dict(), expected)
