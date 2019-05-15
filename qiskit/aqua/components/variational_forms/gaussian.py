# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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
"""
This module contains the definition of a base class for
variational forms. Several types of commonly used ansatz.
"""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.potentials.harmonic import Harmonic
from qiskit.aqua.components.qfts.standard import Standard as StandardQFT
from qiskit.aqua.components.qfts.swap import Swap
from qiskit.aqua.components.iqfts.standard import Standard as StandardIQFT

class Gaussian(VariationalForm):

    CONFIGURATION = {
        'name': 'Gaussian',
        'description': 'Gaussian Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ry_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                },
                'entangler_map': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'entanglement_gate': {
                    'type': 'string',
                    'default': 'cz',
                    'oneOf': [
                        {'enum': ['cz', 'cx']}
                    ]
                }
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, num_qubits, depth = 1, initial_state = None, entangler_map = None, entanglement='full',
                 entanglement_gate='cz'):
        super().__init__()
        self._num_qubits = num_qubits
        self._num_parameters = num_qubits * (depth + 1)
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._depth = depth
        self._initial_state = initial_state
        self._entanglement_gate = entanglement_gate
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)

    def construct_circuit(self, parameters, q=None, c=None):

        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(2*self._num_qubits, name='q')
        if c is None:
            c = ClassicalRegister(2*self._num_qubits, name='c')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q,c)
        else:
            circuit = QuantumCircuit(q,c)

        param_idx = 0
        for qubit in range(self._num_qubits):
            circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])
            param_idx += 1

        for block in range(self._depth):
            circuit.barrier(q)
            for src, targ in self._entangler_map:
                if self._entanglement_gate == 'cz':
                    circuit.u2(0.0, np.pi, q[targ])  # h
                    circuit.cx(q[src], q[targ])
                    circuit.u2(0.0, np.pi, q[targ])  # h
                else:
                    circuit.cx(q[src], q[targ])
            for qubit in range(self._num_qubits):
                circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])
                # circuit.ry(parameters[param_idx, q[qubit])
                param_idx += 1

        circuit.barrier(q)

        for qubit in range(self._num_qubits):

            circuit.cx(q[qubit],q[self._num_qubits+qubit])

        circuit.barrier(q)

        for j in range(self._num_qubits - 1, -1, -1):
            for k in range(self._num_qubits - 1, j, -1):
                lam = 1.0 * np.pi / float(2 ** (k - j))
                circuit.u1(lam / 2, q[j])
                circuit.cx(q[j], q[k])
                circuit.u1(-lam / 2, q[k])
                circuit.cx(q[j], q[k])
                circuit.u1(lam / 2, q[k])
            circuit.u2(0, np.pi, q[j])

        circuit.barrier(q)

        for i in range(int(self._num_qubits / 2)):
            circuit.swap(q[i], q[self._num_qubits - 1 - i])

        circuit.measure(q, c)

        return circuit

