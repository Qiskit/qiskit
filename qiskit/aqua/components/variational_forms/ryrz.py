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

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm


class RYRZ(VariationalForm):
    """Layers of Y+Z rotations followed by entangling gates."""

    CONFIGURATION = {
        'name': 'RYRZ',
        'description': 'RYRZ Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ryrz_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 3,
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

    def __init__(self, num_qubits, depth=3, entangler_map=None,
                 entanglement='full', initial_state=None):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement (str): 'full' or 'linear'
            initial_state (InitialState): an initial state object
        """
        self.validate(locals())
        super().__init__()
        self._num_parameters = num_qubits * (depth + 1) * 2
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._initial_state = initial_state

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        param_idx = 0
        for qubit in range(self._num_qubits):
            circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
            circuit.u1(parameters[param_idx + 1], q[qubit])  # rz
            param_idx += 2

        for block in range(self._depth):
            circuit.barrier(q)
            for src, targ in self._entangler_map:
                # cz
                circuit.u2(0.0, np.pi, q[targ])  # h
                circuit.cx(q[src], q[targ])
                circuit.u2(0.0, np.pi, q[targ])  # h
            for qubit in range(self._num_qubits):
                circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                circuit.u1(parameters[param_idx + 1], q[qubit])  # rz
                param_idx += 2
        circuit.barrier(q)

        return circuit
