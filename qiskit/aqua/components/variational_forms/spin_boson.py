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
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.variational_forms.swaprz import SwapRZ

class SpinBoson(VariationalForm):

    CONFIGURATION = {
        'name': 'SpinBoson',
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

    def __init__(self, num_qubits, depth = 1, initial_state = None, entangler_map=None, entanglement='full'):
        super().__init__()
        self._num_qubits = num_qubits
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._num_parameters = depth * (2+len(self._entangler_map))
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._depth = depth
        self._initial_state = initial_state


    def construct_circuit(self, parameters, q=None):
        """Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray[float]): circuit parameters.
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            A quantum circuit.
        """

        # if len(parameters) != self._num_parameters:
        #     raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))
        #
        # if q1==None:
        #     q1 = QuantumRegister(1, name='q1')
        # if q2==None:
        #     q2 = QuantumRegister(self._num_qubits, name='q2')
        #
        # circuit_tot = QuantumCircuit(q1)
        # circuit_tot.x(q1[0])
        #
        # circuit = QuantumCircuit(q2)
        # circuit.x(q2[0])
        #
        # var_form = SwapRZ(self._num_qubits, depth=self._depth)
        # circuit += var_form.construct_circuit(parameters[3*self._num_qubits:], q=q2)
        #
        # circuit.extend(circuit_tot)
        #
        # for i in range(self._num_qubits):
        #     p = parameters[3*i:3*(i+1)]
        #     circuit.u1((p[2] - p[1]) / 2, q1[0])
        #     circuit.cx(q2[i], q1[0])
        #     circuit.u3(-p[0] / 2, 0, -(p[1] + p[2]) / 2, q1[0])
        #     circuit.cx(q2[i], q1[0])
        #     circuit.u3(p[0] / 2, p[1], 0, q1[0])

        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits+1, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        #circuit.u3(np.pi/2.,0,np.pi,q[-1])
        circuit.h(q[-1])
        circuit.x(q[0])

        #circuit.barrier(q)

        param_idx = 0
        # for qubit in range(self._num_qubits):
        #     circuit.u1(parameters[param_idx], q[qubit])  # rz
        #     param_idx += 1
        for block in range(self._depth):
            circuit.barrier(q)
            for src, targ in self._entangler_map:
                # XX
                circuit.u2(0, np.pi, q[src])
                circuit.u2(0, np.pi, q[targ])
                circuit.cx(q[src], q[targ])

                circuit.u1(parameters[param_idx], q[targ])
                circuit.cx(q[-1], q[targ])
                circuit.u1(-parameters[param_idx], q[targ])
                circuit.cx(q[-1], q[targ])

                circuit.cx(q[src], q[targ])
                circuit.u2(0, np.pi, q[src])
                circuit.u2(0, np.pi, q[targ])

                # YY
                circuit.u3(np.pi / 2, -np.pi / 2, np.pi / 2, q[src])
                circuit.u3(np.pi / 2, -np.pi / 2, np.pi / 2, q[targ])
                circuit.cx(q[src], q[targ])

                circuit.u1(parameters[param_idx], q[targ])
                circuit.cx(q[-1], q[targ])
                circuit.u1(-parameters[param_idx], q[targ])
                circuit.cx(q[-1], q[targ])

                circuit.cx(q[src], q[targ])
                circuit.u3(-np.pi / 2, -np.pi / 2, np.pi / 2, q[src])
                circuit.u3(-np.pi / 2, -np.pi / 2, np.pi / 2, q[targ])
                param_idx += 1
            # for qubit in range(self._num_qubits):
            #    circuit.u1(parameters[param_idx], q[qubit])  # rz
            #    param_idx += 1
            circuit.barrier(q)

            #circuit.h(q[-1])
            circuit.u1(parameters[param_idx],q[-1])
            param_idx+=1
            circuit.u3(parameters[param_idx], 0, 0, q[-1])
            param_idx+=1
            circuit.h(q[-1])

        return circuit
