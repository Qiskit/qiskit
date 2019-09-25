# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Layers of Swap+Z rotations followed by entangling gates."""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.variational_forms import VariationalForm


class SwapRZ(VariationalForm):
    """Layers of Swap+Z rotations followed by entangling gates."""

    CONFIGURATION = {
        'name': 'SWAPRZ',
        'description': 'SWAPRZ Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'swaprz_schema',
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
                    'enum': ['full', 'linear']
                },
                'entangler_map': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'skip_unentangled_qubits': {
                    'type': 'boolean',
                    'default': False
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
                 entanglement='full', initial_state=None, skip_unentangled_qubits=False):
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
            skip_unentangled_qubits (bool): skip the qubits not in the entangler_map
        """
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        # determine the entangled qubits
        all_qubits = []
        for src, targ in self._entangler_map:
            all_qubits.extend([src, targ])

        self._entangled_qubits = sorted(list(set(all_qubits)))
        self._initial_state = initial_state
        self._skip_unentangled_qubits = skip_unentangled_qubits

        # for the first layer
        self._num_parameters = len(self._entangled_qubits) if self._skip_unentangled_qubits \
            else self._num_qubits
        # for repeated block
        self._num_parameters += (len(self._entangled_qubits) + len(self._entangler_map)) * depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._support_parameterized_circuit = True

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (Union(numpy.ndarray, list[Parameter], ParameterVector)): circuit parameters
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
            if not self._skip_unentangled_qubits or qubit in self._entangled_qubits:
                circuit.u1(parameters[param_idx], q[qubit])  # rz
                param_idx += 1

        for _ in range(self._depth):
            circuit.barrier(q)
            for src, targ in self._entangler_map:
                # XX
                circuit.u2(0, np.pi, q[src])
                circuit.u2(0, np.pi, q[targ])
                circuit.cx(q[src], q[targ])
                circuit.u1(parameters[param_idx], q[targ])
                circuit.cx(q[src], q[targ])
                circuit.u2(0, np.pi, q[src])
                circuit.u2(0, np.pi, q[targ])
                # YY
                circuit.u3(np.pi / 2, -np.pi / 2, np.pi / 2, q[src])
                circuit.u3(np.pi / 2, -np.pi / 2, np.pi / 2, q[targ])
                circuit.cx(q[src], q[targ])
                circuit.u1(parameters[param_idx], q[targ])
                circuit.cx(q[src], q[targ])
                circuit.u3(-np.pi / 2, -np.pi / 2, np.pi / 2, q[src])
                circuit.u3(-np.pi / 2, -np.pi / 2, np.pi / 2, q[targ])
                param_idx += 1
            circuit.barrier(q)
            for qubit in self._entangled_qubits:
                circuit.u1(parameters[param_idx], q[qubit])  # rz
                param_idx += 1
        circuit.barrier(q)

        return circuit
