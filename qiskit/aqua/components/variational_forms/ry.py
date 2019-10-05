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

"""Layers of Y rotations followed by entangling gates."""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.variational_forms import VariationalForm


class RY(VariationalForm):
    """Layers of Y rotations followed by entangling gates."""

    CONFIGURATION = {
        'name': 'RY',
        'description': 'RY Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'ry_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 3,
                    'minimum': 0
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'enum': ['full', 'linear', 'sca']
                },
                'entangler_map': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'entanglement_gate': {
                    'type': 'string',
                    'default': 'cz',
                    'enum': ['cz', 'cx', 'crx']
                },
                'skip_unentangled_qubits': {
                    'type': 'boolean',
                    'default': False
                },
                'skip_final_ry': {
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
                 entanglement='full', initial_state=None,
                 entanglement_gate='cz', skip_unentangled_qubits=False,
                 skip_final_ry=False):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement (str): 'full', 'linear' or 'sca'
            initial_state (InitialState): an initial state object
            entanglement_gate (str): cz or cx
            skip_unentangled_qubits (bool): skip the qubits not in the entangler_map
            skip_final_ry (bool): skip the final layer of Y rotations
        """
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._depth = depth
        self._entanglement = entanglement

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
        self._entanglement_gate = entanglement_gate
        self._skip_unentangled_qubits = skip_unentangled_qubits
        self._skip_final_ry = skip_final_ry

        # for the first layer
        self._num_parameters = len(self._entangled_qubits) if self._skip_unentangled_qubits \
            else self._num_qubits

        # for repeated block (minus one ry layer if we skip the last)
        self._num_parameters += len(self._entangled_qubits) * (depth - 1) if skip_final_ry \
            else len(self._entangled_qubits) * depth

        # CRx gates have an additional parameter per entanglement
        if entanglement_gate == 'crx':
            self._num_parameters += len(self._entangler_map) * depth

        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._support_parameterized_circuit = True

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (Union(numpy.ndarray, list[Parameter], ParameterVector)): circuit parameters.
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
                circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                param_idx += 1

        for block in range(self._depth):
            circuit.barrier(q)
            if self._entanglement == 'sca':
                self._entangler_map = VariationalForm.get_entangler_map(
                    self._entanglement,
                    self._num_qubits,
                    offset=block)

            for src, targ in self._entangler_map:
                if self._entanglement_gate == 'cz':
                    circuit.u2(0.0, np.pi, q[targ])  # h
                    circuit.cx(q[src], q[targ])
                    circuit.u2(0.0, np.pi, q[targ])  # h

                elif self._entanglement_gate == 'crx':
                    circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2,
                                q[src], q[targ])  # crx
                    param_idx += 1

                else:
                    circuit.cx(q[src], q[targ])

            # Skip the final RY layer if it is specified and we reached the
            # last block
            if not self._skip_final_ry or block != self._depth - 1:
                circuit.barrier(q)
                for qubit in self._entangled_qubits:
                    circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                    param_idx += 1
        circuit.barrier(q)

        return circuit
