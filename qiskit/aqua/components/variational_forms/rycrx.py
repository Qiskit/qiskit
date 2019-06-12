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

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.variational_forms import VariationalForm


class RYCRX(VariationalForm):
    """
    Layers of Y rotations followed by entangling layers that consist of
    controlled X gates.

    The default behaviour is a circular entanglement, shifted by one gate
    every block, as described in http://arxiv.org/abs/1905.10876 (circuit 14).
    According to this paper, this circuit has a is able to represent the
    underlying Hilbert space well while keeping the number of parameters low.
    """

    CONFIGURATION = {
        'name': 'RYCRX',
        'description': 'RYCRX Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'rycrx_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 3,
                    'minimum': 1
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'circular',
                    'oneOf': [
                        {'enum': ['circular', 'full', 'linear']}
                    ]
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
                 entanglement='circular', initial_state=None,
                 skip_unentangled_qubits=False):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            depth (int): number of rotation layers
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement (str): 'circular', 'full' or 'linear'
            initial_state (InitialState): an initial state object
            skip_unentangled_qubits (bool): skip the qubits not in the entangler_map

        Note that the number of parameters equals the number of gates, since
        the entanglements are controlled X-rotations. Hence a full entanglement
        will have more parameters than linear or circular entanglement.
        """
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        if depth < 1:
            raise AttributeError("depth must at least be 1!")

        self._depth = depth
        self._initial_state = initial_state

        # Get/Set entangler map
        # The circular entanglement is special and cannot be represented with
        # a simple entangler_map since it changes every block, hence the
        # definition of the member '_circular'
        self._circular = False
        if entangler_map is None and entanglement != 'circular':
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        elif entangler_map is not None:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        else:
            self._circular = True

        # determine the entangled qubits
        if self._circular:
            self._num_parameters = 2 * num_qubits * depth

            # all qubits are entangled
            self._entangled_qubits = list(range(num_qubits))
            self._skip_unentangled_qubits = False

        else:
            all_qubits = []
            for src, targ in self._entangler_map:
                all_qubits.extend([src, targ])
            self._entangled_qubits = sorted(list(set(all_qubits)))
            self._skip_unentangled_qubits = skip_unentangled_qubits
            if skip_unentangled_qubits:
                self._num_parameters = (len(self._entangler_map) + len(self._entangled_qubits)) * depth
            else:
                self._num_parameters = (len(self._entangler_map) + num_qubits) * depth

        self._initial_state = initial_state
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters.
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
        for block in range(self._depth):
            if block > 0:
                circuit.barrier(q)

            # After circle_idx CRx in between neighbours we apply a CRx
            # with the first and last qubit ('closing' the circle)
            circle_idx = block % self._num_qubits

            # layer of y rotations
            for qubit in range(self._num_qubits):
                if not self._skip_unentangled_qubits or qubit in self._entangled_qubits:
                    circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                    param_idx += 1

            # layer of entanglements
            if self._circular:
                if block % 2 == 0:  # even block numbers
                    for qubit in reversed(range(circle_idx)):
                        circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[qubit], q[qubit + 1])  # crx
                        param_idx += 1

                    circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[-1], q[0])
                    param_idx += 1

                    for qubit in reversed(range(circle_idx + 1, self._num_qubits)):
                        circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[qubit - 1], q[qubit])
                        param_idx += 1

                else:  # odd block numbers
                    for qubit in range(self._num_qubits - circle_idx - 1, self._num_qubits - 1):
                        circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[qubit + 1], q[qubit])
                        param_idx += 1

                    circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[0], q[-1])
                    param_idx += 1

                    for qubit in range(self._num_qubits - circle_idx - 1):
                        circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[qubit + 1], q[qubit])
                        param_idx += 1

            else:  # use entangler_map (since not circular)
                for src, targ in self._entangler_map:
                    circuit.cu3(parameters[param_idx], -np.pi / 2, np.pi / 2, q[src], q[targ])
                    param_idx += 1

        circuit.barrier(q)

        return circuit
