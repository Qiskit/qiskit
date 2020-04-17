# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=arguments-differ

"""Contains a Python simulator that returns the unitary of the circuit.

It simulates a unitary of a quantum circuit that has been compiled to run on
the simulator. It is exponential in the number of qubits.

.. code-block:: python

    UnitarySimulator().run(circuit)

Where the input is a QuantumCircuit object and the output a unitary matrix,
which is a 2**n x 2**n complex numpy array representing the circuit's unitary
matrix.
"""

import collections
import logging
from math import log2, sqrt
import time
import uuid

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.util import local_hardware_info
from qiskit.providers.v2.backend import Backend
from qiskit.providers.v2.configuration import Configuration
from qiskit.providers.v2.target import Target
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from qiskit.result import Result
from .exceptions import BasicAerError
from .basicaertools import single_gate_matrix
from .basicaertools import cx_gate_matrix
from .basicaertools import einsum_matmul_index
from .basicaertools import assemble_circuit

logger = logging.getLogger(__name__)

MAX_QUBITS_MEMORY = int(
    log2(sqrt(local_hardware_info()['memory'] * (1024 ** 3) / 16)))

class QasmUnitarySimulatorTarget(Target):
    @property
    def num_qubits(self):
        return min(24, MAX_QUBITS_MEMORY)

    @property
    def conditional(self):
        return False

    @property
    def coupling_map(self):
        return None

    @property
    def supported_instructions(self):
        return None

    @property
    def basis_gates(self):
        return ['u1', 'u2', 'u3', 'cx', 'id', 'unitary']

    @property
    def gates(self):
        return [
            {
                'name': 'u1',
                'parameters': ['lambda'],
                'qasm_def': 'gate u1(lambda) q { U(0,0,lambda) q; }'
            },
            {
                'name': 'u2',
                'parameters': ['phi', 'lambda'],
                'qasm_def': 'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'
            },
            {
                'name': 'u3',
                'parameters': ['theta', 'phi', 'lambda'],
                'qasm_def': 'gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'
            },
            {
                'name': 'cx',
                'parameters': ['c', 't'],
                'qasm_def': 'gate cx c,t { CX c,t; }'
            },
            {
                'name': 'id',
                'parameters': ['a'],
                'qasm_def': 'gate id a { U(0,0,0) a; }'
            },
            {
                'name': 'unitary',
                'parameters': ['matrix'],
                'qasm_def': 'unitary(matrix) q1, q2,...'
            }
        ]





# TODO add ["status"] = 'DONE', 'ERROR' especially for empty circuit error
# does not show up

class UnitarySimulatorPy(Backend):
    """Python implementation of a unitary simulator."""
    @property
    def local(self):
        return True

    @property
    def open_pulse(self):
        return False

    @property
    def memory(self):
        return False

    def __init__(self, name='unitary_simulator', **fields):
        super().__init__(name, **fields)
        # Define attributes inside __init__.
        self._target = QasmUnitarySimulatorTarget()
        self._unitary = None
        self._number_of_qubits = 0
        self._initial_unitary = self.configuration.get('initial_unitary')
        self._chop_threshold = self.configuration.get('chop_threshold')

    @classmethod
    def _default_config(cls):
        return Configuration(shots=1,
                             initial_unitary=None, chop_threshold=1e-15)

    def _add_unitary(self, gate, qubits):
        """Apply an N-qubit unitary matrix.

        Args:
            gate (matrix_like): an N-qubit unitary matrix
            qubits (list): the list of N-qubits.
        """
        # Get the number of qubits
        num_qubits = len(qubits)
        # Compute einsum index string for 1-qubit matrix multiplication
        indexes = einsum_matmul_index(qubits, self._number_of_qubits)
        # Convert to complex rank-2N tensor
        gate_tensor = np.reshape(np.array(gate, dtype=complex),
                                 num_qubits * [2, 2])
        # Apply matrix multiplication
        self._unitary = np.einsum(indexes, gate_tensor, self._unitary,
                                  dtype=complex, casting='no')

    def _validate_initial_unitary(self):
        """Validate an initial unitary matrix"""
        # If initial unitary isn't set we don't need to validate
        if self._initial_unitary is None:
            return
        # Check unitary is correct length for number of qubits
        shape = np.shape(self._initial_unitary)
        required_shape = (2 ** self._number_of_qubits,
                          2 ** self._number_of_qubits)
        if shape != required_shape:
            raise BasicAerError('initial unitary is incorrect shape: ' +
                                '{} != 2 ** {}'.format(shape, required_shape))

    def _set_options(self):
        """Set the backend options for all experiments in a qobj"""
        # Reset default options
        self._initial_unitary = self.configuration.get("initial_unitary")
        self._chop_threshold = self.configuration.get("chop_threshold")
        if self._initial_unitary is not None:
            # Check the initial unitary is actually unitary
            shape = np.shape(self._initial_unitary)
            if len(shape) != 2 or shape[0] != shape[1]:
                raise BasicAerError("initial unitary is not a square matrix")
            iden = np.eye(len(self._initial_unitary))
            u_dagger_u = np.dot(self._initial_unitary.T.conj(),
                                self._initial_unitary)
            norm = np.linalg.norm(u_dagger_u - iden)
            if round(norm, 10) != 0:
                raise BasicAerError("initial unitary is not unitary")

    def _initialize_unitary(self):
        """Set the initial unitary for simulation"""
        self._validate_initial_unitary()
        if self._initial_unitary is None:
            # Set to identity matrix
            self._unitary = np.eye(2 ** self._number_of_qubits,
                                   dtype=complex)
        else:
            self._unitary = self._initial_unitary.copy()
        # Reshape to rank-N tensor
        self._unitary = np.reshape(self._unitary,
                                   self._number_of_qubits * [2, 2])

    def _get_unitary(self):
        """Return the current unitary"""
        unitary = np.reshape(self._unitary, 2 * [2 ** self._number_of_qubits])
        unitary[abs(unitary) < self._chop_threshold] = 0.0
        return unitary

    def run(self, circuits):
        """Run qobj asynchronously.

        Args:
            qobj (Qobj): payload of the experiment
            backend_options (dict): backend options

        Returns:
            list: a list of unitary matrices
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        self._set_options()
        job_id = str(uuid.uuid4())
        self._validate(circuits)
        result_dict = collections.OrderedDict()
        start = time.time()
        for experiment in circuits:
            result_dict[experiment.name] = self.run_experiment(experiment)
        end = time.time()
        time_taken = end - start
        return BasicAerJob(job_id, self, result_dict, time_taken=time_taken)

    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.

        Args:
            experiment (QobjExperiment): experiment from qobj experiments list

        Returns:
            dict: A result dictionary which looks something like::

                {
                "name": name of this experiment (obtained from qobj.experiment header)
                "seed": random seed used for simulation
                "shots": number of shots used in the simulation
                "data":
                    {
                    "unitary": [[[0.0, 0.0], [1.0, 0.0]],
                                [[1.0, 0.0], [0.0, 0.0]]]
                    },
                "status": status string for the simulation
                "success": boolean
                "time taken": simulation time of this single experiment
                }

        Raises:
            BasicAerError: if the number of qubits in the circuit is greater than 24.
            Note that the practical qubit limit is much lower than 24.
        """
        start = time.time()
        self._number_of_qubits = experiment.num_qubits

        # Validate the dimension of initial unitary if set
        self._validate_initial_unitary()
        self._initialize_unitary()

        for operation in assemble_circuit(experiment):
            if operation.name == 'unitary':
                qubits = operation.qubits
                gate = operation.params[0]
                self._add_unitary(gate, qubits)
            # Check if single  gate
            elif operation.name in ('U', 'u1', 'u2', 'u3'):
                params = getattr(operation, 'params', None)
                qubit = operation.qubits[0]
                gate = single_gate_matrix(operation.name, params)
                self._add_unitary(gate, [qubit])
            elif operation.name in ('id', 'u0'):
                pass
            # Check if CX gate
            elif operation.name in ('CX', 'cx'):
                qubit0 = operation.qubits[0]
                qubit1 = operation.qubits[1]
                gate = cx_gate_matrix()
                self._add_unitary(gate, [qubit0, qubit1])
            # Check if barrier
            elif operation.name == 'barrier':
                pass
            else:
                backend = self.name()
                err_msg = '{0} encountered unrecognized operation "{1}"'
                raise BasicAerError(err_msg.format(backend, operation.name))
        # Add final state to data
        data = self._get_unitary()
        end = time.time()
        return data

    def _validate(self, circuits):
        """Semantic validations of the circuit.
        Some of these may later move to backend schemas.
        1. No shots
        2. No measurements in the middle
        """
        shots = self.configuration.get('shots')
        if shots and shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.',
                        self.name)
            self.set_configuration(shots=1)
        for experiment in circuits:
            if experiment.num_qubits > self._target.num_qubits:
                raise BasicAerError('Number of qubits {} '.format(experiment.num_qubits) +
                                    'is greater than maximum ({}) '.format(self.num_qubits) +
                                    'for "{}".'.format(self.name))
            name = experiment.name
            for operation in experiment.count_ops():
                if operation in ['measure', 'reset']:
                    raise BasicAerError('Unsupported "%s" instruction "%s" ' +
                                        'in circuit "%s" ', self.name,
                                        operation[0].name, name)
