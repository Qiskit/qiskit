# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=arguments-differ

"""Contains a Python simulator that returns the unitary of the circuit.

It simulates a unitary of a quantum circuit that has been compiled to run on
the simulator. It is exponential in the number of qubits.

.. code-block:: python

    UnitarySimulator().run(qobj)

Where the input is a Qobj object and the output is a SimulatorsJob object, which can
later be queried for the Result object. The result will contain a 'unitary'
data field, which is a 2**n x 2**n complex numpy array representing the
circuit's unitary matrix.
"""
import logging
import uuid
import time
from math import log2, sqrt
import numpy as np
from qiskit._util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from qiskit.providers import BaseBackend
from qiskit.providers.builtinsimulators.simulatorsjob import SimulatorsJob
from qiskit.result import Result
from ._simulatorerror import SimulatorError
from ._simulatortools import single_gate_matrix
from ._simulatortools import cx_gate_matrix
from ._simulatortools import einsum_matmul_index

logger = logging.getLogger(__name__)


# TODO add ["status"] = 'DONE', 'ERROR' especially for empty circuit error
# does not show up


class UnitarySimulatorPy(BaseBackend):
    """Python implementation of a unitary simulator."""

    MAX_QUBITS_MEMORY = int(log2(sqrt(local_hardware_info()['memory'] * (1024 ** 3) / 16)))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'unitary_simulator',
        'backend_version': '1.0.0',
        'n_qubits': min(24, MAX_QUBITS_MEMORY),
        'url': 'https://github.com/Qiskit/qiskit-terra',
        'simulator': True,
        'local': True,
        'conditional': False,
        'open_pulse': False,
        'memory': False,
        'max_shots': 65536,
        'description': 'A python simulator for unitary matrix corresponding to a circuit',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'id'],
        'gates': [
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
            }
        ]
    }

    DEFAULT_OPTIONS = {
        "initial_unitary": None,
        "chop_threshold": 1e-15
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=(configuration or
                                        BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)),
                         provider=provider)

        # Define attributes inside __init__.
        self._unitary = None
        self._number_of_qubits = 0
        self._initial_unitary = None
        self._chop_threshold = 1e-15

    def _add_unitary_single(self, gate, qubit):
        """Apply an arbitrary 1-qubit unitary matrix.

        Args:
            gate (matrix_like): a single qubit gate matrix
            qubit (int): the qubit to apply gate to
        """
        # Convert to complex rank-2 tensor
        gate_tensor = np.array(gate, dtype=complex)
        # Compute einsum index string for 1-qubit matrix multiplication
        indexes = einsum_matmul_index([qubit], self._number_of_qubits)
        # Apply matrix multiplication
        self._unitary = np.einsum(indexes, gate_tensor, self._unitary,
                                  dtype=complex, casting='no')

    def _add_unitary_two(self, gate, qubit0, qubit1):
        """Apply a two-qubit unitary matrix.

        Args:
            gate (matrix_like): a the two-qubit gate matrix
            qubit0 (int): gate qubit-0
            qubit1 (int): gate qubit-1
        """

        # Convert to complex rank-4 tensor
        gate_tensor = np.reshape(np.array(gate, dtype=complex), 4 * [2])

        # Compute einsum index string for 2-qubit matrix multiplication
        indexes = einsum_matmul_index([qubit0, qubit1], self._number_of_qubits)

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
            raise SimulatorError('initial unitary is incorrect shape: ' +
                                 '{} != 2 ** {}'.format(shape, required_shape))

    def _set_options(self, qobj_config=None, backend_options=None):
        """Set the backend options for all experiments in a qobj"""
        # Reset default options
        self._initial_unitary = self.DEFAULT_OPTIONS["initial_unitary"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        if backend_options is None:
            backend_options = {}

        # Check for custom initial statevector in backend_options first,
        # then config second
        if 'initial_unitary' in backend_options:
            self._initial_unitary = np.array(backend_options['initial_unitary'],
                                             dtype=complex)
        elif hasattr(qobj_config, 'initial_unitary'):
            self._initial_unitary = np.array(qobj_config.initial_unitary,
                                             dtype=complex)
        if self._initial_unitary is not None:
            # Check the initial unitary is actually unitary
            shape = np.shape(self._initial_unitary)
            if len(shape) != 2 or shape[0] != shape[1]:
                raise SimulatorError("initial unitary is not a square matrix")
            iden = np.eye(len(self._initial_unitary))
            u_dagger_u = np.dot(self._initial_unitary.T.conj(),
                                self._initial_unitary)
            norm = np.linalg.norm(u_dagger_u - iden)
            if round(norm, 10) != 0:
                raise SimulatorError("initial unitary is not unitary")
            # Check the initial statevector is normalized

        # Check for custom chop threshold
        # Replace with custom options
        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

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
        """Return the current unitary in JSON Result spec format"""
        unitary = np.reshape(self._unitary, 2 * [2 ** self._number_of_qubits])
        # Expand complex numbers
        unitary = np.stack((unitary.real, unitary.imag), axis=-1)
        # Truncate small values
        unitary[abs(unitary) < self._chop_threshold] = 0.0
        return unitary

    def run(self, qobj, backend_options=None):
        """Run qobj asynchronously.

        Args:
            qobj (Qobj): payload of the experiment
            backend_options (dict): backend options

        Returns:
            SimulatorsJob: derived from BaseJob

        Additional Information::

            backend_options: Is a dict of options for the backend. It may contain
                * "initial_unitary": matrix_like
                * "chop_threshold": double

            The "initial_unitary" option specifies a custom initial unitary
            matrix for the simulator to be used instead of the identity
            matrix. This size of this matrix must be correct for the number
            of qubits inall experiments in the qobj.

            The "chop_threshold" option specifies a trunctation value for
            setting small values to zero in the output unitary. The default
            value is 1e-15.

            Example::

                backend_options = {
                    "initial_unitary": np.array([[1, 0, 0, 0],
                                                 [0, 0, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 1, 0, 0]])
                    "chop_threshold": 1e-15
                }
        """
        self._set_options(qobj_config=qobj.config,
                          backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = SimulatorsJob(self, job_id, self._run_job, qobj)
        job.submit()
        return job

    def _run_job(self, job_id, qobj):
        """Run experiments in qobj.

        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description

        Returns:
            Result: Result object
        """
        self._validate(qobj)
        result_list = []
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {'backend_name': self.name(),
                  'backend_version': self._configuration.backend_version,
                  'qobj_id': qobj.qobj_id,
                  'job_id': job_id,
                  'results': result_list,
                  'status': 'COMPLETED',
                  'success': True,
                  'time_taken': (end - start),
                  'header': qobj.header.as_dict()}

        return Result.from_dict(result)

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
            SimulatorError: if the number of qubits in the circuit is greater than 24.
            Note that the practical qubit limit is much lower than 24.
        """
        start = time.time()
        self._number_of_qubits = experiment.header.n_qubits

        # Validate the dimension of initial unitary if set
        self._validate_initial_unitary()
        self._initialize_unitary()

        for operation in experiment.instructions:
            # Check if single  gate
            if operation.name in ('U', 'u1', 'u2', 'u3'):
                params = getattr(operation, 'params', None)
                qubit = operation.qubits[0]
                gate = single_gate_matrix(operation.name, params)
                self._add_unitary_single(gate, qubit)
            elif operation.name in ('id', 'u0'):
                pass
            # Check if CX gate
            elif operation.name in ('CX', 'cx'):
                qubit0 = operation.qubits[0]
                qubit1 = operation.qubits[1]
                gate = cx_gate_matrix()
                self._add_unitary_two(gate, qubit0, qubit1)
            # Check if barrier
            elif operation.name == 'barrier':
                pass
            else:
                backend = self.name()
                err_msg = '{0} encountered unrecognized operation "{1}"'
                raise SimulatorError(err_msg.format(backend, operation.name))
        # Add final state to data
        data = {'unitary': self._get_unitary()}
        end = time.time()
        return {'name': experiment.header.name,
                'shots': 1,
                'data': data,
                'status': 'DONE',
                'success': True,
                'time_taken': (end - start),
                'header': experiment.header.as_dict()}

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.
        1. No shots
        2. No measurements in the middle
        """
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise SimulatorError('Number of qubits {} '.format(n_qubits) +
                                 'is greater than maximum ({}) '.format(max_qubits) +
                                 'for "{}".'.format(self.name()))
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.',
                        self.name())
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info('"%s" only supports 1 shot. ' +
                            'Setting shots=1 for circuit "%s".',
                            self.name(), name)
                experiment.config.shots = 1
            for operation in experiment.instructions:
                if operation.name in ['measure', 'reset']:
                    raise SimulatorError('Unsupported "%s" instruction "%s" ' +
                                         'in circuit "%s" ', self.name(),
                                         operation.name, name)
