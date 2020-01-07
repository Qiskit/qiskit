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

"""Contains a (slow) Python simulator.

It simulates a qasm quantum circuit (an experiment) that has been compiled
to run on the simulator. It is exponential in the number of qubits.

The simulator is run using

.. code-block:: python

    QasmSimulatorPy().run(qobj)

Where the input is a Qobj object and the output is a BasicAerJob object, which can
later be queried for the Result object. The result will contain a 'memory' data
field, which is a result of measurements for each shot.
"""

import uuid
import time
import logging

from math import log2
from collections import Counter
import numpy as np

from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.providers import BaseBackend
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from .exceptions import BasicAerError
from .basicaertools import single_gate_matrix
from .basicaertools import cx_gate_matrix
from .basicaertools import einsum_vecmul_index

logger = logging.getLogger(__name__)


class QasmSimulatorPy(BaseBackend):
    """Python implementation of a qasm simulator."""

    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': '2.0.0',
        'n_qubits': min(24, MAX_QUBITS_MEMORY),
        'url': 'https://github.com/Qiskit/qiskit-terra',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'coupling_map': None,
        'description': 'A python simulator for qasm experiments',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'id', 'unitary'],
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
            },
            {
                'name': 'unitary',
                'parameters': ['matrix'],
                'qasm_def': 'unitary(matrix) q1, q2,...'
            }
        ]
    }

    DEFAULT_OPTIONS = {
        "initial_statevector": None,
        "chop_threshold": 1e-15
    }

    # Class level variable to return the final state at the end of simulation
    # This should be set to True for the statevector simulator
    SHOW_FINAL_STATE = False

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=(
            configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)),
                         provider=provider)

        # Define attributes in __init__.
        self._local_random = np.random.RandomState()
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = self.DEFAULT_OPTIONS["initial_statevector"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        self._qobj_config = None
        # TEMP
        self._sample_measure = False

    def _add_unitary(self, gate, qubits):
        """Apply an N-qubit unitary matrix.

        Args:
            gate (matrix_like): an N-qubit unitary matrix
            qubits (list): the list of N-qubits.
        """
        # Get the number of qubits
        num_qubits = len(qubits)
        # Compute einsum index string for 1-qubit matrix multiplication
        indexes = einsum_vecmul_index(qubits, self._number_of_qubits)
        # Convert to complex rank-2N tensor
        gate_tensor = np.reshape(np.array(gate, dtype=complex),
                                 num_qubits * [2, 2])
        # Apply matrix multiplication
        self._statevector = np.einsum(indexes, gate_tensor, self._statevector,
                                      dtype=complex, casting='no')

    def _get_measure_outcome(self, qubit):
        """Simulate the outcome of measurement of a qubit.

        Args:
            qubit (int): the qubit to measure

        Return:
            tuple: pair (outcome, probability) where outcome is '0' or '1' and
            probability is the probability of the returned outcome.
        """
        # Axis for numpy.sum to compute probabilities
        axis = list(range(self._number_of_qubits))
        axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis))
        # Compute einsum index string for 1-qubit matrix multiplication
        random_number = self._local_random.rand()
        if random_number < probabilities[0]:
            return '0', probabilities[0]
        # Else outcome was '1'
        return '1', probabilities[1]

    def _add_sample_measure(self, measure_params, num_samples):
        """Generate memory samples from current statevector.

        Args:
            measure_params (list): List of (qubit, cmembit) values for
                                   measure instructions to sample.
            num_samples (int): The number of memory samples to generate.

        Returns:
            list: A list of memory values in hex format.
        """
        # Get unique qubits that are actually measured and sort in
        # ascending order
        measured_qubits = sorted(list({qubit for qubit, cmembit in measure_params}))
        num_measured = len(measured_qubits)
        # We use the axis kwarg for numpy.sum to compute probabilities
        # this sums over all non-measured qubits to return a vector
        # of measure probabilities for the measured qubits
        axis = list(range(self._number_of_qubits))
        for qubit in reversed(measured_qubits):
            # Remove from largest qubit to smallest so list position is correct
            # with respect to position from end of the list
            axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.reshape(np.sum(np.abs(self._statevector) ** 2,
                                          axis=tuple(axis)),
                                   2 ** num_measured)
        # Generate samples on measured qubits as ints with qubit
        # position in the bit-string for each int given by the qubit
        # position in the sorted measured_qubits list
        samples = self._local_random.choice(range(2 ** num_measured),
                                            num_samples, p=probabilities)
        # Convert the ints to bitstrings
        memory = []
        for sample in samples:
            classical_memory = self._classical_memory
            for qubit, cmembit in measure_params:
                pos = measured_qubits.index(qubit)
                qubit_outcome = int((sample & (1 << pos)) >> pos)
                membit = 1 << cmembit
                classical_memory = (classical_memory & (~membit)) | (qubit_outcome << cmembit)
            value = bin(classical_memory)[2:]
            memory.append(hex(int(value, 2)))
        return memory

    def _add_qasm_measure(self, qubit, cmembit, cregbit=None):
        """Apply a measure instruction to a qubit.

        Args:
            qubit (int): qubit is the qubit measured.
            cmembit (int): is the classical memory bit to store outcome in.
            cregbit (int, optional): is the classical register bit to store outcome in.
        """
        # get measure outcome
        outcome, probability = self._get_measure_outcome(qubit)
        # update classical state
        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (~membit)) | (int(outcome) << cmembit)

        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = \
                (self._classical_register & (~regbit)) | (int(outcome) << cregbit)

        # update quantum state
        if outcome == '0':
            update_diag = [[1 / np.sqrt(probability), 0], [0, 0]]
        else:
            update_diag = [[0, 0], [0, 1 / np.sqrt(probability)]]
        # update classical state
        self._add_unitary(update_diag, [qubit])

    def _add_qasm_reset(self, qubit):
        """Apply a reset instruction to a qubit.

        Args:
            qubit (int): the qubit being rest

        This is done by doing a simulating a measurement
        outcome and projecting onto the outcome state while
        renormalizing.
        """
        # get measure outcome
        outcome, probability = self._get_measure_outcome(qubit)
        # update quantum state
        if outcome == '0':
            update = [[1 / np.sqrt(probability), 0], [0, 0]]
            self._add_unitary(update, [qubit])
        else:
            update = [[0, 1 / np.sqrt(probability)], [0, 0]]
            self._add_unitary(update, [qubit])

    def _validate_initial_statevector(self):
        """Validate an initial statevector"""
        # If initial statevector isn't set we don't need to validate
        if self._initial_statevector is None:
            return
        # Check statevector is correct length for number of qubits
        length = len(self._initial_statevector)
        required_dim = 2 ** self._number_of_qubits
        if length != required_dim:
            raise BasicAerError('initial statevector is incorrect length: ' +
                                '{} != {}'.format(length, required_dim))

    def _set_options(self, qobj_config=None, backend_options=None):
        """Set the backend options for all experiments in a qobj"""
        # Reset default options
        self._initial_statevector = self.DEFAULT_OPTIONS["initial_statevector"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        if backend_options is None:
            backend_options = {}

        # Check for custom initial statevector in backend_options first,
        # then config second
        if 'initial_statevector' in backend_options:
            self._initial_statevector = np.array(backend_options['initial_statevector'],
                                                 dtype=complex)
        elif hasattr(qobj_config, 'initial_statevector'):
            self._initial_statevector = np.array(qobj_config.initial_statevector,
                                                 dtype=complex)
        if self._initial_statevector is not None:
            # Check the initial statevector is normalized
            norm = np.linalg.norm(self._initial_statevector)
            if round(norm, 12) != 1:
                raise BasicAerError('initial statevector is not normalized: ' +
                                    'norm {} != 1'.format(norm))
        # Check for custom chop threshold
        # Replace with custom options
        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

    def _initialize_statevector(self):
        """Set the initial statevector for simulation"""
        if self._initial_statevector is None:
            # Set to default state of all qubits in |0>
            self._statevector = np.zeros(2 ** self._number_of_qubits,
                                         dtype=complex)
            self._statevector[0] = 1
        else:
            self._statevector = self._initial_statevector.copy()
        # Reshape to rank-N tensor
        self._statevector = np.reshape(self._statevector,
                                       self._number_of_qubits * [2])

    def _get_statevector(self):
        """Return the current statevector"""
        vec = np.reshape(self._statevector, 2 ** self._number_of_qubits)
        vec[abs(vec) < self._chop_threshold] = 0.0
        return vec

    def _validate_measure_sampling(self, experiment):
        """Determine if measure sampling is allowed for an experiment

        Args:
            experiment (QobjExperiment): a qobj experiment.
        """
        # If shots=1 we should disable measure sampling.
        # This is also required for statevector simulator to return the
        # correct final statevector without silently dropping final measurements.
        if self._shots <= 1:
            self._sample_measure = False
            return

        # Check for config flag
        if hasattr(experiment.config, 'allows_measure_sampling'):
            self._sample_measure = experiment.config.allows_measure_sampling
        # If flag isn't found do a simple test to see if a circuit contains
        # no reset instructions, and no gates instructions after
        # the first measure.
        else:
            measure_flag = False
            for instruction in experiment.instructions:
                # If circuit contains reset operations we cannot sample
                if instruction.name == "reset":
                    self._sample_measure = False
                    return
                # If circuit contains a measure option then we can
                # sample only if all following operations are measures
                if measure_flag:
                    # If we find a non-measure instruction
                    # we cannot do measure sampling
                    if instruction.name not in ["measure", "barrier", "id", "u0"]:
                        self._sample_measure = False
                        return
                elif instruction.name == "measure":
                    measure_flag = True
            # If we made it to the end of the circuit without returning
            # measure sampling is allowed
            self._sample_measure = True

    def run(self, qobj, backend_options=None):
        """Run qobj asynchronously.

        Args:
            qobj (Qobj): payload of the experiment
            backend_options (dict): backend options

        Returns:
            BasicAerJob: derived from BaseJob

        Additional Information:
            backend_options: Is a dict of options for the backend. It may contain
                * "initial_statevector": vector_like

            The "initial_statevector" option specifies a custom initial
            initial statevector for the simulator to be used instead of the all
            zero state. This size of this vector must be correct for the number
            of qubits in all experiments in the qobj.

            Example::

                backend_options = {
                    "initial_statevector": np.array([1, 0, 0, 1j]) / np.sqrt(2),
                }
        """
        self._set_options(qobj_config=qobj.config,
                          backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = BasicAerJob(self, job_id, self._run_job, qobj)
        job.submit()
        return job

    def _run_job(self, job_id, qobj):
        """Run experiments in qobj

        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description

        Returns:
            Result: Result object
        """
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._memory = getattr(qobj.config, 'memory', False)
        self._qobj_config = qobj.config
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
                  'header': qobj.header.to_dict()}

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
                    "counts": {'0x9: 5, ...},
                    "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                    },
                "status": status string for the simulation
                "success": boolean
                "time_taken": simulation time of this single experiment
                }
        Raises:
            BasicAerError: if an error occurred.
        """
        start = time.time()
        self._number_of_qubits = experiment.config.n_qubits
        self._number_of_cmembits = experiment.config.memory_slots
        self._statevector = 0
        self._classical_memory = 0
        self._classical_register = 0
        self._sample_measure = False
        # Validate the dimension of initial statevector if set
        self._validate_initial_statevector()
        # Get the seed looking in circuit, qobj, and then random.
        if hasattr(experiment.config, 'seed_simulator'):
            seed_simulator = experiment.config.seed_simulator
        elif hasattr(self._qobj_config, 'seed_simulator'):
            seed_simulator = self._qobj_config.seed_simulator
        else:
            # For compatibility on Windows force dyte to be int32
            # and set the maximum value to be (2 ** 31) - 1
            seed_simulator = np.random.randint(2147483647, dtype='int32')

        self._local_random.seed(seed=seed_simulator)
        # Check if measure sampling is supported for current circuit
        self._validate_measure_sampling(experiment)

        # List of final counts for all shots
        memory = []
        # Check if we can sample measurements, if so we only perform 1 shot
        # and sample all outcomes from the final state vector
        if self._sample_measure:
            shots = 1
            # Store (qubit, cmembit) pairs for all measure ops in circuit to
            # be sampled
            measure_sample_ops = []
        else:
            shots = self._shots
        for _ in range(shots):
            self._initialize_statevector()
            # Initialize classical memory to all 0
            self._classical_memory = 0
            self._classical_register = 0
            for operation in experiment.instructions:
                conditional = getattr(operation, 'conditional', None)
                if isinstance(conditional, int):
                    conditional_bit_set = (self._classical_register >> conditional) & 1
                    if not conditional_bit_set:
                        continue
                elif conditional is not None:
                    mask = int(operation.conditional.mask, 16)
                    if mask > 0:
                        value = self._classical_memory & mask
                        while (mask & 0x1) == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation.conditional.val, 16):
                            continue

                # Check if single  gate
                if operation.name == 'unitary':
                    qubits = operation.qubits
                    gate = operation.params[0]
                    self._add_unitary(gate, qubits)
                elif operation.name in ('U', 'u1', 'u2', 'u3'):
                    params = getattr(operation, 'params', None)
                    qubit = operation.qubits[0]
                    gate = single_gate_matrix(operation.name, params)
                    self._add_unitary(gate, [qubit])
                # Check if CX gate
                elif operation.name in ('id', 'u0'):
                    pass
                elif operation.name in ('CX', 'cx'):
                    qubit0 = operation.qubits[0]
                    qubit1 = operation.qubits[1]
                    gate = cx_gate_matrix()
                    self._add_unitary(gate, [qubit0, qubit1])
                # Check if reset
                elif operation.name == 'reset':
                    qubit = operation.qubits[0]
                    self._add_qasm_reset(qubit)
                # Check if barrier
                elif operation.name == 'barrier':
                    pass
                # Check if measure
                elif operation.name == 'measure':
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = operation.register[0] if hasattr(operation, 'register') else None

                    if self._sample_measure:
                        # If sampling measurements record the qubit and cmembit
                        # for this measurement for later sampling
                        measure_sample_ops.append((qubit, cmembit))
                    else:
                        # If not sampling perform measurement as normal
                        self._add_qasm_measure(qubit, cmembit, cregbit)
                elif operation.name == 'bfunc':
                    mask = int(operation.mask, 16)
                    relation = operation.relation
                    val = int(operation.val, 16)

                    cregbit = operation.register
                    cmembit = operation.memory if hasattr(operation, 'memory') else None

                    compared = (self._classical_register & mask) - val

                    if relation == '==':
                        outcome = (compared == 0)
                    elif relation == '!=':
                        outcome = (compared != 0)
                    elif relation == '<':
                        outcome = (compared < 0)
                    elif relation == '<=':
                        outcome = (compared <= 0)
                    elif relation == '>':
                        outcome = (compared > 0)
                    elif relation == '>=':
                        outcome = (compared >= 0)
                    else:
                        raise BasicAerError('Invalid boolean function relation.')

                    # Store outcome in register and optionally memory slot
                    regbit = 1 << cregbit
                    self._classical_register = \
                        (self._classical_register & (~regbit)) | (int(outcome) << cregbit)
                    if cmembit is not None:
                        membit = 1 << cmembit
                        self._classical_memory = \
                            (self._classical_memory & (~membit)) | (int(outcome) << cmembit)
                else:
                    backend = self.name()
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise BasicAerError(err_msg.format(backend, operation.name))

            # Add final creg data to memory list
            if self._number_of_cmembits > 0:
                if self._sample_measure:
                    # If sampling we generate all shot samples from the final statevector
                    memory = self._add_sample_measure(measure_sample_ops, self._shots)
                else:
                    # Turn classical_memory (int) into bit string and pad zero for unused cmembits
                    outcome = bin(self._classical_memory)[2:]
                    memory.append(hex(int(outcome, 2)))

        # Add data
        data = {'counts': dict(Counter(memory))}
        # Optionally add memory list
        if self._memory:
            data['memory'] = memory
        # Optionally add final statevector
        if self.SHOW_FINAL_STATE:
            data['statevector'] = self._get_statevector()
            # Remove empty counts and memory for statevector simulator
            if not data['counts']:
                data.pop('counts')
            if 'memory' in data and not data['memory']:
                data.pop('memory')
        end = time.time()
        return {'name': experiment.header.name,
                'seed_simulator': seed_simulator,
                'shots': self._shots,
                'data': data,
                'status': 'DONE',
                'success': True,
                'time_taken': (end - start),
                'header': experiment.header.to_dict()}

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas."""
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise BasicAerError('Number of qubits {} '.format(n_qubits) +
                                'is greater than maximum ({}) '.format(max_qubits) +
                                'for "{}".'.format(self.name()))
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning('No classical registers in circuit "%s", '
                               'counts will be empty.', name)
            elif 'measure' not in [op.name for op in experiment.instructions]:
                logger.warning('No measurements in circuit "%s", '
                               'classical register will remain all zeros.', name)
