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

    QasmSimulatorPy().run(circuits)

Where the input is a Qobj object and the output is a BasicAerJob object, which can
later be queried for the Result object. The result will contain a 'memory' data
field, which is a result of measurements for each shot.
"""

import collections
import logging
from math import log2
import time
import uuid
import warnings

import numpy as np

from qiskit.assembler.disassemble import disassemble
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.util import local_hardware_info
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from qiskit.result.counts import Counts
from qiskit.providers.models import BackendStatus
from qiskit.providers.v2 import Backend
from qiskit.providers.v2 import Configuration
from qiskit.providers.v2.target import Target
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.version import VERSION as __version__
from .exceptions import BasicAerError
from .basicaertools import single_gate_matrix
from .basicaertools import cx_gate_matrix
from .basicaertools import einsum_vecmul_index
from .basicaertools import assemble_circuit

logger = logging.getLogger(__name__)

MAX_QUBITS_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))


class QasmSimulatorTarget(Target):
    @property
    def num_qubits(self):
        return min(24, MAX_QUBITS_MEMORY)

    @property
    def conditional(self):
        return True

    @property
    def basis_gates(self):
        return ['u1', 'u2', 'u3', 'cx', 'id', 'unitary']

    @property
    def supported_instructions(self):
        return None

    @property
    def coupling_map(self):
        return None

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


class QasmSimulatorPy(Backend):
    """Python implementation of a qasm simulator."""

    # Class level variable to return the final state at the end of simulation
    # This should be set to True for the statevector simulator
    SHOW_FINAL_STATE = False

    @property
    def local(self):
        return True

    @property
    def open_pulse(self):
        return False

    @property
    def memory(self):
        return True

    def __init__(self, name='qasm_simulator', **fields):
        super().__init__(name, **fields)
        self._target = QasmSimulatorTarget()
        # Define attributes in __init__.
        self._local_random = np.random.RandomState()
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = None
        self._chop_threshold = self.configuration.get('chop_threshold')
        # TEMP
        self._sample_measure = self.configuration.get('allow_sample_measuring')

    @classmethod
    def _default_config(cls):
        return Configuration(shots=1024, memory=False,
                             initial_statevector=None, chop_threshold=1e-15,
                             allow_sample_measuring=False)

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
        if self.configuration.get('initial_statevector') is None:
            return
        # Check statevector is correct length for number of qubits
        length = len(self._initial_statevector)
        required_dim = 2 ** self._number_of_qubits
        if length != required_dim:
            raise BasicAerError('initial statevector is incorrect length: ' +
                                '{} != {}'.format(length, required_dim))

    def _set_options(self):
        """Set the backend options for all experiments"""
        # Reset default options
        self._initial_statevector = self.configuration.get('initial_statevector')
        self._chop_threshold = self.configuration.get('chop_threshold')

        if self._initial_statevector is not None:
            # Check the initial statevector is normalized
            norm = np.linalg.norm(self._initial_statevector)
            if round(norm, 12) != 1:
                raise BasicAerError('initial statevector is not normalized: ' +
                                    'norm {} != 1'.format(norm))

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
            experiment (QobjCircuit): a QuantumCircuit experiment.
        """
        # If shots=1 we should disable measure sampling.
        # This is also required for statevector simulator to return the
        # correct final statevector without silently dropping final measurements.
        if self._shots <= 1:
            self._sample_measure = False
            return

        # If flag isn't found do a simple test to see if a circuit contains
        # no reset instructions, and no gates instructions after
        # the first measure.
        measure_flag = False
        for instruction in experiment.data:
            # If circuit contains reset operations we cannot sample
            if instruction[0].name == "reset":
                self._sample_measure = False
                return
            # If circuit contains a measure option then we can
            # sample only if all following operations are measures
            if measure_flag:
                # If we find a non-measure instruction
                # we cannot do measure sampling
                if instruction[0].name not in ["measure", "barrier", "id", "u0"]:
                    self._sample_measure = False
                    return
            elif instruction[0].name == "measure":
                measure_flag = True
        # If we made it to the end of the circuit without returning
        # measure sampling is allowed
        self._sample_measure = True

    def run(self, circuits):
        """Run circuits

        Args:
            circuits (list(QuantumCircuits): payload of the experiment

        Returns:
            List[Counts]: list
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        self._set_options()
        job_id = str(uuid.uuid4())
        result_list = []
        self._shots = self.configuration.get('shots')
        self._memory = self.configuration.get('memory', False)
        start = time.time()
        result_dict = collections.OrderedDict()
        if isinstance(circuits, QasmQobj):
            warnings.warn(
                'Passing in a Qobj object to run() is deprecated and support '
                'for it will be removed in the future. Instead pass circuits '
                'directly and use the backend to set run configuration ',
                DeprecationWarning, stacklevel=2)
            circuits, _, __ = disassemble(circuits)

        for experiment in circuits:
            result_dict[experiment.name] = self.run_experiment(experiment)
        end = time.time()
        time_taken = end - start
        return BasicAerJob(job_id, self,  result_dict, time_taken=time_taken)

    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.

        Args:
            experiment (QuantumCircuit): A quantumcircuit to run

        Returns:
            Counts: A counts object with the result of the experiment
        Raises:
            BasicAerError: if an error occurred.
        """
        start = time.time()
        self._number_of_qubits = experiment.num_qubits
        self._number_of_cmembits = experiment.num_clbits
        self._statevector = 0
        self._classical_memory = 0
        self._classical_register = 0
        self._sample_measure = False
        # Validate the dimension of initial statevector if set
        self._validate_initial_statevector()
        # Get the seed looking in circuit, qobj, and then random.
        seed_config = self.configuration.get('seed_simulator')
        if seed_config:
            seed_simulator = seed_config
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
            for operation in assemble_circuit(experiment):
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
                    backend = self.name
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
        data = {'counts': dict(collections.Counter(memory))}
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
        return Counts(data, name=experiment.name, shots=self._shots,
                      time_taken=end - start, seed_simulator=seed_simulator)

    def _validate(self, circuits):
        """Semantic validations of the qobj which cannot be done via schemas."""
        for circuit in circuits:
            if circuit.num_qubits > self.num_qubits:
                raise BasicAerError('Number of qubits {} '.format(circuit.num_qubits) +
                                    'is greater than maximum ({}) '.format(self.num_qubits) +
                                    'for "{}".'.format(self.name()))
        for experiment in circuits:
            name = experiment.name
            if experiment.num_clbits == 0:
                logger.warning('No classical registers in circuit "%s", '
                               'counts will be empty.', name)
            elif 'measure' not in experiment.count_ops:
                logger.warning('No measurements in circuit "%s", '
                               'classical register will remain all zeros.', name)

    def status(self):
        warnings.warn("The status method for QasmSimulatorPy is deprecated "
                      "and will be removed in a future release.",
                      DeprecationWarning, stacklevel=2)
        return BackendStatus(backend_name=self.name,
                             backend_version=__version__,
                             operational=True,
                             pending_jobs=0,
                             status_msg='')
