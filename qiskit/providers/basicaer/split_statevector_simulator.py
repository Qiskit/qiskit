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


"""Contains a (slow) python statevector simulator which splits for every non-zero probability measurement

It simulates the statevector through a quantum circuit. It is exponential in
the number of qubits, and in the number of non-zero probability measurements within the circuit.

We advise using the c++ simulator or online simulator for larger size systems.

The input is a qobj dictionary and the output is a Result object.

The input qobj to this simulator has no shots, no reset, no noise.

The final result is a dictionary containing a binary tree data structure for the different splits, kept under statevector_tree
"""

import logging
import copy
from math import log2
import numpy as np
from qiskit.util import local_hardware_info
from qiskit.providers.basicaer.exceptions import BasicAerError
from qiskit.providers.models import QasmBackendConfiguration
from .qasm_simulator import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulatorPy

logger = logging.getLogger(__name__)


class SplitStatevectorSimulatorPy(StatevectorSimulatorPy):
    """Python split statevector simulator."""

    # TODO: Set a memory bound which depends on the amount of measurements as well
    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'split_statevector_simulator',
        'backend_version': '1.0.0',
        'n_qubits': min(24, MAX_QUBITS_MEMORY),
        'url': 'https://github.com/Qiskit/qiskit-terra',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'coupling_map': None,
        'description': 'A Python statevector simulator for qobj files',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'id', 'snapshot'],
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
                'name': 'snapshot',
                'parameters': ['slot'],
                'qasm_def': 'gate snapshot(slot) q { TODO }'
            }
        ]
    }

    # This should be set to True to use the split statevector reperesentation for measurments
    SPLIT_STATES = True

    
    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=(
            configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)),
                         provider=provider)
        # _substates field will contain 2 statevectors corresponding to measurement results 0 and 1
        self._substates = []
        self._substate_probabilities = []

    def run(self, qobj, backend_options=None):
        """Run qobj asynchronously.

        Args:
            qobj (Qobj): payload of the experiment
            backend_options (dict): backend options

        Returns:
            BasicAerJob: derived from BaseJob

        Additional Information::

            backend_options: Is a dict of options for the backend. It may contain
                * "initial_statevector": vector_like
                * "chop_threshold": double

            The "initial_statevector" option specifies a custom initial
            initial statevector for the simulator to be used instead of the all
            zero state. This size of this vector must be correct for the number
            of qubits in all experiments in the qobj.

            The "chop_threshold" option specifies a truncation value for
            setting small values to zero in the output statevector. The default
            value is 1e-15.

            Example::

                backend_options = {
                    "initial_statevector": np.array([1, 0, 0, 1j]) / np.sqrt(2),
                    "chop_threshold": 1e-15
                }
        """
        return super().run(qobj, backend_options=backend_options)

    def _get_single_outcome(self, qubit, wanted_state):
        """Simulate the outcome of a determined measurement of a qubit. 
        This function mainly exists to perserve the existing function division, 
        replacing _get_measure_outcome which is called for the other simulators

        Args:
            qubit (int): the qubit to measure
            wanted_state (int): the outcome state to return

        Return:
            tuple: pair (outcome, probability) where outcome is '0' or '1' and
            probability is the probability of the returned outcome.
        """
        # Axis for numpy.sum to compute probabilities
        axis = list(range(self._number_of_qubits))
        axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis))
        # Compute einsum index string for 1-qubit matrix multiplication

        return str(wanted_state), probabilities[wanted_state]

    def _split_statevector(self, qubit, probabilities, cmembit, cregbit=None):
        """Split the statevector into two substates corresponding to the two measurement options.

        Args:
            qubit (int): the measured qubit
            probabilities (list[float]): the probabilities of measuring 0 or 1
            cmembit (int): the classical memory bit to store outcome in.
            cregbit (int, optional): is the classical register bit to store outcome in.

        Raises:
            BasicAerError: if an error occurred.
        """
        # Getting the data for each split possibility
        outcome0, probability0 = self._get_single_outcome(qubit, 0)
        outcome1, probability1 = self._get_single_outcome(qubit, 1)

        # Copying the statevector into its two substates
        temp = copy.deepcopy(self)
        self._substates.append(copy.deepcopy(temp))
        self._substate_probabilities.append(probabilities[0]) 
        self._substates.append(copy.deepcopy(temp))
        self._substate_probabilities.append(probabilities[1])

        # Updating the states after the split
        self._substates[0]._update_state_after_measure(qubit, cmembit, outcome0, probability0, cregbit)
        self._substates[1]._update_state_after_measure(qubit, cmembit, outcome1, probability1, cregbit)

    def _generate_data(self):
        """Run an experiment (circuit) and return a single experiment result.

        Args:
            experiment (QobjExperiment): experiment from qobj experiments list

        Returns:
             data: A result dictionary in the form of a binary tree which looks something like::

                {
                "value": The statevector of the state at the first measurement
                "prob_0": The probability to get a measurement of 0 at the first measurement
                "prob_1": The probability to get a measurement of 1 at the first measurement
                "path_0": 
                    {
                        "value": The statevector evolved from result 0 in the first measurement,
                                 at the second measurement or at the end of the circuit
                        "prob_0": The probability to get a measurement of 0 at the second measurement
                        "prob_1": The probability to get a measurement of 1 at the second measurement
                        "path_0":
                            {
                                ...
                            }
                        "path_1":
                            {
                                ...
                            }
                    }
                "path_1":
                    {
                        ...
                    }
                }
        Raises:
            BasicAerError: if an error occurred.
        """    
        if len(self._substates) != 0:
            data = {'value': self._get_statevector(),
                    'path_0': self._substates[0]._generate_data(), 'path_0_probability': self._substate_probabilities[0],
                    'path_1': self._substates[1]._generate_data(), 'path_1_probability': self._substate_probabilities[1]}
        else:
            data = {'value': self._get_statevector()}
        
        return data
    # TODO: Add a _validate method