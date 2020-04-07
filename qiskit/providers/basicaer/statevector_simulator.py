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


"""Contains a (slow) python statevector simulator.

It simulates the statevector through a quantum circuit. It is exponential in
the number of qubits.

We advise using the c++ simulator or online simulator for larger size systems.

The input is a qobj dictionary and the output is a Result object.

The input qobj to this simulator has no shots, no measures, no reset, no noise.
"""

import logging
from math import log2
from qiskit.util import local_hardware_info
from qiskit.providers.basicaer.exceptions import BasicAerError
from qiskit.providers.models import QasmBackendConfiguration
from .qasm_simulator import QasmSimulatorPy

logger = logging.getLogger(__name__)


class StatevectorSimulatorPy(QasmSimulatorPy):
    """Python statevector simulator."""

    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'statevector_simulator',
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

    # Override base class value to return the final state vector
    SHOW_FINAL_STATE = True

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=(
            configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)),
                         provider=provider)

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

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. No shots
        2. No measurements in the middle
        """
        num_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if num_qubits > max_qubits:
            raise BasicAerError('Number of qubits {} '.format(num_qubits) +
                                'is greater than maximum ({}) '.format(max_qubits) +
                                'for "{}".'.format(self.name()))
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.',
                        self.name())
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info('"%s" only supports 1 shot. '
                            'Setting shots=1 for circuit "%s".',
                            self.name(), name)
                experiment.config.shots = 1
