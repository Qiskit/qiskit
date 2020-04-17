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

    # Override base class value to return the final state vector
    SHOW_FINAL_STATE = True

    def __init__(self, **fields):
        super().__init__('statevector_simulator', **fields)

    def run(self, circuits):
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
        return super().run(circuits)

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
