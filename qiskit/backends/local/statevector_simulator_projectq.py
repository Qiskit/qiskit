# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Interface to projectq circuit simulator.
"""

import logging

from qiskit._result import Result
from .qasm_simulator_projectq import QasmSimulatorProjectQ
from ._simulatorerror import SimulatorError

logger = logging.getLogger(__name__)


class StatevectorSimulatorProjectQ(QasmSimulatorProjectQ):
    """ProjectQ statevector simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_statevector_simulator_projectq',
        'url': 'https://projectq.ch',
        'simulator': True,
        'local': True,
        'description': 'A ProjectQ statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        'basis_gates': 'h,s,t,cx,id'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

    def run(self, q_job):
        """Run a QuantumJob on the backend."""
        
        qobj = q_job.qobj
        self._validate(qobj)
        result = super().run(q_job)._result
            
        return Result(result, qobj)

    # TODO: Remove duplication with other files, e.g. statevector_simulator_cpp.py
    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. No shots
        2. No measurements in the middle
        """
        if qobj['config']['shots'] != 1:
            logger.warning("statevector simulator only supports 1 shot. "
                           "Setting shots=1.")
            qobj['config']['shots'] = 1
        for circuit in qobj['circuits']:
            if 'shots' in circuit['config'] and circuit['config']['shots'] != 1:
                logger.warning("statevector simulator only supports 1 shot. "
                               "Setting shots=1 for circuit %s", circuit['name'])
                circuit['config']['shots'] = 1
            for op in circuit['compiled_circuit']['operations']:
                if op['name'] == 'measure':
                    raise SimulatorError("In circuit {}: statevector simulator does "
                                         "not support measure.".format(circuit['name']))
        return
