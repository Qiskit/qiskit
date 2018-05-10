# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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
Interface to C++ quantum circuit simulator with realistic noise.
"""

import logging

from .qasm_simulator_cpp import QasmSimulatorCpp
from ._simulatorerror import SimulatorError
from .localjob import LocalJob

logger = logging.getLogger(__name__)


class StatevectorSimulatorCpp(QasmSimulatorCpp):
    """C++ statevector simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_statevector_simulator_cpp',
        'url': 'https://github.com/QISKit/qiskit-sdk-py/src/qasm-simulator-cpp',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,load,save,snapshot'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

    def run(self, q_job):
        """Run a QuantumJob on the the backend."""
        return LocalJob(self._run_job, q_job)

    def _run_job(self, q_job):
        """Run a QuantumJob on the backend."""
        qobj = q_job.qobj
        self._validate(qobj)
        final_state_key = 32767  # Internal key for final state snapshot
        # Add final snapshots to circuits
        for circuit in qobj['circuits']:
            circuit['compiled_circuit']['operations'].append(
                {'name': 'snapshot', 'params': [final_state_key]})
        result = super()._run_job(q_job)
        # Extract final state snapshot and move to 'statevector' data field
        for res in result._result['result']:
            snapshots = res['data']['snapshots']
            if str(final_state_key) in snapshots:
                final_state_key = str(final_state_key)
            # Pop off final snapshot added above
            final_state = snapshots.pop(final_state_key, None)
            final_state = final_state['statevector'][0]
            # Add final state to results data
            res['data']['statevector'] = final_state
            # Remove snapshot dict if empty
            if snapshots == {}:
                res['data'].pop('snapshots', None)
        return result

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. No shots
        2. No measurements in the middle
        """
        if qobj['config']['shots'] != 1:
            logger.info("statevector simulator only supports 1 shot. "
                        "Setting shots=1.")
            qobj['config']['shots'] = 1
        for circuit in qobj['circuits']:
            if 'shots' in circuit['config'] and circuit['config']['shots'] != 1:
                logger.info("statevector simulator only supports 1 shot. "
                            "Setting shots=1 for circuit %s", circuit['name'])
                circuit['config']['shots'] = 1
            for op in circuit['compiled_circuit']['operations']:
                if op['name'] in ['measure', 'reset']:
                    raise SimulatorError("In circuit {}: statevector simulator does "
                                         "not support measure or reset.".format(circuit['name']))
        return
