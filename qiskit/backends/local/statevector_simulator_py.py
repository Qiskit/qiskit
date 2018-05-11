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

"""Contains a (slow) python statevector simulator.

It simulates the statevector through a quantum circuit. It is exponential in
the number of qubits.

We advise using the c++ simulator or online simulator for larger size systems.

The input is a qobj dictionary and the output is a Result object.

The input qobj to this simulator has no shots, no measures, no reset, no noise.
"""
import logging
from qiskit._result import Result
from qiskit.backends.local.localjob import LocalJob
from qiskit.backends.local._simulatorerror import SimulatorError
from .qasm_simulator_py import QasmSimulatorPy

logger = logging.getLogger(__name__)


class StatevectorSimulatorPy(QasmSimulatorPy):
    """Python statevector simulator."""

    DEFAULT_CONFIGURATION = {
        'name': 'local_statevector_simulator_py',
        'url': 'https://github.com/QISKit/qiskit-sdk-py',
        'simulator': True,
        'local': True,
        'description': 'A Python statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,id,snapshot'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

    def run(self, q_job):
        """Run q_job asynchronously.

        Args:
            q_job (QuantumJob): QuantumJob object

        Returns:
            LocalJob: derived from BaseJob
        """
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
        result = super()._run_job(q_job)._result
        # Replace backend name with current backend
        result['backend'] = self._configuration['name']
        # Extract final state snapshot and move to 'statevector' data field
        for res in result['result']:
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
        return Result(result, qobj)

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
                            "Setting shots=1 for circuit %s.", circuit['name'])
                circuit['config']['shots'] = 1
            for op in circuit['compiled_circuit']['operations']:
                if op['name'] in ['measure', 'reset']:
                    raise SimulatorError("In circuit {}: statevector simulator does "
                                         "not support measure or reset.".format(circuit['name']))
        return
