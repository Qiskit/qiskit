# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Interface to C++ quantum circuit simulator with realistic noise.
"""

import logging

from qiskit.qobj import QobjInstruction
from .qasm_simulator_cpp import QasmSimulatorCpp
from ._simulatorerror import SimulatorError
from .localjob import LocalJob

logger = logging.getLogger(__name__)


class StatevectorSimulatorCpp(QasmSimulatorCpp):
    """C++ statevector simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_statevector_simulator_cpp',
        'url': 'https://github.com/QISKit/qiskit-terra/src/qasm-simulator-cpp',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,load,save,snapshot'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

    def run(self, qobj):
        """Run a qobj on the the backend."""
        return LocalJob(self._run_job, qobj)

    def _run_job(self, qobj):
        """Run a Qobj on the backend."""
        self._validate(qobj)
        final_state_key = 32767  # Internal key for final state snapshot
        # Add final snapshots to circuits
        for experiment in qobj.experiments:
            experiment.instructions.append(
                QobjInstruction.from_dict({'name': '#snapshot', 'params': [final_state_key]})
            )
        result = super()._run_job(qobj)
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
        if qobj.config.shots != 1:
            logger.info("statevector simulator only supports 1 shot. "
                        "Setting shots=1.")
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info("statevector simulator only supports 1 shot. "
                            "Setting shots=1 for circuit %s.", experiment.name)
                experiment.config.shots = 1
            for op in experiment.instructions:
                if op.name in ['measure', 'reset']:
                    raise SimulatorError(
                        "In circuit {}: statevector simulator does not support "
                        "measure or reset.".format(experiment.header.name))
