# -*- coding: utf-8 -*-

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
"""IbmQ module

This module is used for connecting to the Quantum Experience.
"""
import logging

from qiskit import QISKitError
from qiskit._compiler import compile_circuit
from qiskit._resulterror import ResultError

from qiskit._util import _snake_case_to_camel_case
from qiskit.backends import BaseBackend
from qiskit.backends.ibmq.ibmqjob import IBMQJob

logger = logging.getLogger(__name__)


class IBMQBackend(BaseBackend):
    """Backend class interfacing with the Quantum Experience remotely.
    """

    def __init__(self, configuration, api=None):
        """Initialize remote backend for IBM Quantum Experience.

        Args:
            configuration (dict): configuration of backend.
            api (IBMQuantumExperience.IBMQuantumExperience.IBMQuantumExperience):
                api for communicating with the Quantum Experience.
        """
        super().__init__(configuration=configuration)
        self._api = api
        if self._configuration:
            configuration_edit = {}
            for key, vals in self._configuration.items():
                new_key = _snake_case_to_camel_case(key)
                configuration_edit[new_key] = vals
            self._configuration = configuration_edit
            # FIXME: This is a hack to make sure that the
            # local : False is added to the online device
            self._configuration['local'] = False

    def run(self, q_job):
        """Run q_job asynchronously.

        Args:
            q_job (QuantumJob): description of job

        Returns:
            IBMQJob: an instance derived from BaseJob
        """
        submit_info = self._submit(q_job)
        return IBMQJob(q_job, self._api, submit_info,
                       not self.configuration['simulator'])

    def _submit(self, q_job):
        """Submit job to IBM Q.

        Args:
            q_job (QuantumJob): job to run

        Returns:
            dict: submission info including job id from server

        Raises:
            QISKitError: The backend name in the job doesn't match this backend.
            ResultError: If the API reported an error with the submitted job.
        """
        qobj = q_job.qobj
        api_jobs = []
        for circuit in qobj['circuits']:
            if (('compiled_circuit_qasm' not in circuit) or
                    (circuit['compiled_circuit_qasm'] is None)):
                compiled_circuit = compile_circuit(circuit['circuit'])
                circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)
            if isinstance(circuit['compiled_circuit_qasm'], bytes):
                api_jobs.append({'qasm': circuit['compiled_circuit_qasm'].decode()})
            else:
                api_jobs.append({'qasm': circuit['compiled_circuit_qasm']})

        seed0 = qobj['circuits'][0]['config']['seed']
        hpc = None
        if (qobj['config']['backend_name'] == 'ibmq_qasm_simulator_hpc' and
                'hpc' in qobj['config']):
            try:
                # Use CamelCase when passing the hpc parameters to the API.
                hpc = {
                    'multiShotOptimization':
                        qobj['config']['hpc']['multi_shot_optimization'],
                    'ompNumThreads':
                        qobj['config']['hpc']['omp_num_threads']
                }
            except (KeyError, TypeError):
                hpc = None

        backend_name = qobj['config']['backend_name']
        if backend_name != self.name:
            raise QISKitError("inconsistent qobj backend "
                              "name ({0} != {1})".format(backend_name, self.name))
        submit_info = self._api.run_job(api_jobs, backend_name,
                                        shots=qobj['config']['shots'],
                                        max_credits=qobj['config']['max_credits'],
                                        seed=seed0,
                                        hpc=hpc)
        if 'error' in submit_info:
            raise ResultError(submit_info['error'])
        return submit_info

    @property
    def calibration(self):
        """Return the online backend calibrations.

        The return is via QX API call.

        Returns:
            dict: The calibration of the backend.

        Raises:
            LookupError: If a configuration for the backend can't be found.
        """
        try:
            backend_name = self.configuration['name']
            calibrations = self._api.backend_calibration(backend_name)
            # FIXME a hack to remove calibration data that is none.
            # Needs to be fixed in api
            if backend_name == 'ibmqx_hpc_qasm_simulator':
                calibrations = {}
            # FIXME a hack to remove calibration data that is none.
            # Needs to be fixed in api
            if backend_name == 'ibmqx_qasm_simulator':
                calibrations = {}
        except Exception as ex:
            raise LookupError(
                "Couldn't get backend calibration: {0}".format(ex))

        calibrations_edit = {}
        for key, vals in calibrations.items():
            new_key = _snake_case_to_camel_case(key)
            calibrations_edit[new_key] = vals

        return calibrations_edit

    @property
    def parameters(self):
        """Return the online backend parameters.

        Returns:
            dict: The parameters of the backend.

        Raises:
            LookupError: If parameters for the backend can't be found.
        """
        try:
            backend_name = self.configuration['name']
            parameters = self._api.backend_parameters(backend_name)
            # FIXME a hack to remove parameters data that is none.
            # Needs to be fixed in api
            if backend_name == 'ibmqx_hpc_qasm_simulator':
                parameters = {}
            # FIXME a hack to remove parameters data that is none.
            # Needs to be fixed in api
            if backend_name == 'ibmqx_qasm_simulator':
                parameters = {}
        except Exception as ex:
            raise LookupError(
                "Couldn't get backend parameters: {0}".format(ex))

        parameters_edit = {}
        for key, vals in parameters.items():
            new_key = _snake_case_to_camel_case(key)
            parameters_edit[new_key] = vals

        return parameters_edit

    @property
    def status(self):
        """Return the online backend status.

        Returns:
            dict: The status of the backend.

        Raises:
            LookupError: If status for the backend can't be found.
        """
        try:
            backend_name = self.configuration['name']
            status = self._api.backend_status(backend_name)
            # FIXME a hack to rename the key. Needs to be fixed in api
            status['name'] = status['backend']
            del status['backend']
            # FIXME a hack to remove the key busy.  Needs to be fixed in api
            if 'busy' in status:
                del status['busy']
            # FIXME a hack to add available to the hpc simulator.  Needs to
            # be fixed in api
            if status['name'] == 'ibmqx_hpc_qasm_simulator':
                status['available'] = True

        except Exception as ex:
            raise LookupError(
                "Couldn't get backend status: {0}".format(ex))
        return status
