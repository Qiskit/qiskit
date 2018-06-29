# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""IbmQ module

This module is used for connecting to the Quantum Experience.
"""
import logging

from qiskit import QISKitError
from qiskit._util import _camel_case_to_snake_case, AvailableToOperationalDict
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
                new_key = _camel_case_to_snake_case(key)
                configuration_edit[new_key] = vals
            self._configuration = configuration_edit
            # FIXME: This is a hack to make sure that the
            # local : False is added to the online device
            self._configuration['local'] = False

    def run(self, qobj):
        """Run qobj asynchronously.

        Args:
            qobj (dict): description of job

        Returns:
            IBMQJob: an instance derived from BaseJob
        """
        return IBMQJob(qobj, self._api, not self.configuration['simulator'])

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
            if backend_name in ('ibmq_qasm_simulator', 'ibmqx_qasm_simulator'):
                calibrations = {}
        except Exception as ex:
            raise LookupError(
                "Couldn't get backend calibration: {0}".format(ex))

        calibrations_edit = {}
        for key, vals in calibrations.items():
            new_key = _camel_case_to_snake_case(key)
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
            if backend_name == 'ibmq_qasm_simulator':
                parameters = {}
        except Exception as ex:
            raise LookupError(
                "Couldn't get backend parameters: {0}".format(ex))

        parameters_edit = {}
        for key, vals in parameters.items():
            new_key = _camel_case_to_snake_case(key)
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

            # FIXME: this needs to be replaced at the API level - eventually
            # it will.
            if 'available' in status:
                status['operational'] = status['available']
                del status['available']
        except Exception as ex:
            raise LookupError(
                "Couldn't get backend status: {0}".format(ex))
        return AvailableToOperationalDict(status)

    def jobs(self, limit=50, skip=0):
        """Attempt to get the jobs submitted to the backend

        Args:
            limit (int): number of jobs to retrieve
            skip (int): starting index of retrieval
        Returns:
            list(IBMQJob): list of IBMQJob instances
        """
        backend_name = self.configuration['name']
        job_info_list = self._api.get_jobs(limit=limit, skip=skip,
                                           backend=backend_name)
        job_list = []
        for job_info in job_info_list:
            is_device = not bool(self._configuration.get('simulator'))
            job = IBMQJob.from_api(job_info, self._api, is_device)
            job_list.append(job)
        return job_list

    def retrieve_job(self, job_id):
        """Attempt to get the specified job by job_id

        Args:
            job_id (str): the job id of the job to retrieve

        Returns:
            IBMQJob: class instance

        Raises:
            IBMQBackendError: if retrieval failed
        """
        job_info = self._api.get_job(job_id)
        if 'error' in job_info:
            raise IBMQBackendError('failed to get job id "{}"'.format(job_id))
        is_device = not bool(self._configuration.get('simulator'))
        job = IBMQJob.from_api(job_info, self._api, is_device)
        return job


class IBMQBackendError(QISKitError):
    """IBM Q Backend Errors"""
    pass
