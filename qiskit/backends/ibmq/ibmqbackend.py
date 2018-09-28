# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""IbmQ module

This module is used for connecting to the Quantum Experience.
"""
import warnings
import logging

from IBMQuantumExperience import ApiError
from qiskit import QISKitError
from qiskit._util import _camel_case_to_snake_case, AvailableToOperationalDict, _dict_merge
from qiskit.backends import BaseBackend
from qiskit.backends.ibmq.ibmqjob import IBMQJob, IBMQJobPreQobj
from qiskit.backends import JobStatus

logger = logging.getLogger(__name__)


class IBMQBackend(BaseBackend):
    """Backend class interfacing with the Quantum Experience remotely.
    """

    def __init__(self, configuration, provider, credentials, api):
        """Initialize remote backend for IBM Quantum Experience.

        Args:
            configuration (dict): configuration of backend.
            provider (IBMQProvider): provider.
            credentials (Credentials): credentials.
            api (IBMQuantumExperience.IBMQuantumExperience.IBMQuantumExperience):
                api for communicating with the Quantum Experience.
        """
        super().__init__(provider=provider, configuration=configuration)
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

        self._credentials = credentials
        self.hub = credentials.hub
        self.group = credentials.group
        self.project = credentials.project

    def run(self, qobj):
        """Run qobj asynchronously.

        Args:
            qobj (dict): description of job

        Returns:
            IBMQJob: an instance derived from BaseJob
        """
        job_class = _job_class_from_backend_support(self)
        job = job_class(self, None, self._api, not self.configuration()['simulator'], qobj=qobj)
        job.submit()
        return job

    def calibration(self):
        """Return the online backend calibrations.

        The return is via QX API call.

        Returns:
            dict: The calibration of the backend.

        Raises:
            LookupError: If a calibration for the backend can't be found.

        :deprecated: will be removed after 0.7
        """
        warnings.warn("Backends will no longer return a calibration dictionary, "
                      "use backend.properties() instead.", DeprecationWarning)

        try:
            backend_name = self.name()
            calibrations = self._api.backend_calibration(backend_name)
            # FIXME a hack to remove calibration data that is none.
            # Needs to be fixed in api
            if backend_name == 'ibmq_qasm_simulator':
                calibrations = {}
        except Exception as ex:
            raise LookupError(
                "Couldn't get backend calibration: {0}".format(ex))

        calibrations_edit = {}
        for key, vals in calibrations.items():
            new_key = _camel_case_to_snake_case(key)
            calibrations_edit[new_key] = vals

        return calibrations_edit

    def parameters(self):
        """Return the online backend parameters.

        Returns:
            dict: The parameters of the backend.

        Raises:
            LookupError: If parameters for the backend can't be found.

        :deprecated: will be removed after 0.7
        """
        warnings.warn("Backends will no longer return a parameters dictionary, "
                      "use backend.properties() instead.", DeprecationWarning)

        try:
            backend_name = self.name()
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

    def properties(self):
        """Return the online backend properties.

        The return is via QX API call.

        Returns:
            dict: The properties of the backend.

        Raises:
            LookupError: If properties for the backend can't be found.
        """
        # FIXME: make this an actual call to _api.backend_properties
        # for now this api endpoint does not exist.
        warnings.simplefilter("ignore")
        calibration = self.calibration()
        parameters = self.parameters()
        _dict_merge(calibration, parameters)
        properties = calibration
        warnings.simplefilter("default")
        return properties

    def status(self):
        """Return the online backend status.

        Returns:
            dict: The status of the backend.

        Raises:
            LookupError: If status for the backend can't be found.
        """
        try:
            backend_name = self.configuration()['name']
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

    def jobs(self, limit=50, skip=0, status=None, db_filter=None):
        """Attempt to get the jobs submitted to the backend.

        Args:
            limit (int): number of jobs to retrieve
            skip (int): starting index of retrieval
            status (None or JobStatus or str): only get jobs with this status,
                where status is e.g. `JobStatus.RUNNING` or `'RUNNING'`
            db_filter (dict): `loopback-based filter
                <https://loopback.io/doc/en/lb2/Querying-data.html>`_.
                This is an interface to a database ``where`` filter. Some
                examples of its usage are:

                Filter last five jobs with errors::

                   job_list = backend.jobs(limit=5, status=JobStatus.ERROR)

                Filter last five jobs with counts=1024, and counts for
                states ``00`` and ``11`` each exceeding 400::

                  cnts_filter = {'shots': 1024,
                                 'qasms.result.data.counts.00': {'gt': 400},
                                 'qasms.result.data.counts.11': {'gt': 400}}
                  job_list = backend.jobs(limit=5, db_filter=cnts_filter)

                Filter last five jobs from 30 days ago::

                   past_date = datetime.datetime.now() - datetime.timedelta(days=30)
                   date_filter = {'creationDate': {'lt': past_date.isoformat()}}
                   job_list = backend.jobs(limit=5, db_filter=date_filter)

        Returns:
            list(IBMQJob): list of IBMQJob instances

        Raises:
            IBMQBackendValueError: status keyword value unrecognized
        """
        backend_name = self.configuration()['name']
        api_filter = {'backend.name': backend_name}
        if status:
            if isinstance(status, str):
                status = JobStatus[status]
            if status == JobStatus.RUNNING:
                this_filter = {'status': 'RUNNING',
                               'infoQueue': {'exists': False}}
            elif status == JobStatus.QUEUED:
                this_filter = {'status': 'RUNNING',
                               'infoQueue.status': 'PENDING_IN_QUEUE'}
            elif status == JobStatus.CANCELLED:
                this_filter = {'status': 'CANCELLED'}
            elif status == JobStatus.DONE:
                this_filter = {'status': 'COMPLETED'}
            elif status == JobStatus.ERROR:
                this_filter = {'status': {'regexp': '^ERROR'}}
            else:
                raise IBMQBackendValueError('unrecongized value for "status" keyword '
                                            'in job filter')
            api_filter.update(this_filter)
        if db_filter:
            # status takes precendence over db_filter for same keys
            api_filter = {**db_filter, **api_filter}
        job_info_list = self._api.get_jobs(limit=limit, skip=skip,
                                           filter=api_filter)
        job_list = []
        for job_info in job_info_list:
            job_class = _job_class_from_job_response(job_info)
            is_device = not bool(self._configuration.get('simulator'))
            job = job_class(self, job_info.get('id'), self._api, is_device,
                            creation_date=job_info.get('creationDate'))
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
        try:
            job_info = self._api.get_job(job_id)
            if 'error' in job_info:
                raise IBMQBackendError('Failed to get job "{}": {}'
                                       .format(job_id, job_info['error']))
        except ApiError as ex:
            raise IBMQBackendError('Failed to get job "{}":{}'
                                   .format(job_id, str(ex)))
        job_class = _job_class_from_job_response(job_info)
        is_device = not bool(self._configuration.get('simulator'))
        job = job_class(self, job_info.get('id'), self._api, is_device,
                        creation_date=job_info.get('creationDate'))
        return job

    def __repr__(self):
        credentials_info = ''
        if self.hub:
            credentials_info = '{}, {}, {}'.format(self.hub, self.group,
                                                   self.project)
        return "<{}('{}') from IBMQ({})>".format(
            self.__class__.__name__, self.name(), credentials_info)


class IBMQBackendError(QISKitError):
    """IBM Q Backend Errors"""
    pass


class IBMQBackendValueError(IBMQBackendError, ValueError):
    """ Value errors thrown within IBMQBackend """
    pass


def _job_class_from_job_response(job_response):
    is_qobj = job_response.get('kind', None) == 'q-object'
    return IBMQJob if is_qobj else IBMQJobPreQobj


def _job_class_from_backend_support(backend):
    support_qobj = backend.configuration().get('allow_q_object')
    return IBMQJob if support_qobj else IBMQJobPreQobj
