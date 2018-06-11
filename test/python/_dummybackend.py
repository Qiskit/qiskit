# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Dummy backend simulator.
The purpose of this class is to create a Simulator that we can trick for testing
purposes. Testing local timeouts, arbitrary responses or behavior, etc.
"""

import uuid
import logging
from concurrent import futures
import time

from qiskit import Result
from qiskit.backends import BaseBackend
from qiskit.backends import BaseJob
from qiskit.backends import JobStatus
from qiskit.backends.baseprovider import BaseProvider

logger = logging.getLogger(__name__)


class DummyProvider(BaseProvider):
    """Dummy provider just for testing purposes."""
    def get_backend(self, name):
        return DummySimulator()

    def available_backends(self, filters=None):
        # pylint: disable=arguments-differ
        backends = {DummySimulator.name: DummySimulator()}

        filters = filters or {}
        for key, value in filters.items():
            backends = {name: instance for name, instance in backends.items()
                        if instance.configuration.get(key) == value}
        return list(backends.values())


class DummySimulator(BaseBackend):
    """ This is Dummy backend simulator just for testing purposes """

    DEFAULT_CONFIGURATION = {
        'name': 'local_dummy_simulator',
        'url': 'https://github.com/QISKit/qiskit-core',
        'simulator': True,
        'local': True,
        'description': 'A dummy simulator for testing purposes',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,id'
    }

    def __init__(self, configuration=None, time_alive=10):
        """
        Args:
            configuration (dict): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
        self.time_alive = time_alive

    def run(self, q_job):
        return DummyJob(self.run_job, q_job)

    # pylint: disable=unused-argument
    def run_job(self, q_job):
        """ Main dummy simulator loop """
        job_id = str(uuid.uuid4())

        time.sleep(self.time_alive)

        return Result({'job_id': job_id, 'result': [], 'status': 'COMPLETED'})


class DummyJob(BaseJob):
    """dummy simulator job"""
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, qobj):
        super().__init__()
        self._qobj = qobj
        self._future = self._executor.submit(fn, qobj)

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    def status(self):
        if self.running:
            _status = JobStatus.RUNNING
        elif not self.done:
            _status = JobStatus.QUEUED
        elif self.cancelled:
            _status = JobStatus.CANCELLED
        elif self.done:
            _status = JobStatus.DONE
        elif self.error:
            _status = JobStatus.ERROR
        else:
            raise Exception('Unexpected state of {0}'.format(
                self.__class__.__name__))
        _status_msg = None
        return {'status': _status,
                'status_msg': _status_msg}

    @property
    def cancelled(self):
        return self._future.cancelled()

    @property
    def done(self):
        return self._future.done()

    @property
    def running(self):
        return self._future.running()

    @property
    def error(self):
        """
        Return Exception object if exception occured else None.

        Returns:
            Exception: exception raised by attempting to run job.
        """
        return self._future.exception(timeout=0)
