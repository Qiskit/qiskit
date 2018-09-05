# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Supporting fake, stubs and mocking classes.

The module includes, among other, a dummy backend simulator. The purpose of
this class is to create a Simulator that we can trick for testing purposes:
testing local timeouts, arbitrary responses or behavior, etc.
"""

import uuid
import logging
from concurrent import futures
import time

from qiskit import Result
from qiskit.backends import BaseBackend
from qiskit.backends import BaseJob
from qiskit.qobj import Qobj, QobjItem, QobjConfig, QobjHeader, QobjInstruction
from qiskit.qobj import QobjExperiment, QobjExperimentHeader
from qiskit.backends.jobstatus import JobStatus
from qiskit.backends.baseprovider import BaseProvider

logger = logging.getLogger(__name__)


class DummyProvider(BaseProvider):
    """Dummy provider just for testing purposes."""
    def __init__(self):
        self._backend = DummySimulator()

        super().__init__()

    def get_backend(self, name):
        return self._backend

    def available_backends(self, filters=None):
        # pylint: disable=arguments-differ
        backends = {DummySimulator.name: self._backend}

        filters = filters or {}
        for key, value in filters.items():
            backends = {name: instance for name, instance in backends.items()
                        if instance.configuration().get(key) == value}
        return list(backends.values())


class DummySimulator(BaseBackend):
    """ This is Dummy backend simulator just for testing purposes """

    DEFAULT_CONFIGURATION = {
        'name': 'local_dummy_simulator',
        'url': 'https://github.com/QISKit/qiskit-terra',
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

    def run(self, qobj):
        job = DummyJob(self.run_job, qobj)
        job.submit()
        return job

    # pylint: disable=unused-argument
    def run_job(self, qobj):
        """ Main dummy simulator loop """
        job_id = str(uuid.uuid4())
        time.sleep(self.time_alive)

        return Result(
            {'job_id': job_id, 'result': [], 'status': 'COMPLETED'})


class DummyJob(BaseJob):
    """Dummy simulator job"""
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, qobj):
        super().__init__()
        self._qobj = qobj
        self._future = None
        self._future_callback = fn

    def submit(self):
        self._future = self._executor.submit(self._future_callback, self._qobj)

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    def status(self):
        if self._running:
            _status = JobStatus.RUNNING
        elif not self._done:
            _status = JobStatus.QUEUED
        elif self._cancelled:
            _status = JobStatus.CANCELLED
        elif self._done:
            _status = JobStatus.DONE
        elif self._error:
            _status = JobStatus.ERROR
        else:
            raise Exception('Unexpected state of {0}'.format(
                self.__class__.__name__))
        _status_msg = None
        return {'status': _status,
                'status_msg': _status_msg}

    @property
    def _cancelled(self):
        return self._future.cancelled()

    @property
    def _done(self):
        return self._future.done()

    @property
    def _running(self):
        return self._future.running()

    @property
    def _error(self):
        return self._future.exception(timeout=0)


def new_fake_qobj():
    """Create fake `Qobj` and backend instances."""
    backend = FakeBackend()
    return Qobj(
        qobj_id='test-id',
        config=QobjConfig(shots=1024, memory_slots=1, max_credits=100),
        header=QobjHeader(backend_name=backend.name()),
        experiments=[QobjExperiment(
            instructions=[
                QobjInstruction(name='barrier', qubits=[1])
            ],
            header=QobjExperimentHeader(compiled_circuit_qasm='fake-code'),
            config=QobjItem(seed=123456)
        )]
    )


class FakeBackend():
    """Fakes qiskit.backends.basebackend.BaseBackend instances."""

    def name(self):
        """Return the name of the backend."""
        return 'test-backend'
