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

from qiskit.result import Result
from qiskit.providers import BaseBackend
from qiskit.providers import BaseJob
from qiskit.providers.models import BackendProperties, BackendConfiguration
from qiskit.providers.models.backendconfiguration import GateConfig
from qiskit.qobj import Qobj, QobjItem, QobjConfig, QobjHeader, QobjInstruction
from qiskit.qobj import QobjExperiment, QobjExperimentHeader
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.baseprovider import BaseProvider

logger = logging.getLogger(__name__)


class DummyProvider(BaseProvider):
    """Dummy provider just for testing purposes."""

    def get_backend(self, name=None, **kwargs):
        return self._backend

    def backends(self, name=None, **kwargs):
        return [self._backend]

    def __init__(self):
        self._backend = DummySimulator()

        super().__init__()


class DummySimulator(BaseBackend):
    """ This is Dummy backend simulator just for testing purposes """

    DEFAULT_CONFIGURATION = {
        'name': 'local_dummy_simulator',
        'url': 'https://github.com/Qiskit/qiskit-terra',
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

    def properties(self):
        """Return backend properties"""
        properties = {
            'backend_name': self.name(),
            'backend_version': self.configuration().backend_version,
            'last_update_date': '2000-01-01 00:00:00Z',
            'qubits': [[{'name': 'TODO', 'date': '2000-01-01 00:00:00Z',
                         'unit': 'TODO', 'value': 0}]],
            'gates': [{'qubits': [0], 'gate': 'TODO',
                       'parameters':
                           [{'name': 'TODO', 'date': '2000-01-01 00:00:00Z',
                             'unit': 'TODO', 'value': 0}]}],
            'general': []
        }

        return BackendProperties.from_dict(properties)

    def run(self, qobj):
        job_id = str(uuid.uuid4())
        job = DummyJob(self.run_job, qobj, job_id, self)
        job.submit()
        return job

    # pylint: disable=unused-argument
    def run_job(self, job_id, qobj):
        """ Main dummy simulator loop """
        time.sleep(self.time_alive)

        return Result.from_dict(
            {'job_id': job_id, 'result': [], 'status': 'COMPLETED'})


class DummyJob(BaseJob):
    """Dummy simulator job"""
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, qobj, job_id, backend):
        super().__init__()
        self._job_id = job_id
        self._backend = backend
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

    def job_id(self):
        return self._job_id

    def backend(self):
        return self._backend

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


class FakeBackend(object):
    """A fake backend.
    """

    def name(self):
        """ name of fake backend"""
        return 'qiskit_is_cool'

    def configuration(self):
        """Return a make up configuration for a fake device."""
        qx5_cmap = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                    [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10], [11, 10],
                    [12, 5], [12, 11], [12, 13], [13, 4], [13, 14], [15, 0],
                    [15, 2], [15, 14]]
        return BackendConfiguration(
            backend_name='fake',
            backend_version='0.0.0',
            n_qubits=16,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=qx5_cmap,
        )
