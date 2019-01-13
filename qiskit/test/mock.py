# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""
Utilities for mocking the IBMQ provider, including job responses and backends.

The module includes dummy provider, backends, and jobs. The purpose of
these classes is to trick backends for testing purposes:
testing local timeouts, arbitrary responses or behavior, etc.

The mock devices are mainly for testing the compiler.
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
from qiskit.providers.ibmq.api import ApiError
from qiskit.qobj import Qobj, QobjItem, QobjConfig, QobjHeader, QobjInstruction
from qiskit.qobj import QobjExperiment, QobjExperimentHeader
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.baseprovider import BaseProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError


logger = logging.getLogger(__name__)


class FakeProvider(BaseProvider):
    """Dummy provider just for testing purposes.

    Only filtering backends by name is implemented.
    """

    def get_backend(self, name=None, **kwargs):
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends
                                 if backend.name() == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()
            else:
                backend = filtered_backends[0]
        return backend

    def backends(self, name=None, **kwargs):
        return self._backends

    def __init__(self):
        self._backends = [FakeQasmSimulator(),
                          FakeTenerife(),
                          FakeMelbourne(),
                          FakeRueschlikon(),
                          FakeTokyo()]
        super().__init__()


class FakeBackend(BaseBackend):
    """This is a dummy backend just for testing purposes."""

    def __init__(self, configuration, time_alive=10):
        """
        Args:
            configuration (dict): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration)
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
        job = FakeJob(self, job_id, self.run_job, qobj)
        job.submit()
        return job

    # pylint: disable=unused-argument
    def run_job(self, job_id, qobj):
        """Main dummy run loop"""
        time.sleep(self.time_alive)

        return Result.from_dict(
            {'job_id': job_id, 'result': [], 'status': 'COMPLETED'})


class FakeQasmSimulator(FakeBackend):
    """A fake simulator backend."""

    def __init__(self):
        _configuration = BackendConfiguration(
            backend_name='fake_qasm_simulator',
            backend_version='0.0.0',
            n_qubits=5,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                         'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
                         'snapshot', 'unitary'],
            simulator=True,
            local=True,
            conditional=True,
            open_pulse=False,
            memory=True,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')]
        )

        super().__init__(_configuration)


class FakeTenerife(FakeBackend):
    """A fake 5 qubit backend."""

    def __init__(self):
        """
            1
          / |
        0 - 2 - 3
            | /
            4
        """
        cmap = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]

        _configuration = BackendConfiguration(
            backend_name='fake_tenerife',
            backend_version='0.0.0',
            n_qubits=5,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=cmap,
        )

        super().__init__(_configuration)


class FakeMelbourne(FakeBackend):
    """A fake 14 qubit backend."""

    def __init__(self):
        """
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        cmap = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                [11, 3], [11, 10], [11, 12], [12, 2], [13, 1], [13, 12]]

        _configuration = BackendConfiguration(
            backend_name='fake_melbourne',
            backend_version='0.0.0',
            n_qubits=14,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=cmap,
        )

        super().__init__(_configuration)


class FakeRueschlikon(FakeBackend):
    """A fake 16 qubit backend."""

    def __init__(self):
        """
        1  -  2  -  3  -  4  -  5  -  6  -  7   -  8

        |     |     |     |     |     |     |      |

        0  -  15 -  14 -  13 -  12 -  11 -  10  -  9
        """
        cmap = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10],
                [11, 10], [12, 5], [12, 11], [12, 13], [13, 4],
                [13, 14], [15, 0], [15, 2], [15, 14]]

        _configuration = BackendConfiguration(
            backend_name='fake_rueschlikon',
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
            coupling_map=cmap,
        )

        super().__init__(_configuration)


class FakeTokyo(FakeBackend):
    """A fake 20 qubit backend."""

    def __init__(self):
        """
        0  =  1   =  2   =  3     4

        ||    ||    ||     ||  X  ||

        5  =  6   =  7   =  8  =  9

        || X  ||    ||   X  ||

        10 =  11  =  12  =  13 =  14

        ||    ||  X         || X  ||

        15 =  16  =  17     18    19
        """
        cmap = [[0, 1], [0, 5], [1, 0], [1, 2], [1, 6], [2, 1],
                [2, 3], [2, 6], [3, 2], [3, 8], [3, 9], [4, 8], [4, 9],
                [5, 0], [5, 6], [5, 10], [5, 11], [6, 1], [6, 2], [6, 5],
                [6, 7], [6, 10], [6, 11], [7, 1], [7, 6], [7, 8], [7, 12],
                [7, 13], [8, 3], [8, 4], [8, 7], [8, 9], [8, 12], [8, 13],
                [9, 3], [9, 4], [9, 8], [10, 5], [10, 6], [10, 11], [10, 15],
                [11, 5], [11, 6], [11, 10], [11, 12], [11, 16], [11, 17],
                [12, 7], [12, 8], [12, 11], [12, 13], [12, 16], [13, 7],
                [13, 8], [13, 12], [13, 14], [13, 18], [13, 19], [14, 13],
                [14, 18], [14, 19], [15, 10], [15, 16], [16, 11], [16, 12],
                [16, 15], [16, 17], [17, 11], [17, 16], [18, 13], [18, 14],
                [19, 13], [19, 14]]

        _configuration = BackendConfiguration(
            backend_name='fake_tokyo',
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
            coupling_map=cmap,
        )

        super().__init__(_configuration)


class FakeJob(BaseJob):
    """Fake simulator job"""
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, backend, job_id, fn, qobj):
        super().__init__(backend, job_id)
        self._backend = backend
        self._job_id = job_id
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
    backend = FakeQasmSimulator()
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


def _auto_progress_api(api, interval=0.2):
    """Progress a `BaseFakeAPI` instance every `interval` seconds until reaching
    the final state.
    """
    while True:
        time.sleep(interval)
        api.progress()


class BaseFakeAPI():
    """Base class for faking the IBM-Q API."""

    class NoMoreStatesError(Exception):
        """Raised when it is not possible to progress more."""

    _job_status = []

    _can_cancel = False

    def __init__(self):
        self._state = 0
        self.config = {'hub': None, 'group': None, 'project': None}
        if self._can_cancel:
            self.config.update({
                'hub': 'test-hub',
                'group': 'test-group',
                'project': 'test-project'
            })

    def get_job(self, job_id):
        if not job_id:
            return {'status': 'Error', 'error': 'Job ID not specified'}
        return self._job_status[self._state]

    def get_status_job(self, job_id):
        summary_fields = ['status', 'error', 'infoQueue']
        complete_response = self.get_job(job_id)
        return {key: value for key, value in complete_response.items()
                if key in summary_fields}

    def run_job(self, *_args, **_kwargs):
        time.sleep(0.2)
        return {'id': 'TEST_ID'}

    def cancel_job(self, job_id, *_args, **_kwargs):
        if not job_id:
            return {'status': 'Error', 'error': 'Job ID not specified'}
        return {} if self._can_cancel else {
            'error': 'testing fake API can not cancel'}

    def progress(self):
        if self._state == len(self._job_status) - 1:
            raise self.NoMoreStatesError()
        self._state += 1


class UnknownStatusAPI(BaseFakeAPI):
    """Class for emulating an API with unknown status codes."""

    _job_status = [
        {'status': 'UNKNOWN'}
    ]


class ValidatingAPI(BaseFakeAPI):
    """Class for emulating an API with job validation."""

    _job_status = [
        {'status': 'VALIDATING'},
        {'status': 'RUNNING'}
    ]


class ErrorWhileValidatingAPI(BaseFakeAPI):
    """Class for emulating an API processing an invalid job."""

    _job_status = [
        {'status': 'VALIDATING'},
        {'status': 'ERROR_VALIDATING_JOB'}
    ]


class NonQueuedAPI(BaseFakeAPI):
    """Class for emulating a successfully-completed non-queued API."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'COMPLETED', 'qasms': []}
    ]


class ErrorWhileCreatingAPI(BaseFakeAPI):
    """Class emulating an API processing a job that errors while creating
    the job.
    """

    _job_status = [
        {'status': 'ERROR_CREATING_JOB'}
    ]


class ErrorWhileRunningAPI(BaseFakeAPI):
    """Class emulating an API processing a job that errors while running."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'ERROR_RUNNING_JOB', 'error': 'Error running job'}
    ]


class QueuedAPI(BaseFakeAPI):
    """Class for emulating a successfully-completed queued API."""

    _job_status = [
        {'status': 'RUNNING', 'infoQueue': {'status': 'PENDING_IN_QUEUE'}},
        {'status': 'RUNNING'},
        {'status': 'COMPLETED'}
    ]


class RejectingJobAPI(BaseFakeAPI):
    """Class for emulating an API unable of initializing."""

    def run_job(self, *_args, **_kwargs):
        return {'error': 'invalid qobj'}


class UnavailableRunAPI(BaseFakeAPI):
    """Class for emulating an API throwing before even initializing."""

    def run_job(self, *_args, **_kwargs):
        time.sleep(0.2)
        raise ApiError()


class ThrowingAPI(BaseFakeAPI):
    """Class for emulating an API throwing in the middle of execution."""

    _job_status = [
        {'status': 'RUNNING'}
    ]

    def get_job(self, job_id):
        raise ApiError()


class ThrowingNonJobRelatedErrorAPI(BaseFakeAPI):
    """Class for emulating an scenario where the job is done but the API
    fails some times for non job-related errors.
    """

    _job_status = [
        {'status': 'COMPLETED'}
    ]

    def __init__(self, errors_before_success=2):
        super().__init__()
        self._number_of_exceptions_to_throw = errors_before_success

    def get_job(self, job_id):
        if self._number_of_exceptions_to_throw != 0:
            self._number_of_exceptions_to_throw -= 1
            raise ApiError()

        return super().get_job(job_id)


class ThrowingGetJobAPI(BaseFakeAPI):
    """Class for emulating an API throwing in the middle of execution. But not in
       get_status_job() , just in get_job().
       """

    _job_status = [
        {'status': 'COMPLETED'}
    ]

    def get_status_job(self, job_id):
        return self._job_status[self._state]

    def get_job(self, job_id):
        raise ApiError('Unexpected error')


class CancellableAPI(BaseFakeAPI):
    """Class for emulating an API with cancellation."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'CANCELLED'}
    ]

    _can_cancel = True


class NonCancellableAPI(BaseFakeAPI):
    """Class for emulating an API without cancellation running a long job."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'RUNNING'},
        {'status': 'RUNNING'}
    ]


class ErroredCancellationAPI(BaseFakeAPI):
    """Class for emulating an API with cancellation but throwing while
    trying.
    """

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'RUNNING'},
        {'status': 'RUNNING'}
    ]

    _can_cancel = True

    def cancel_job(self, job_id, *_args, **_kwargs):
        return {'status': 'Error', 'error': 'test-error-while-cancelling'}
