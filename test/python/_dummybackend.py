# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
Dummy backend simulator.
The purpose of this class is to create a Simulator that we can trick for testing
purposes. Testing local timeouts, arbitrary responses or behavior, etc.
"""

import uuid
import logging
from threading import Event
from concurrent import futures

from qiskit import Result, QISKitError
from qiskit.backends import BaseBackend
from qiskit.backends import BaseJob
from qiskit.backends.baseprovider import BaseProvider

logger = logging.getLogger(__name__)


class DummyProvider(BaseProvider):
    """Dummy provider just for testing purposes."""
    def get_backend(self, name):
        return DummySimulator()

    def available_backends(self, *args, **kwargs):
        return ['local_dummy_simulator']


class DummySimulator(BaseBackend):
    """ This is Dummy backend simulator just for testing purposes """
    def __init__(self, configuration=None, time_alive=10):
        super().__init__(configuration)
        self.time_alive = time_alive

        if configuration is None:
            self._configuration = {'name': 'local_dummy_simulator',
                                   'url': 'https://github.com/IBM/qiskit-sdk-py',
                                   'simulator': True,
                                   'local': True,
                                   'description': 'A dummy simulator for testing purposes',
                                   'coupling_map': 'all-to-all',
                                   'basis_gates': 'u1,u2,u3,cx,id'}
        else:
            self._configuration = configuration

    def run(self, q_job):
        return DummyJob(self.run_job, q_job)
    
    def run_job(self, q_job):
        """ Main dummy simulator loop """
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj
        timeout = q_job.timeout
        wait_time = q_job.wait

        time_passed = 0
        while time_passed <= self.time_alive:
            Event().wait(timeout=wait_time)
            time_passed += wait_time
            if time_passed >= timeout:
                raise QISKitError('Dummy backend has timed out!')

        return Result({'job_id': job_id, 'result': [], 'status': 'COMPLETED'}, qobj)

class DummyJob(BaseJob):
    """dummy simulator job"""
    _executor = futures.ProcessPoolExecutor()
    
    def __init__(self, fn, qobj):
        self._qobj = qobj
        self._future = self._executor.submit(fn, qobj)

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def cancelled(self):
        return self._future.cancelled()

    def done(self):
        return self._future.done()

    def status(self):
        if self.running():
            return "running"
        elif self.cancelled():
            return "cancelled"
        elif self.done():
            return "done"
        else:
            return "unknown"

    def cancel(self):
        return self._future.cancel()

    def running(self):
        return self._future.running()

    def add_done_callback(self, fn):
        self._future.add_done_callback(fn)
    
