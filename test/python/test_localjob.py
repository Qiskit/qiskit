# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,broad-except

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

"""LocalJob Test."""

import sys
import unittest
from concurrent import futures
import time

import numpy
from scipy.stats import chi2_contingency

import qiskit._compiler
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                    QuantumJob)
from qiskit.backends.local import LocalProvider, LocalJob
from .common import requires_qe_access, QiskitTestCase, slow_test


class TestLocalJob(QiskitTestCase):
    """
    Test localjob module.
    """

    @classmethod
    @requires_qe_access
    def setUpClass(cls, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        # pylint: disable=arguments-differ
        super().setUpClass()
        # create QuantumCircuit
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)
        cls._qc = qc
        cls._provider = LocalProvider(QE_TOKEN, QE_URL, hub, group, project)

    def setUp(self):
        # Store the original executor, for restoring later.
        self._original_executor = LocalJob._executor

    def tearDown(self):
        # Restore the original executor.
        LocalJob._executor = self._original_executor

    def test_run(self):
        backend = self._provider.get_backend('local_qasm_simulator_py')
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        job = backend.run(quantum_job)
        result = job.result()
        counts_qx = result.get_counts(result.get_names()[0])
        counts_ex = {'00': 512, '11': 512}
        states = counts_qx.keys() | counts_ex.keys()
        # contingency table
        ctable = numpy.array([[counts_qx.get(key, 0) for key in states],
                              [counts_ex.get(key, 0) for key in states]])
        contingency = chi2_contingency(ctable)
        self.log.info('chi2_contingency: %s', str(contingency))
        self.assertGreater(contingency[1], 0.01)

    @slow_test
    def test_run_async(self):
        if sys.platform == 'darwin':
            LocalJob._executor = futures.ThreadPoolExecutor(max_workers=2)
        else:
            LocalJob._executor = futures.ProcessPoolExecutor(max_workers=2)
        try:
            backend = self._provider.get_backend('local_qasm_simulator_cpp')
        except KeyError:
            backend = self._provider.get_backend('local_qasm_simulator_py')
        num_qubits = 15
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile(qc, backend)
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        num_jobs = 5
        job_array = [backend.run(quantum_job) for _ in range(num_jobs)]
        found_async_jobs = False
        timeout = 30
        start_time = time.time()
        self.log.info('testing with simulator: %s', backend.name)
        while not found_async_jobs:
            check = sum([job.running for job in job_array])
            if check >= 2:
                self.log.info('found %d simultaneous jobs', check)
                found_async_jobs = True
            if all([job.done for job in job_array]):
                self.log.warning('all jobs completed before simultaneous jobs '
                                 'could be detected')
                break
            for job in job_array:
                self.log.info('%s %s %s', job.status['status'],
                              job.running, check)
            self.log.info('%s %.4f', '-'*20, time.time()-start_time)
            if time.time() - start_time > timeout:
                raise TimeoutError('failed to see multiple running jobs after '
                                   '{0} s'.format(timeout))
            time.sleep(1)

        # Wait for all the jobs to finish.
        # TODO: this causes the test to wait until the 15 qubit jobs are
        # finished, which might take long (hence the @slow_test). Waiting for
        # the result is needed as otherwise the jobs would still be running
        # once the test is completed, causing failures in subsequent tests as
        # the executor's queue might be overloaded.

        _ = [job.result() for job in job_array]

    def test_cancel(self):
        """Test the cancelation of jobs.

        Since only Jobs that are still in the executor queue pending to be
        executed can be cancelled, this test launches a lot of jobs, passing
        if some of them can be cancelled.
        """
        # Force the number of workers to 1, as only Jobs that are still in
        # the executor queue can be canceled.
        if sys.platform == 'darwin':
            LocalJob._executor = futures.ThreadPoolExecutor(max_workers=1)
        else:
            LocalJob._executor = futures.ProcessPoolExecutor(max_workers=1)

        backend = self._provider.get_backend('local_qasm_simulator_py')
        num_qubits = 5
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile(qc, backend)
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        num_jobs = 10
        timeout = 10
        start_time = time.time()
        self.log.info('testing with simulator: %s', backend.name)
        job_array = [backend.run(quantum_job) for _ in range(num_jobs)]
        for job in job_array:
            job.cancel()
        found_cancelled = False
        while not found_cancelled:
            check = sum([job.cancelled for job in job_array])
            if check >= 1:
                self.log.info('found %d cancelled jobs', check)
                found_cancelled = True
            if all([job.done for job in job_array]):
                self.log.warning('all jobs completed before simultaneous jobs '
                                 'could be detected')
                break
            for job in job_array:
                self.log.info('%s %s %s', job.status['status'], job.cancelled,
                              check)
            self.log.info('{0} {1:0.2f}'.format('-'*20, time.time()-start_time))
            if time.time() - start_time > timeout:
                raise TimeoutError('failed to see multiple running jobs after '
                                   '{0} s'.format(timeout))
            time.sleep(1)

        # Wait for all the jobs to finish.
        _ = [job.result() for job in job_array if not job.cancelled]

    def test_done(self):
        backend = self._provider.get_backend('local_qasm_simulator_py')
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        job = backend.run(quantum_job)
        job.result()
        self.assertTrue(job.done)

    def test_get_backend_name(self):
        backend_name = 'local_qasm_simulator_py'
        backend = self._provider.get_backend(backend_name)
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        job = backend.run(quantum_job)
        self.assertTrue(job.backend_name == backend_name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
