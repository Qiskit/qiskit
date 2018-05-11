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

"""IBMQJob Test."""

import unittest
import time
import numpy
from scipy.stats import chi2_contingency

from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                    QuantumJob)
import qiskit._compiler
from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.basejob import JobStatus
from .common import requires_qe_access, QiskitTestCase, slow_test


def lowest_pending_jobs(backends):
    """Returns the backend with lowest pending jobs."""
    backends = filter(lambda x: x.status.get('available', False), backends)
    by_pending_jobs = sorted(backends,
                             key=lambda x: x.status['pending_jobs'])
    return by_pending_jobs[0]


class TestIBMQJob(QiskitTestCase):
    """
    Test ibmqjob module.
    """

    @classmethod
    @requires_qe_access
    def setUpClass(cls, QE_TOKEN, QE_URL):
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
        cls._provider = IBMQProvider(QE_TOKEN, QE_URL)

    def test_run_simulator(self):
        backend = self._provider.get_backend('ibmqx_qasm_simulator')
        qobj = qiskit._compiler.compile(self._qc, backend)
        shots = qobj['config']['shots']
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        job = backend.run(quantum_job)
        result = job.result()
        counts_qx = result.get_counts(result.get_names()[0])
        counts_ex = {'00': shots/2, '11': shots/2}
        states = counts_qx.keys() | counts_ex.keys()
        # contingency table
        ctable = numpy.array([[counts_qx.get(key, 0) for key in states],
                              [counts_ex.get(key, 0) for key in states]])
        self.log.info('states: %s', str(states))
        self.log.info('ctable: %s', str(ctable))
        contingency = chi2_contingency(ctable)
        self.log.info('chi2_contingency: %s', str(contingency))
        self.assertGreater(contingency[1], 0.01)

    @slow_test
    def test_run_device(self):
        backends = self._provider.available_backends({'simulator': False})
        backend = lowest_pending_jobs(backends)
        self.log.info(backend.name)
        qobj = qiskit._compiler.compile(self._qc, backend)
        shots = qobj['config']['shots']
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        job = backend.run(quantum_job)
        while not (job.done or job.exception):
            self.log.info(job.status)
            time.sleep(4)
        if job.exception:
            raise job.exception
        self.log.info(job.status)
        result = job.result()
        counts_qx = result.get_counts(result.get_names()[0])
        counts_ex = {'00': shots/2, '11': shots/2}
        states = counts_qx.keys() | counts_ex.keys()
        # contingency table
        ctable = numpy.array([[counts_qx.get(key, 0) for key in states],
                              [counts_ex.get(key, 0) for key in states]])
        self.log.info('states: %s', str(states))
        self.log.info('ctable: %s', str(ctable))
        contingency = chi2_contingency(ctable)
        self.log.info('chi2_contingency: %s', str(contingency))
        self.assertDictAlmostEqual(counts_qx, counts_ex, shots*0.1)

    def test_run_async_simulator(self):
        backend = self._provider.get_backend('ibmqx_qasm_simulator')
        self.log.info('submitting to backend %s', backend.name)
        num_qubits = 5
        qr = QuantumRegister(num_qubits, 'qr')
        cr = ClassicalRegister(num_qubits, 'cr')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile([qc]*10, backend)
        quantum_job = QuantumJob(qobj, backend, preformatted=True)
        num_jobs = 5
        job_array = [backend.run(quantum_job) for _ in range(num_jobs)]
        time.sleep(3)  # give time for jobs to start (better way?)
        job_status = [job.status['status'] for job in job_array]
        num_init = sum([status == JobStatus.INITIALIZING for status in job_status])
        num_queued = sum([status == JobStatus.QUEUED for status in job_status])
        num_running = sum([status == JobStatus.RUNNING for status in job_status])
        num_done = sum([status == JobStatus.DONE for status in job_status])
        num_error = sum([status == JobStatus.ERROR for status in job_status])
        self.log.info('number of currently initializing jobs: %d/%d',
                      num_init, num_jobs)
        self.log.info('number of currently queued jobs: %d/%d',
                      num_queued, num_jobs)
        self.log.info('number of currently running jobs: %d/%d',
                      num_running, num_jobs)
        self.log.info('number of currently done jobs: %d/%d',
                      num_done, num_jobs)
        self.log.info('number of errored jobs: %d/%d',
                      num_error, num_jobs)
        self.assertTrue(num_jobs - num_error - num_done > 0)

        # Wait for all the results.
        result_array = [job.result() for job in job_array]

        # Ensure all jobs have finished.
        self.assertTrue(all([job.done for job in job_array]))
        self.assertTrue(all([result.get_status() == 'COMPLETED' for result in result_array]))

        # Ensure job ids are unique.
        job_ids = [job.job_id for job in job_array]
        self.assertEqual(sorted(job_ids), sorted(list(set(job_ids))))

    @slow_test
    def test_run_async_device(self):
        backends = self._provider.available_backends({'simulator': False})
        backend = lowest_pending_jobs(backends)
        self.log.info('submitting to backend %s', backend.name)
        num_qubits = 5
        qr = QuantumRegister(num_qubits, 'qr')
        cr = ClassicalRegister(num_qubits, 'cr')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile(qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1e5, preformatted=True)
        num_jobs = 3
        job_array = [backend.run(quantum_job) for _ in range(num_jobs)]
        time.sleep(3)  # give time for jobs to start (better way?)
        job_status = [job.status['status'] for job in job_array]
        num_init = sum([status == JobStatus.INITIALIZING for status in job_status])
        num_queued = sum([status == JobStatus.QUEUED for status in job_status])
        num_running = sum([status == JobStatus.RUNNING for status in job_status])
        num_done = sum([status == JobStatus.DONE for status in job_status])
        num_error = sum([status == JobStatus.ERROR for status in job_status])
        self.log.info('number of currently initializing jobs: %d/%d',
                      num_init, num_jobs)
        self.log.info('number of currently queued jobs: %d/%d',
                      num_queued, num_jobs)
        self.log.info('number of currently running jobs: %d/%d',
                      num_running, num_jobs)
        self.log.info('number of currently done jobs: %d/%d',
                      num_done, num_jobs)
        self.log.info('number of errored jobs: %d/%d',
                      num_error, num_jobs)
        self.assertTrue(num_jobs - num_error - num_done > 0)

        # Wait for all the results.
        result_array = [job.result() for job in job_array]

        # Ensure all jobs have finished.
        self.assertTrue(all([job.done for job in job_array]))
        self.assertTrue(all([result.get_status() == 'COMPLETED' for result in result_array]))

        # Ensure job ids are unique.
        job_ids = [job.job_id for job in job_array]
        self.assertEqual(sorted(job_ids), sorted(list(set(job_ids))))

    @unittest.skip('cancel is not currently possible on IBM Q')
    def test_cancel(self):
        backend = self._provider.get_backend('ibmqx4')
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1024, preformatted=True)
        job = backend.run(quantum_job)
        job.cancel()
        self.assertTrue(job.cancelled)

    def test_job_id(self):
        backend = self._provider.get_backend('ibmqx_qasm_simulator')
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1024, preformatted=True)
        job = backend.run(quantum_job)
        while job.status['status'] == JobStatus.INITIALIZING:
            time.sleep(0.1)
        self.log.info('job_id: %s', job.job_id)
        self.assertTrue(job.job_id is not None)

    def test_get_backend_name(self):
        backend_name = 'ibmqx_qasm_simulator'
        backend = self._provider.get_backend(backend_name)
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1024, preformatted=True)
        job = backend.run(quantum_job)
        self.assertTrue(job.backend_name == backend_name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
