# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except

"""IBMQJob Test."""

import time
import unittest
from concurrent import futures
import datetime

import numpy
from scipy.stats import chi2_contingency

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import transpiler
from qiskit.backends import JobStatus
from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.ibmq.ibmqbackend import IBMQBackendError
from qiskit.backends.ibmq.ibmqjob import IBMQJob, IBMQJobError
from .common import requires_qe_access, QiskitTestCase, slow_test


def _least_busy(backends):
    """Helper version of `wrapper.least_busy()` that works on instances.
    Args:
        backends (list[BaseBackend]): list of backends.

    Returns:
        BaseBackend: least busy backend instance.
    """
    return min([b for b in backends if
                b.status['available'] and 'pending_jobs' in b.status],
               key=lambda b: b.status['pending_jobs'])


class TestIBMQJob(QiskitTestCase):
    """
    Test ibmqjob module.
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
        cls._provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        cls._using_hub = bool(hub and group and project)

    def test_run_simulator(self):
        backend = self._provider.get_backend('ibmq_qasm_simulator')
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr, name='hadamard')
        qc.h(qr)
        qc.measure(qr, cr)
        qobj = transpiler.compile([self._qc, qc], backend)
        shots = qobj['config']['shots']
        job = backend.run(qobj)
        result = job.result()
        counts_qx1 = result.get_counts(result.get_names()[0])
        counts_qx2 = result.get_counts('hadamard')
        counts_ex1 = {'00': shots/2, '11': shots/2}
        counts_ex2 = {'00': shots/4, '11': shots/4, '10': shots/4,
                      '01': shots/4}
        states1 = counts_qx1.keys() | counts_ex1.keys()
        states2 = counts_qx2.keys() | counts_ex2.keys()
        # contingency table
        ctable1 = numpy.array([[counts_qx1.get(key, 0) for key in states1],
                               [counts_ex1.get(key, 0) for key in states1]])
        ctable2 = numpy.array([[counts_qx2.get(key, 0) for key in states2],
                               [counts_ex2.get(key, 0) for key in states2]])
        self.log.info('states1: %s', str(states1))
        self.log.info('states2: %s', str(states2))
        self.log.info('ctable1: %s', str(ctable1))
        self.log.info('ctable2: %s', str(ctable2))
        contingency1 = chi2_contingency(ctable1)
        contingency2 = chi2_contingency(ctable2)
        self.log.info('chi2_contingency1: %s', str(contingency1))
        self.log.info('chi2_contingency2: %s', str(contingency2))
        self.assertGreater(contingency1[1], 0.01)
        self.assertGreater(contingency2[1], 0.01)

    @slow_test
    def test_run_device(self):
        backends = [backend for backend in self._provider.available_backends()
                    if not backend.configuration['simulator']]
        self.log.info('devices: %s', [b.name for b in backends])
        backend = _least_busy(backends)
        self.log.info('using backend: %s', backend.name)
        qobj = transpiler.compile(self._qc, backend)
        shots = qobj['config']['shots']
        job = backend.run(qobj)
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

    @slow_test
    def test_run_async_simulator(self):
        IBMQJob._executor = futures.ThreadPoolExecutor(max_workers=2)
        backend = self._provider.get_backend('ibmq_qasm_simulator')
        self.log.info('submitting to backend %s', backend.name)
        num_qubits = 16
        qr = QuantumRegister(num_qubits, 'qr')
        cr = ClassicalRegister(num_qubits, 'cr')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = transpiler.compile([qc]*10, backend)
        num_jobs = 5
        job_array = [backend.run(qobj) for _ in range(num_jobs)]
        found_async_jobs = False
        timeout = 30
        start_time = time.time()
        while not found_async_jobs:
            check = sum([job.running for job in job_array])
            if check >= 2:
                self.log.info('found %d simultaneous jobs', check)
                break
            if all([job.done for job in job_array]):
                # done too soon? don't generate error
                self.log.warning('all jobs completed before simultaneous jobs '
                                 'could be detected')
                break
            for job in job_array:
                self.log.info('%s %s %s %s', job.status['status'], job.running,
                              check, job.id)
            self.log.info('-'*20 + ' ' + str(time.time()-start_time))
            if time.time() - start_time > timeout:
                raise TimeoutError('failed to see multiple running jobs after '
                                   '{0} s'.format(timeout))
            time.sleep(0.2)

        result_array = [job.result() for job in job_array]
        self.log.info('got back all job results')
        # Ensure all jobs have finished.
        self.assertTrue(all([job.done for job in job_array]))
        self.assertTrue(all([result.get_status() == 'COMPLETED' for result in result_array]))

        # Ensure job ids are unique.
        job_ids = [job.id for job in job_array]
        self.assertEqual(sorted(job_ids), sorted(list(set(job_ids))))

    @slow_test
    def test_run_async_device(self):
        backends = [backend for backend in self._provider.available_backends()
                    if not backend.configuration['simulator']]
        backend = _least_busy(backends)
        self.log.info('submitting to backend %s', backend.name)
        num_qubits = 5
        qr = QuantumRegister(num_qubits, 'qr')
        cr = ClassicalRegister(num_qubits, 'cr')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = transpiler.compile(qc, backend)
        num_jobs = 3
        job_array = [backend.run(qobj) for _ in range(num_jobs)]
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
        job_ids = [job.id for job in job_array]
        self.assertEqual(sorted(job_ids), sorted(list(set(job_ids))))

    def test_cancel(self):
        if not self._using_hub:
            self.skipTest('job cancellation currently only available on hubs')
        backends = [backend for backend in self._provider.available_backends()
                    if not backend.configuration['simulator']]
        self.log.info('devices: %s', [b.name for b in backends])
        backend = backends[0]
        self.log.info('using backend: %s', backend.name)
        num_qubits = 5
        qr = QuantumRegister(num_qubits, 'qr')
        cr = ClassicalRegister(num_qubits, 'cr')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = transpiler.compile(qc, backend)
        num_jobs = 3
        job_array = [backend.run(qobj) for _ in range(num_jobs)]
        success = False
        self.log.info('jobs submitted: %s', num_jobs)
        while any([job.status['status'] == JobStatus.INITIALIZING for job in job_array]):
            self.log.info('jobs initializing')
            time.sleep(1)
        for job in job_array:
            job.cancel()
        while not success:
            job_status = [job.status for job in job_array]
            for status in job_status:
                self.log.info(status)
            if any([status['status'] == JobStatus.CANCELLED for status in job_status]):
                success = True
            if all([status['status'] == JobStatus.DONE for status in job_status]):
                raise IBMQJobError('all jobs completed before any could be cancelled')
            self.log.info('-' * 20)
            time.sleep(2)
        self.assertTrue(success)

    def test_job_id(self):
        backend = self._provider.get_backend('ibmq_qasm_simulator')
        qobj = transpiler.compile(self._qc, backend)
        job = backend.run(qobj)
        self.log.info('job_id: %s', job.id)
        self.assertTrue(job.id is not None)

    def test_get_backend_name(self):
        backend_name = 'ibmq_qasm_simulator'
        backend = self._provider.get_backend(backend_name)
        qobj = transpiler.compile(self._qc, backend)
        job = backend.run(qobj)
        self.assertTrue(job.backend_name == backend_name)

    def test_get_jobs_from_backend(self):
        backend = _least_busy(self._provider.available_backends())
        start_time = time.time()
        job_list = backend.jobs(limit=5, skip=0)
        self.log.info('time to get jobs: %0.3f s', time.time() - start_time)
        self.log.info('found %s jobs on backend %s', len(job_list), backend.name)
        for job in job_list:
            self.log.info('status: %s', job.status)
            self.assertTrue(isinstance(job.id, str))
        self.log.info('time to get job statuses: %0.3f s', time.time() - start_time)

    def test_retrieve_job(self):
        backend = self._provider.get_backend('ibmq_qasm_simulator')
        qobj = transpiler.compile(self._qc, backend)
        job = backend.run(qobj)
        rjob = backend.retrieve_job(job.id)
        self.assertTrue(job.id == rjob.id)
        self.assertTrue(job.result().get_counts() == rjob.result().get_counts())

    def test_retrieve_job_error(self):
        backends = [backend for backend in self._provider.available_backends()
                    if not backend.configuration['simulator']]
        backend = _least_busy(backends)
        self.assertRaises(IBMQBackendError, backend.retrieve_job, 'BAD_JOB_ID')

    def test_get_jobs_filter_job_status(self):
        backends = self._provider.available_backends()
        backend = _least_busy(backends)
        job_list = backend.jobs(limit=5, skip=0, status=JobStatus.DONE)
        self.log.info('found %s matching jobs', len(job_list))
        for i, job in enumerate(job_list):
            self.log.info('match #%d: %s', i, job.result()._result['status'])
            self.assertTrue(job.status['status'] == JobStatus.DONE)

    def test_get_jobs_filter_counts(self):
        # TODO: consider generalizing backend name
        backend = self._provider.get_backend('ibmq_qasm_simulator')
        my_filter = {'backend.name': 'ibmq_qasm_simulator',
                     'shots': 1024,
                     'qasms.result.data.counts.00': {'lt': 500}}
        self.log.info('searching for at most 5 jobs with 1024 shots, a count '
                      'for "00" of < 500, on the ibmq_qasm_simulator backend')
        job_list = backend.jobs(limit=5, skip=0, db_filter=my_filter)
        self.log.info('found %s matching jobs', len(job_list))
        for i, job in enumerate(job_list):
            self.log.info('match #%d', i)
            result = job.result()
            self.assertTrue(any(cresult['data']['counts']['00'] < 500
                                for cresult in result._result['result']))
            for circuit_name in result.get_names():
                self.log.info('\tcircuit_name: %s', circuit_name)
                counts = result.get_counts(circuit_name)
                self.log.info('\t%s', str(counts))

    def test_get_jobs_filter_date(self):
        backends = self._provider.available_backends()
        backend = _least_busy(backends)
        past_day_30 = datetime.datetime.now() - datetime.timedelta(days=30)
        my_filter = {'creationDate': {'lt': past_day_30.isoformat()}}
        job_list = backend.jobs(limit=5, db_filter=my_filter)
        self.log.info('found %s matching jobs', len(job_list))
        for i, job in enumerate(job_list):
            self.log.info('match #%d: %s', i, job.creation_date)
            self.assertTrue(job.creation_date < past_day_30.isoformat())


if __name__ == '__main__':
    unittest.main(verbosity=2)
