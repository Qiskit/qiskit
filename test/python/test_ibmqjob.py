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
from .common import requires_qe_access, QiskitTestCase


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

    def test_run(self):
        backend = self._provider.get_backend('ibmqx_qasm_simulator')
        qobj = qiskit._compiler.compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1024, preformatted=True)
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

    def test_run_async(self):
        #backend = self._provider.get_backend('ibmqx_qasm_simulator')
        backend = self._provider.get_backend('ibmqx4')
        num_qubits = 5
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        for i in range(num_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile(qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1e5, preformatted=True)
        num_jobs = 3
        job_array = [backend.run(quantum_job) for _ in range(num_jobs)]
        time.sleep(0.1)  # give time for jobs to start (better way?)
        num_queued = sum([job.queued for job in job_array])
        num_running = sum([job.running for job in job_array])
        self.log.info('number of currently queued jobs: %d/%d' % (
            num_queued, num_jobs))
        self.log.info('number of currently running jobs: %d/%d' % (
            num_running, num_jobs))
        self.assertTrue(all([(job.running or job.queued) for job in job_array]))

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
        self.log.info('job_id: %s' % job.job_id)
        self.assertTrue(job.job_id is not None)

if __name__ == '__main__':
    unittest.main(verbosity=2)
