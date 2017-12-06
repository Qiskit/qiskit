# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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

import os
import pprint
import unittest

from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumProgram,
                    QuantumRegister, QISKitError)
from qiskit import _openquantumcompiler as openquantumcompiler
import qiskit._jobprocessor as jobprocessor
import qiskit.backends
from qiskit import QuantumJob

from ._random_circuit_generator import RandomCircuitGenerator
from .common import QiskitTestCase, TRAVIS_FORK_PULL_REQUEST


def mock_run_local_backend(self):
    raise Exception("Mocking job error!!")


class TestJobProcessor(QiskitTestCase):
    """
    Test job_pocessor module.
    """

    @classmethod
    def setUpClass(cls):
        super(TestJobProcessor, cls).setUpClass()

        try:
            import Qconfig
            cls.QE_TOKEN = Qconfig.APItoken
            cls.QE_URL = Qconfig.config['url']
        except ImportError:
            if 'QE_TOKEN' in os.environ:
                cls.QE_TOKEN = os.environ['QE_TOKEN']
            if 'QE_URL' in os.environ:
                cls.QE_URL = os.environ['QE_URL']

        nCircuits = 20
        minDepth = 1
        maxDepth = 40
        minQubits = 1
        maxQubits = 5
        randomCircuits = RandomCircuitGenerator(100,
                                                minQubits=minQubits,
                                                maxQubits=maxQubits,
                                                minDepth=minDepth,
                                                maxDepth=maxDepth)
        randomCircuits.add_circuits(nCircuits)
        cls.rqg = randomCircuits
        if hasattr(cls, 'QE_TOKEN'):
            cls._api = IBMQuantumExperience(cls.QE_TOKEN,
                                            {"url": cls.QE_URL},
                                            verify=True)
            qiskit.backends.discover_remote_backends(cls._api)


    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.seed = 88
        self.qasmFileName = self._get_resource_path('qasm/example.qasm')
        with open(self.qasmFileName, 'r') as qasm_file:
            self.qasm_text = qasm_file.read()
        # create QuantumCircuit
        qr = QuantumRegister('q', 2)
        cr = ClassicalRegister('c', 2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qc = qc
        # create qobj
        compiled_circuit1 = openquantumcompiler.compile(self.qc.qasm())
        compiled_circuit2 = openquantumcompiler.compile(self.qasm_text)
        self.qobj = {'id': 'test_qobj',
                     'config': {
                         'max_credits': 3,
                         'shots': 100,
                         'backend': 'local_qasm_simulator',
                     },
                     'circuits': [
                         {
                             'name': 'test_circuit1',
                             'compiled_circuit': compiled_circuit1,
                             'basis_gates': 'u1,u2,u3,cx,id',
                             'layout': None,
                             'seed': None
                         },
                         {
                             'name': 'test_circuit2',
                             'compiled_circuit': compiled_circuit2,
                             'basis_gates': 'u1,u2,u3,cx,id',
                             'layout': None,
                             'seed': None
                         }
                     ]
                     }

    def tearDown(self):
        pass

    def test_load_unroll_qasm_file(self):
        unrolled = openquantumcompiler.load_unroll_qasm_file(self.qasmFileName)

    def test_init_quantum_job(self):
        quantum_job = QuantumJob(self.qc)

    def test_init_quantum_job_qobj(self):
        formatted_circuit = self.qasm_text
        qobj = {'id': 'qobj_init',
                'config': {
                    'max_credits': 3,
                    'shots': 1024,
                    'seed': None,
                    'backend': 'local_qasm_simulator'},
                'circuits': [
                    {'name': 'example',
                     'compiled_circuit': formatted_circuit,
                     'layout': None,
                     'seed': None}
                ]
               }
        quantum_job = QuantumJob(qobj, preformatted=True)

    def test_init_job_processor(self):
        njobs = 5
        job_list = []
        for i in range(njobs):
            quantum_job = QuantumJob(self.qc, do_compile=False)
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, callback=None)

    def test_run_local_backend_qasm(self):
        compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
        quantum_job = QuantumJob(compiled_circuit, do_compile=False,
                               backend='local_qasm_simulator')
        jobprocessor.run_backend(quantum_job)

    def test_run_local_backend_unitary(self):
        compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
        quantum_job = QuantumJob(compiled_circuit, do_compile=False,
                               backend='local_unitary_simulator')
        jobprocessor.run_backend(quantum_job)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_run_remote_simulator(self):
        compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
        quantum_job = QuantumJob(compiled_circuit, do_compile=False,
                                 backend='ibmqx_qasm_simulator')
        jobprocessor.run_backend(quantum_job)

    def test_run_local_backend_compile(self):
        quantum_job = QuantumJob(self.qasm_text, do_compile=True,
                               backend='local_qasm_simulator')
        jobprocessor.run_backend(quantum_job)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_run_remote_simulator_compile(self):
        quantum_job = QuantumJob(self.qc, do_compile=True,
                                 backend='ibmqx_qasm_simulator')
        jobprocessor.run_backend(quantum_job)

    def test_compile_job(self):
        """Test compilation as part of job"""
        quantum_job = QuantumJob(self.qasm_text, do_compile=True,
                                 backend='local_qasm_simulator')
        jp = jobprocessor.JobProcessor([quantum_job], callback=None)
        jp.submit()

    def test_run_job_processor_local(self):
        njobs = 5
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            quantum_job = QuantumJob(compiled_circuit,
                                     backend='local_qasm_simulator',
                                     do_compile=False)
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, callback=None)
        jp.submit()

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_run_job_processor_online(self):
        njobs = 1
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            quantum_job = QuantumJob(compiled_circuit,
                                     backend='ibmqx_qasm_simulator')
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, callback=None)
        jp.submit()

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_quantum_program_online(self):
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit('qc', [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        backend = 'ibmqx_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        qp.set_api(self.QE_TOKEN, self.QE_URL)
        result = qp.execute(['qc'], backend=backend, shots=shots,
                            seed=78)

    def test_run_job_processor_local_parallel(self):
        def job_done_callback(results):
            try:
                self.log.info(pprint.pformat(results))
                for result in results:
                    self.assertTrue(result.get_status() == 'COMPLETED')
            except Exception as e:
                self.job_processor_exception = e
            finally:
                self.job_processor_finished = True

        njobs = 20
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            quantum_job = QuantumJob(compiled_circuit,
                                     backend='local_qasm_simulator')
            job_list.append(quantum_job)

        self.job_processor_finished = False
        self.job_processor_exception = None
        jp = jobprocessor.JobProcessor(job_list, max_workers=None,
                                       callback=job_done_callback)
        jp.submit()

        while not self.job_processor_finished:
            # Wait until the job_done_callback is invoked and completed.
            pass

        if self.job_processor_exception:
            raise self.job_processor_exception

    def test_random_local(self):
        """test randomly generated circuits on local_qasm_simulator"""
        njobs = 5
        job_list = []
        basis = 'u1,u2,u3,cx,id'
        backend = 'local_qasm_simulator'
        for circuit in self.rqg.get_circuits(format='QuantumCircuit')[:njobs]:
            compiled_circuit = openquantumcompiler.compile(circuit.qasm())
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend)
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, max_workers=1, callback=None)
        jp.submit()

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_mix_local_remote_jobs(self):
        """test mixing local and remote jobs

        Internally local jobs execute in seperate processes since
        they are CPU bound and remote jobs execute in seperate threads
        since they are I/O bound. The module gets results from potentially
        both kinds in one list. Test that this works.
        """
        njobs = 6
        job_list = []
        basis = 'u1,u2,u3,cx'
        backend_type = ['local_qasm_simulator', 'ibmqx_qasm_simulator']
        i = 0
        for circuit in self.rqg.get_circuits(format='QuantumCircuit')[:njobs]:
            compiled_circuit = openquantumcompiler.compile(circuit.qasm())
            backend = backend_type[i % len(backend_type)]
            self.log.info(backend)
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend)
            job_list.append(quantum_job)
            i += 1
        jp = jobprocessor.JobProcessor(job_list, max_workers=None,
                               callback=None)
        jp.submit()

    def test_error_in_job(self):
        def job_done_callback(results):
            try:
                for result in results:
                    self.log.info(pprint.pformat(result))
                    self.assertTrue(result.get_status() == 'ERROR')
            except Exception as e:
                self.job_processor_exception = e
            finally:
                self.job_processor_finished = True

        njobs = 5
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            quantum_job = QuantumJob(compiled_circuit,
                                     backend='local_qasm_simulator')
            job_list.append(quantum_job)

        self.job_processor_finished = False
        self.job_processor_exception = None
        jp = jobprocessor.JobProcessor(job_list, max_workers=None,
                                       callback=job_done_callback)
        tmp = jobprocessor.run_backend
        jobprocessor.run_backend = mock_run_local_backend

        jp.submit()
        jobprocessor.run_backend = tmp

        while not self.job_processor_finished:
            # Wait until the job_done_callback is invoked and completed.
            pass

        if self.job_processor_exception:
            raise self.job_processor_exception

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_backend_not_found(self):
        compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
        job = QuantumJob(compiled_circuit,
                         backend='non_existing_backend')
        self.assertRaises(QISKitError, jobprocessor.JobProcessor, [job],
                          callback=None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
