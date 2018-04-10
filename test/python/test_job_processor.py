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

import pprint
import unittest

import qiskit
import qiskit._jobprocessor as jobprocessor
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumProgram,
                    QuantumRegister)
from qiskit import QuantumJob
from qiskit._compiler import (compile_circuit, load_unroll_qasm_file)
from qiskit.backends._qasmsimulator import QasmSimulator
from qiskit.backends._unitarysimulator import UnitarySimulator
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider
from ._random_circuit_generator import RandomCircuitGenerator
from .common import requires_qe_access, QiskitTestCase


def mock_run_local_backend(self):
    # pylint: disable=unused-argument
    raise Exception("Mocking job error!!")


class TestJobProcessor(QiskitTestCase):
    """
    Test job_processor module.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        n_circuits = 20
        min_depth = 1
        max_depth = 40
        min_qubits = 1
        max_qubits = 5
        random_circuits = RandomCircuitGenerator(100,
                                                 min_qubits=min_qubits,
                                                 max_qubits=max_qubits,
                                                 min_depth=min_depth,
                                                 max_depth=max_depth)
        random_circuits.add_circuits(n_circuits)
        cls.rqg = random_circuits

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/example.qasm')
        with open(self.qasm_filename, 'r') as qasm_file:
            self.qasm_text = qasm_file.read()
            self.qasm_ast = qiskit.qasm.Qasm(data=self.qasm_text).parse()
            self.qasm_be = qiskit.unroll.CircuitBackend(['u1', 'u2', 'u3', 'id', 'cx'])
            self.qasm_circ = qiskit.unroll.Unroller(self.qasm_ast, self.qasm_be).execute()
        # create QuantumCircuit
        qr = QuantumRegister('q', 2)
        cr = ClassicalRegister('c', 2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qc = qc
        # create qobj
        compiled_circuit1 = compile_circuit(self.qc)
        compiled_circuit2 = compile_circuit(self.qasm_circ)
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
                     ]}
        self.job_processor_exception = Exception()
        self.job_processor_finished = False

    def test_load_unroll_qasm_file(self):
        _ = load_unroll_qasm_file(self.qasm_filename)

    def test_init_quantum_job(self):
        backend = QasmSimulator()
        _ = QuantumJob(self.qc, backend)

    def test_init_quantum_job_qobj(self):
        formatted_circuit = self.qasm_text
        backend = QasmSimulator()
        qobj = {'id': 'qobj_init',
                'config': {
                    'max_credits': 3,
                    'shots': 1024,
                    'seed': None,
                    'backend': backend},
                'circuits': [
                    {'name': 'example',
                     'compiled_circuit': formatted_circuit,
                     'layout': None,
                     'seed': None}
                ]}
        _ = QuantumJob(qobj, preformatted=True)

    def test_init_job_processor(self):
        njobs = 5
        job_list = []
        backend = QasmSimulator()
        for _ in range(njobs):
            quantum_job = QuantumJob(self.qc, backend, do_compile=False)
            job_list.append(quantum_job)
        _ = jobprocessor.JobProcessor(job_list, callback=None)

    def test_run_local_backend_qasm(self):
        backend = QasmSimulator()
        dag_circuit = compile_circuit(self.qc)
        quantum_job = QuantumJob(dag_circuit, do_compile=False,
                                 backend=backend)
        jobprocessor.run_backend(quantum_job)

    def test_run_local_backend_unitary(self):
        backend = UnitarySimulator()
        compiled_circuit = compile_circuit(self.qc)
        quantum_job = QuantumJob(compiled_circuit, do_compile=False,
                                 backend=backend)
        jobprocessor.run_backend(quantum_job)

    @requires_qe_access
    def test_run_remote_simulator(self, QE_TOKEN, QE_URL):
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.get_backend('ibmqx_qasm_simulator')

        compiled_circuit = compile_circuit(self.qc)
        quantum_job = QuantumJob(compiled_circuit, do_compile=False,
                                 backend=backend)
        jobprocessor.run_backend(quantum_job)

    def test_run_local_backend_compile(self):
        backend = QasmSimulator()
        quantum_job = QuantumJob(self.qasm_circ, do_compile=True,
                                 backend=backend)
        jobprocessor.run_backend(quantum_job)

    @requires_qe_access
    def test_run_remote_simulator_compile(self, QE_TOKEN, QE_URL):
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.get_backend('ibmqx_qasm_simulator')

        quantum_job = QuantumJob(self.qc, do_compile=True,
                                 backend=backend)
        jobprocessor.run_backend(quantum_job)

    def test_compile_job(self):
        """Test compilation as part of job"""
        backend = QasmSimulator()
        quantum_job = QuantumJob(self.qasm_circ, do_compile=True,
                                 backend=backend)
        jp = jobprocessor.JobProcessor([quantum_job], callback=None)
        jp.submit()

    def test_run_job_processor_local(self):
        njobs = 5
        job_list = []
        backend = QasmSimulator()
        for _ in range(njobs):
            compiled_circuit = compile_circuit(self.qc)
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend,
                                     do_compile=False)
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, callback=None)
        jp.submit()

    @requires_qe_access
    def test_run_job_processor_online(self, QE_TOKEN, QE_URL):
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        backend = provider.get_backend('ibmqx_qasm_simulator')

        njobs = 1
        job_list = []
        for _ in range(njobs):
            compiled_circuit = compile_circuit(self.qc)
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend)
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, callback=None)
        jp.submit()

    @requires_qe_access
    def test_quantum_program_online(self, QE_TOKEN, QE_URL):
        qp = QuantumProgram()
        qp.set_api(QE_TOKEN, QE_URL)
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit('qc', [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        backend = 'ibmqx_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        _ = qp.execute(['qc'], backend=backend, shots=shots, seed=78)

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
        backend = QasmSimulator()
        for _ in range(njobs):
            compiled_circuit = compile_circuit(self.qc)
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend)
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
        backend = QasmSimulator()
        for circuit in self.rqg.get_circuits(format_='QuantumCircuit')[:njobs]:
            compiled_circuit = compile_circuit(circuit)
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend)
            job_list.append(quantum_job)
        jp = jobprocessor.JobProcessor(job_list, max_workers=1, callback=None)
        jp.submit()

    @requires_qe_access
    def test_mix_local_remote_jobs(self, QE_TOKEN, QE_URL):
        """test mixing local and remote jobs

        Internally local jobs execute in seperate processes since
        they are CPU bound and remote jobs execute in seperate threads
        since they are I/O bound. The module gets results from potentially
        both kinds in one list. Test that this works.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL)
        remote_backend = provider.get_backend('ibmqx_qasm_simulator')
        local_backend = QasmSimulator()

        njobs = 6
        job_list = []

        backend_type = [local_backend, remote_backend]
        i = 0
        for circuit in self.rqg.get_circuits(format_='QuantumCircuit')[:njobs]:
            compiled_circuit = compile_circuit(circuit)
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
        backend = QasmSimulator()
        for _ in range(njobs):
            compiled_circuit = compile_circuit(self.qc)
            quantum_job = QuantumJob(compiled_circuit,
                                     backend=backend)
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
