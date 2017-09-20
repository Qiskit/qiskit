import unittest
import time
import os
import sys
import io
import logging
import random
import pprint
import qiskit
from qiskit import QuantumProgram
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
import qiskit.qasm as qasm
import qiskit.unroll as unroll
import qiskit._jobprocessor as jobp
from qiskit.simulators import _localsimulator
from qiskit import _openquantumcompiler as openquantumcompiler
from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience

from .common import QiskitTestCase, TRAVIS_FORK_PULL_REQUEST

if __name__ == '__main__':
    from _random_circuit_generator import RandomCircuitGenerator
else:
    from ._random_circuit_generator import RandomCircuitGenerator


def mock_run_local_simulator(self):
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

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.seed = 88
        self.qasmFileName = os.path.join(qiskit.__path__[0],
                                         '../test/python/qasm/example.qasm')
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
        qjob = jobp.QuantumJob(self.qc)

    def test_init_quantum_job_qobj(self):
        formatted_circuit = self.qasm_text
        qobj = {'id': 'qobj_init',
                'config': {
                    'max_credits': 3,
                    'shots': 1024,
                    'backend': 'local_qasm_simulator'},
                'circuits': [
                    {'name': 'example',
                     'compiled_circuit': formatted_circuit,
                     'layout': None,
                     'seed': None}
                ]
               }
        qjob = jobp.QuantumJob(qobj, preformatted=True)

    def test_init_job_processor(self):
        njobs = 5
        job_list = []
        for i in range(njobs):
            qjob = jobp.QuantumJob(self.qc, doCompile=False)
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, callback=None)

    def testrun_local_simulator(self):
        compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
        qjob = jobp.QuantumJob(compiled_circuit, doCompile=False,
                               backend='local_qasm_simulator')
        jobp.run_local_simulator(qjob.qobj)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_run_remote_simulator(self):
        compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
        qjob = jobp.QuantumJob(compiled_circuit, doCompile=False,
                               backend='ibmqx_qasm_simulator')
        api = IBMQuantumExperience(self.QE_TOKEN,
                                   {"url": self.QE_URL},
                                   verify=True)
        jobp.run_remote_backend(qjob.qobj, api)

    def testrun_local_simulator_compile(self):
        qjob = jobp.QuantumJob(self.qasm_text, doCompile=True,
                               backend='local_qasm_simulator')
        jobp.run_local_simulator(qjob.qobj)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_run_remote_simulator_compile(self):
        qjob = jobp.QuantumJob(self.qc, doCompile=True,
                               backend='ibmqx_qasm_simulator')
        api = IBMQuantumExperience(self.QE_TOKEN,
                                   {"url": self.QE_URL},
                                   verify=True)
        jobp.run_remote_backend(qjob.qobj, api)

    def test_compile_job(self):
        """Test compilation as part of job"""
        qjob = jobp.QuantumJob(self.qasm_text, doCompile=True,
                               backend='local_qasm_simulator')
        jp = jobp.JobProcessor([qjob], callback=None)
        jp.submit(silent=True)

    def test_run_job_processor_local(self):
        njobs = 5
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            qjob = jobp.QuantumJob(compiled_circuit,
                                   backend='local_qasm_simulator',
                                   doCompile=False)
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, callback=None)
        jp.submit(silent=True)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_run_job_processor_online(self):
        njobs = 1
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            qjob = jobp.QuantumJob(compiled_circuit, backend='ibmqx_qasm_simulator')
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, token=self.QE_TOKEN,
                               url=self.QE_URL,
                               callback=None)
        jp.submit(silent=True)

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
                for result, _ in results:
                    self.assertTrue(result['result'][0]['status'] == 'DONE')
            except Exception as e:
                self.job_processor_exception = e
            finally:
                self.job_processor_finished = True

        njobs = 20
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            qjob = jobp.QuantumJob(compiled_circuit, backend='local_qasm_simulator')
            job_list.append(qjob)

        self.job_processor_finished = False
        self.job_processor_exception = None
        jp = jobp.JobProcessor(job_list, max_workers=None,
                               callback=job_done_callback)
        jp.submit(silent=True)

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
            qjob = jobp.QuantumJob(compiled_circuit, backend=backend)
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, max_workers=1, callback=None)
        jp.submit(silent=True)

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
            qjob = jobp.QuantumJob(compiled_circuit, backend=backend)
            job_list.append(qjob)
            i += 1
        jp = jobp.JobProcessor(job_list, max_workers=None,
                               token=self.QE_TOKEN, url=self.QE_URL,
                               callback=None)
        jp.submit(silent=True)

    def test_error_in_job(self):
        def job_done_callback(results):
            try:
                for result, _ in results:
                    self.log.info(pprint.pformat(result))
                    self.assertTrue(result['status'] == 'ERROR')
            except Exception as e:
                self.job_processor_exception = e
            finally:
                self.job_processor_finished = True

        njobs = 5
        job_list = []
        for i in range(njobs):
            compiled_circuit = openquantumcompiler.compile(self.qc.qasm())
            qjob = jobp.QuantumJob(compiled_circuit, backend='local_qasm_simulator')
            job_list.append(qjob)

        jp = jobp.JobProcessor(job_list, max_workers=None,
                               callback=job_done_callback)

        self.job_processor_finished = False
        self.job_processor_exception = None
        tmp = jobp.run_local_simulator
        jobp.run_local_simulator = mock_run_local_simulator
        jp.submit(silent=True)
        jobp.run_local_simulator = tmp

        while not self.job_processor_finished:
            # Wait until the job_done_callback is invoked and completed.
            pass

        if self.job_processor_exception:
            raise self.job_processor_exception


if __name__ == '__main__':
    unittest.main(verbosity=2)
