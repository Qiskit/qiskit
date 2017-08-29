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
import qiskit.job_processor as jobp
from qiskit.simulators import _localsimulator
from qiskit import oqc
from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience

if __name__ == '__main__':
    from _random_circuit_generator import RandomCircuitGenerator
else:
    from ._random_circuit_generator import RandomCircuitGenerator

class TestJobProcessor(unittest.TestCase):
    """
    Test job_pocessor module.
    """

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.log = logging.getLogger(__name__)
        cls.log.setLevel(logging.INFO)
        logFileName = cls.moduleName + '.log'
        handler = logging.FileHandler(logFileName)
        handler.setLevel(logging.INFO)
        log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                   ' %(message)s'.format(cls.__name__))
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        cls.log.addHandler(handler)
        
        cls.QE_TOKEN = os.environ["QE_TOKEN"]
        cls.QE_URL = os.environ["QE_URL"]
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
        compiled_circuit1 = oqc.compile(self.qc.qasm())
        compiled_circuit2 = oqc.compile(self.qasm_text)
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
        unrolled = oqc.load_unroll_qasm_file(self.qasmFileName)
        
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
        jp = jobp.JobProcessor(job_list)

    def test_run_local_simulator(self):
        compiled_circuit = oqc.compile(self.qc.qasm())
        qjob = jobp.QuantumJob(compiled_circuit, doCompile=False,
                               backend='local_qasm_simulator')
        jobp.run_local_simulator(qjob.qobj)

    def test_run_remote_simulator(self):
        compiled_circuit = oqc.compile(self.qc.qasm())
        qjob = jobp.QuantumJob(compiled_circuit, doCompile=False,
                               backend='ibmqx_qasm_simulator')
        api = IBMQuantumExperience(self.QE_TOKEN,
                                   {"url": self.QE_URL},
                                   verify=True)
        jobp.run_remote_backend(qjob.qobj, api)

    def test_run_local_simulator_compile(self):
        qjob = jobp.QuantumJob(self.qasm_text, doCompile=True,
                               backend='local_qasm_simulator')
        jobp.run_local_simulator(qjob.qobj)

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
        jp = jobp.JobProcessor(qjob)
        jp.submit(silent=True)

    def test_run_job_processor_local(self):
        njobs = 5
        job_list = []
        for i in range(njobs):
            compiled_circuit = oqc.compile(self.qc.qasm())
            qjob = jobp.QuantumJob(compiled_circuit,
                                   backend='local_qasm_simulator',
                                   doCompile=False)
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list)
        jp.submit(silent=True)

    def test_run_job_processor_online(self):
        njobs = 1
        job_list = []
        for i in range(njobs):
            compiled_circuit = oqc.compile(self.qc.qasm())            
            qjob = jobp.QuantumJob(compiled_circuit, backend='ibmqx_qasm_simulator')
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, token=self.QE_TOKEN,
                               url=self.QE_URL)
        jp.submit(silent=True)

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
        njobs = 20
        job_list = []
        for i in range(njobs):
            compiled_circuit = oqc.compile(self.qc.qasm())
            qjob = jobp.QuantumJob(compiled_circuit, backend='local_qasm_simulator')
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, max_workers=None)
        jp.submit(silent=True)
        results = jp.results()
        self.log.info(pprint.pformat(results))
        for i in range(njobs):
            self.assertTrue(results[i]['result'][0]['status'] == 'DONE')
            
        

    def test_random_local(self):
        """test randomly generated circuits on local_qasm_simulator"""
        njobs = 5
        job_list = []
        basis = 'u1,u2,u3,cx,id'
        backend = 'local_qasm_simulator'
        for circuit in self.rqg.get_circuits(format='QuantumCircuit')[:njobs]:
            compiled_circuit = oqc.compile(circuit.qasm())
            qjob = jobp.QuantumJob(compiled_circuit, backend=backend)
            job_list.append(qjob)
        jp = jobp.JobProcessor(job_list, max_workers=1)
        jp.submit(silent=True)

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
            compiled_circuit = oqc.compile(circuit.qasm())
            backend = backend_type[i % len(backend_type)]
            self.log.info(backend)
            qjob = jobp.QuantumJob(compiled_circuit, backend=backend)
            job_list.append(qjob)
            i += 1
        jp = jobp.JobProcessor(job_list, max_workers=None,
                               token=self.QE_TOKEN, url=self.QE_URL)
        jp.submit(silent=True)            
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
