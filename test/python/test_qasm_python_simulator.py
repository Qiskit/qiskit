#!/usr/bin/env python
import unittest
import time
import numpy as np
import os
import sys
import cProfile
import pstats
import io
import logging
import random
import string
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
try:
    import qiskit
except ImportError as ierr:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    import qiskit
from qiskit import QuantumProgram
from qiskit.simulators._qasmsimulator import QasmSimulator
import qiskit.qasm as qasm
import qiskit.unroll as unroll
if __name__ == '__main__':
    from _random_qasm_generator import RandomQasmGenerator
else:
    from test.python._random_qasm_generator import RandomQasmGenerator

class LocalQasmSimulatorTest(unittest.TestCase):
    """Test local qasm simulator."""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.pdf = PdfPages(cls.moduleName + '.pdf')
        cls.logFileName = cls.moduleName + '.log'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO)

    @classmethod        
    def tearDownClass(cls):
        cls.pdf.close()
    
    def setUp(self):
        self.seed = 88
        self.qasmFileName = os.path.join(qiskit.__path__[0],
                                         '../test/python/qasm/example.qasm')
        self.qp = QuantumProgram()
    
    def tearDown(self):
        pass

    def test_qasm_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        self.qp.load_qasm('example', qasm_file=self.qasmFileName)
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm("example")).parse(),
                      unroll.JsonBackend(basis_gates))
        unroller.execute()
        circuit = unroller.backend.circuit
        job = {'compiled_circuit': circuit, 'shots': shots, 'seed': self.seed}
        result = QasmSimulator(job).run()
        self.assertEqual(result['status'], 'DONE')

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        shots = 1024
        self.qp.load_qasm('example', qasm_file=self.qasmFileName)
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm("example")).parse(),
                      unroll.JsonBackend(basis_gates))
        unroller.execute()
        circuit = unroller.backend.circuit
        job = {'compiled_circuit': circuit, 'shots': shots, 'seed': self.seed}
        result = QasmSimulator(job).run()
        expected = {'100100': 137, '011011': 131, '101101': 117, '111111': 127,
                    '000000': 131, '010010': 141, '110110': 116, '001001': 124}
        self.assertEqual(result['data']['counts'], expected)

    def profile_qasm_simulator(self):
        """Profile randomly generated circuits. 

        Writes profile results to <this_module>.prof as well as recording
        to the log file.

        number of circuits = 100.
        number of operations/circuit in [1, 40]
        number of qubits in [1, 5]
        """
        seed = 88
        shots = 1024
        nCircuits = 100
        minDepth = 1
        maxDepth = 40
        minQubits = 1
        maxQubits = 5
        pr = cProfile.Profile()
        randomCircuits = RandomQasmGenerator(seed,
                                             minQubits=minQubits,
                                             maxQubits=maxQubits,
                                             minDepth=minDepth,
                                             maxDepth=maxDepth)
        randomCircuits.add_circuits(nCircuits)
        self.qp = randomCircuits.getProgram()
        pr.enable()
        self.qp.execute(self.qp.get_circuit_names(),
                        backend='local_qasm_simulator',
                        shots=shots)
        pr.disable()
        sout = io.StringIO()
        ps = pstats.Stats(pr, stream=sout).sort_stats('cumulative')
        logging.info('------- start profiling QasmSimulator -----------')
        ps.print_stats()
        logging.info(sout.getvalue())
        logging.info('------- stop profiling QasmSimulator -----------')
        sout.close()
        pr.dump_stats(self.moduleName + '.prof')

    def profile_nqubit_speed_grow_depth(self):
        """simulation time vs the number of qubits 

        where the circuit depth is 10x the number of simulated
        qubits. Also creates a pdf file with this module name showing a
        plot of the results. Compilation is not included in speed.
        """
        qubitRangeMax = 15
        nQubitList = range(1,qubitRangeMax + 1)
        nCircuits = 10
        shots = 1024
        seed = 88
        maxTime = 30 # seconds; timing stops when simulation time exceeds this number
        fmtStr1 = 'profile_nqubit_speed::nqubits:{0}, backend:{1}, elapsed_time:{2:.2f}'
        fmtStr2 = 'backend:{0}, circuit:{1}, numOps:{2}, result:{3}'
        fmtStr3 = 'minDepth={minDepth}, maxDepth={maxDepth}, num circuits={nCircuits}, shots={shots}'        
        backendList = ['local_qasm_simulator', 'local_unitary_simulator']
        if shutil.which('qasm_simulator'):
            backendList.append('local_qasm_cpp_simulator')
        else:
            logging.info('profile_nqubit_speed::\"qasm_simulator\" executable not in path...skipping')
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_axes((0.1, 0.25, 0.8, 0.6))        
        for i, backend in enumerate(backendList):
            elapsedTime = np.zeros(len(nQubitList))
            if backend is 'local_unitary_simulator':
                doMeasure = False
            else:
                doMeasure = True
            j, timedOut = 0, False
            while j < qubitRangeMax and not timedOut:
                nQubits = nQubitList[j]
                randomCircuits = RandomQasmGenerator(seed,
                                                     minQubits=nQubits,
                                                     maxQubits=nQubits,
                                                     minDepth=nQubits*10,
                                                     maxDepth=nQubits*10)
                randomCircuits.add_circuits(nCircuits, doMeasure=doMeasure)
                qp = randomCircuits.getProgram()
                cnames = qp.get_circuit_names()
                qp.compile(cnames, backend=backend, shots=shots, seed=seed)
                start = time.perf_counter()
                qp.run()
                stop = time.perf_counter()
                elapsedTime[j] = stop - start
                if elapsedTime[j] > maxTime:
                    timedOut = True
                logging.info(fmtStr1.format(nQubits, backend, elapsedTime[j]))
                if backend is not 'local_unitary_simulator':                
                    for name in cnames:
                        logging.info(fmtStr2.format(
                            backend, name, len(qp.get_circuit(name)),
                            qp.get_result(name)))
                j += 1
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if backend is 'local_unitary_simulator':
                ax.plot(nQubitList[:j], elapsedTime[:j], label=backend, marker='o')
            else:
                ax.plot(nQubitList[:j], elapsedTime[:j]/shots, label=backend,
                        marker='o')
            ax.set_yscale('log', basey=10)
            ax.set_xlabel('number of qubits')
            ax.set_ylabel('process time/shot')
            ax.set_title('profile_nqubit_speed_grow_depth')
            fig.text(0.1, 0.05,
                     fmtStr3.format(minDepth='10*nQubits', maxDepth='10*nQubits',
                                    nCircuits=nCircuits, shots=shots))
            ax.legend()
        self.pdf.savefig(fig)

    def profile_nqubit_speed_constant_depth(self):
        """simulation time vs the number of qubits 

        where the circuit depth is fixed at 40. Also creates a pdf file
        with this module name showing a plot of the results. Compilation
        is not included in speed.
        """
        qubitRangeMax = 15
        nQubitList = range(1,qubitRangeMax + 1)
        maxDepth = 40
        minDepth = 40
        nCircuits = 10
        shots = 1024
        seed = 88
        maxTime = 30 # seconds; timing stops when simulation time exceeds this number
        fmtStr1 = 'profile_nqubit_speed::nqubits:{0}, backend:{1}, elapsed_time:{2:.2f}'
        fmtStr2 = 'backend:{0}, circuit:{1}, numOps:{2}, result:{3}'
        fmtStr3 = 'minDepth={minDepth}, maxDepth={maxDepth}, num circuits={nCircuits}, shots={shots}'
        backendList = ['local_qasm_simulator', 'local_unitary_simulator']
        if shutil.which('qasm_simulator'):
            backendList.append('local_qasm_cpp_simulator')
        else:
            logging.info('profile_nqubit_speed::\"qasm_simulator\" executable not in path...skipping')
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.6))
        for i, backend in enumerate(backendList):
            elapsedTime = np.zeros(len(nQubitList))
            if backend is 'local_unitary_simulator':
                doMeasure = False
            else:
                doMeasure = True
            j, timedOut = 0, False
            while j < qubitRangeMax and not timedOut:
                nQubits = nQubitList[j]
                randomCircuits = RandomQasmGenerator(seed,
                                                     minQubits=nQubits,
                                                     maxQubits=nQubits,
                                                     minDepth=minDepth,
                                                     maxDepth=maxDepth)
                randomCircuits.add_circuits(nCircuits, doMeasure=doMeasure)
                qp = randomCircuits.getProgram()
                cnames = qp.get_circuit_names()
                qp.compile(cnames, backend=backend, shots=shots, seed=seed)
                start = time.perf_counter()
                qp.run()
                stop = time.perf_counter()
                elapsedTime[j] = stop - start
                if elapsedTime[j] > maxTime:
                    timedOut = True
                logging.info(fmtStr1.format(nQubits, backend, elapsedTime[j]))
                if backend is not 'local_unitary_simulator':                
                    for name in cnames:
                        logging.info(fmtStr2.format(
                            backend, name, len(qp.get_circuit(name)),
                            qp.get_result(name)))
                j += 1
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if backend is 'local_unitary_simulator':
                ax.plot(nQubitList[:j], elapsedTime[:j], label=backend, marker='o')
            else:
                ax.plot(nQubitList[:j], elapsedTime[:j]/shots, label=backend,
                        marker='o')
            ax.set_yscale('log', basey=10)
            ax.set_xlabel('number of qubits')
            ax.set_ylabel('process time/shot')
            ax.set_title('profile_nqubit_speed_constant_depth')
            fig.text(0.1, 0.05,
                     fmtStr3.format(minDepth=minDepth, maxDepth=maxDepth,
                                    nCircuits=nCircuits, shots=shots))
            ax.legend()
        self.pdf.savefig(fig)
        
def generateTestSuite():
    """
    Generate module test suite.
    """
    testSuite = unittest.TestSuite()
    testSuite.addTest(LocalQasmSimulatorTest('test_qasm_simulator_single_shot'))
    testSuite.addTest(LocalQasmSimulatorTest('test_qasm_simulator'))
    return testSuite

def generateProfileSuite():
    """
    Generate module profile suite.
    """
    profSuite = unittest.TestSuite()
    profSuite.addTest(LocalQasmSimulatorTest('profile_qasm_simulator'))
    profSuite.addTest(LocalQasmSimulatorTest('profile_nqubit_speed_constant_depth'))
    profSuite.addTest(LocalQasmSimulatorTest('profile_nqubit_speed_grow_depth'))
    return profSuite

def main():
    """
    Optional command line entry point for testing. 
    """
    moduleName = os.path.splitext(__file__)[0]
    testSuite = generateTestSuite()
    profSuite = generateProfileSuite()
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    runner.run(testSuite)
    runner.run(profSuite)

if __name__ == '__main__':
    main()
