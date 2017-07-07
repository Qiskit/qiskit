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
    
    def setUp(self):
        self.seed = 88
        self.qasmFileName = os.path.join(qiskit.__path__[0],
                                         '../test/python/qasm/example.qasm')
        self.qp = QuantumProgram()
        self.moduleName = os.path.splitext(__file__)[0]
        logFileName = self.moduleName + '.log'
        logging.basicConfig(filename=logFileName, level=logging.INFO)

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
        result = QasmSimulator(circuit, shots, self.seed).run()
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
        result = QasmSimulator(circuit, shots, self.seed).run()
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
        shots = 1024
        nCircuits = 100
        maxDepth = 40
        maxQubits = 5
        pr = cProfile.Profile()
        randomCircuits = RandomQasmGenerator(seed=self.seed,
                                             maxDepth=maxDepth,
                                             maxQubits=maxQubits)
        randomCircuits.add_circuits(nCircuits)
        self.qp = randomCircuits.getProgram()
        pr.enable()
        self.qp.execute(self.qp.get_circuit_names(),
                        device='local_qasm_simulator',
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

    def profile_nqubit_speed(self):
        """
        Record the elapsed time of the simulators vs the number of qubits to
        the log file. Also creates a pdf file with this module name showing a
        plot of the results. Compilation is not included in speed.
        """
        qubitRangeMax = 8
        nQubitList = range(1, qubitRangeMax+1)
        nCircuits = 10
        shots = 1024
        seed = 1
        fmtStr = 'profile_nqubit_speed::nqubits:{0}, simulator:{1}, elapsed_time:{2:.2f}'
        pdf = PdfPages(self.moduleName + '.pdf')
        deviceList = ['local_qasm_simulator', 'local_unitary_simulator']
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_subplot(111)
        for i, device in enumerate(deviceList):
            elapsedTime = np.zeros(len(nQubitList))
            if device is 'local_unitary_simulator':
                doMeasure = False
            else:
                doMeasure = True
            for j, nQubits in enumerate(nQubitList):
                randomCircuits = RandomQasmGenerator(seed, maxQubits=nQubits,
                                                     minQubits=nQubits,
                                                     minDepth=nQubits*10,
                                                     maxDepth=nQubits*10)
                randomCircuits.add_circuits(nCircuits, doMeasure=doMeasure)
                qp = randomCircuits.getProgram()
                print('-'*40)
                cnames = qp.get_circuit_names()
                qp.compile(cnames, device=device, shots=shots, seed=seed)
                start = time.process_time()
                qp.run()
                stop = time.process_time()
                elapsedTime[j] = stop - start
                logging.info(fmtStr.format(nQubits, device, elapsedTime[j]))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.plot(nQubitList, elapsedTime, label=device, marker='o')
            ax.set_yscale('log', basey=10)
            ax.set_xlabel('number of qubits')
            ax.set_ylabel('process time')
            ax.legend()
        pdf.savefig(fig)
        pdf.close()

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
    profSuite.addTest(LocalQasmSimulatorTest('profile_nqubit_speed'))
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
