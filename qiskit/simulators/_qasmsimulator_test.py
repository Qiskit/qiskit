#!/usr/bin/env python
import unittest
import array
import math
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
try:
    import qiskit
except ImportError as ierr:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    import qiskit
from qiskit import QuantumProgram
from qiskit.simulators._unitarysimulator import UnitarySimulator
from qiskit.simulators._qasmsimulator import QasmSimulator
import qiskit.qasm as qasm
import qiskit.unroll as unroll

class LocalSimulatorTest(unittest.TestCase):
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
        c = QasmSimulator(circuit, shots, self.seed).run()
        expected = {'100100': 137, '011011': 131, '101101': 117, '111111': 127,
                    '000000': 131, '010010': 141, '110110': 116, '001001': 124}
        self.assertEqual(c['data']['counts'], expected)

    def profile_qasm_simulator(self):
        """Profile randomly generated circuits. 

        Writes profile results to <this_module>.prof as well as recording
        to the log file.

        number of circuits = 100.
        number of operations/circuit in [1, 40]
        number of qubits in [1, 5]
        """
        shots = 1024
        ncircuits = 100
        maxDepth = 40
        maxQubits = 5
        pr = cProfile.Profile()
        randomCircuits = RandomQasmGenerator(seed=self.seed,
                                             maxDepth=maxDepth,
                                             maxQubits=maxQubits)
        randomCircuits.add_circuits(ncircuits)
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

class RandomQasmGenerator():
    """
    Generate random size circuits for profiling.
    """
    def __init__(self, seed=None, maxDepth=100, maxQubits=5):
        """
        Args:
          seed: Random number seed. If none, don't seed the generator.
          maxDepth: Maximum number of operations in a circuit.
          maxQubits: Maximum number of qubits in a circuit.
        """
        self.maxDepth = maxDepth
        self.maxQubits = maxQubits
        self.qp = QuantumProgram()
        self.qr = self.qp.create_quantum_registers('qr', maxQubits)
        self.cr = self.qp.create_classical_registers('cr', maxQubits)
        self.circuitNameList = []
        self.nQubitList = []
        self.depthList = []
        if seed is not None:
            random.seed(a=seed)
            
    def add_circuits(self, ncircuits):
        """Adds circuits to program.

        Args:
          ncircuits (int): Number of circuits to add.
        """
        self.circuitNameList = []
        self.nQubitList = random.choices(range(1, self.maxQubits+1), k=ncircuits)
        self.depthList = random.choices(range(1, self.maxDepth+1), k=ncircuits)
        for i in range(ncircuits):
            circuitName = ''.join(random.choices(string.ascii_uppercase
                                                 + string.digits, k=10))
            self.circuitNameList.append(circuitName)
            nQubits = self.nQubitList[i]
            depth = self.depthList[i]
            circuit = self.qp.create_circuit(circuitName, ['qr'], ['cr'])
            for j in range(depth):
                if nQubits == 1:
                    opInd = 0
                else:
                    opInd = random.randint(0, 1)
                if opInd == 0: # U3
                    qind = random.randint(0, nQubits-1)
                    if qind==5:
                        import pdb;pdb.set_trace()
                    circuit.u3(random.random(), random.random(), random.random(),
                               self.qr[qind])
                elif opInd == 1: # CX
                    source, target = random.sample(range(nQubits), 2)
                    circuit.cx(self.qr[source], self.qr[target])
            # add measurements to end of circuit
            nmeasure = random.randint(0, nQubits-1)
            for j in range(nmeasure):
                qind = random.randint(0, nQubits-1)
                circuit.measure(self.qr[qind], self.cr[qind])

    def get_circuit_names(self):
        return self.circuitNameList

    def getProgram(self):
        return self.qp
    
        
    
def generateTestSuite():
    """
    Generate module test suite.
    """
    testSuite = unittest.TestSuite()
    testSuite.addTest(LocalSimulatorTest('test_qasm_simulator'))
    return testSuite

def generateProfileSuite():
    """
    Generate module profile suite.
    """
    profSuite = unittest.TestSuite()
    profSuite.addTest(LocalSimulatorTest('profile_qasm_simulator'))
    return profSuite

def main():
    """
    Optional command line entry point for testing. Sends unittest output 
    to file with same name as this module but with ".txt" extension.
    """
    moduleName = os.path.splitext(__file__)[0]
    reportFileName = moduleName + '.txt'
    testSuite = generateTestSuite()
    profSuite = generateProfileSuite()
    with open(reportFileName, 'w') as ofile:
        runner = unittest.TextTestRunner(stream=ofile, verbosity=2)
        runner.run(testSuite)
        runner.run(profSuite)        

if __name__ == '__main__':
    main()
