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
import json
import shutil
from matplotlib.backends.backend_pdf import PdfPages
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
        log_fmt = 'LocalQasmSimulatorTest:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)

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
        self.qp.load_qasm_file(self.qasmFileName, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm("example")).parse(),
                      unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        config = {'shots': shots, 'seed': self.seed}
        job = {'compiled_circuit': circuit, 'config': config}
        result = QasmSimulator(job).run()
        self.assertEqual(result['status'], 'DONE')

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        shots = 1024
        self.qp.load_qasm_file(self.qasmFileName, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm("example")).parse(),
                      unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        config = {'shots': shots, 'seed': self.seed}
        job = {'compiled_circuit': circuit, 'config': config}
        result = QasmSimulator(job).run()
        expected = {'100 100': 137, '011 011': 131, '101 101': 117, '111 111': 127,
                    '000 000': 131, '010 010': 141, '110 110': 116, '001 001': 124}
        self.assertEqual(result['data']['counts'], expected)

    def test_if_statement(self):
        logging.info('test_if_statement_x')
        shots = 100
        max_qubits = 3
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', max_qubits)
        cr = qp.create_classical_register('cr', max_qubits)
        circuit = qp.create_circuit('test_if', [qr], [cr])
        circuit.x(qr[0])
        circuit.x(qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.x(qr[2]).c_if(cr, 0x3)
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[2], cr[2])
        circuit2 = qp.create_circuit('test_if_case_2', [qr], [cr])
        circuit2.x(qr[0])
        circuit2.measure(qr[0], cr[0])
        circuit2.measure(qr[1], cr[1])
        circuit2.x(qr[2]).c_if(cr, 0x3)
        circuit2.measure(qr[0], cr[0])
        circuit2.measure(qr[1], cr[1])
        circuit2.measure(qr[2], cr[2])
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=qp.get_qasm('test_if')).parse(),
            unroll.JsonBackend(basis_gates))
        ucircuit = unroller.execute()
        unroller = unroll.Unroller(
            qasm.Qasm(data=qp.get_qasm('test_if_case_2')).parse(),
            unroll.JsonBackend(basis_gates))
        ucircuit2 = unroller.execute()
        config = {'shots': shots, 'seed': self.seed}
        job = {'compiled_circuit': ucircuit, 'config': config}
        result_if_true = QasmSimulator(job).run()
        job = {'compiled_circuit': ucircuit2, 'config': config}
        result_if_false = QasmSimulator(job).run()

        logging.info('result_if_true circuit:')
        logging.info(circuit.qasm())
        logging.info('result_if_true={0}'.format(result_if_true))

        del circuit.data[1]
        logging.info('result_if_false circuit:')
        logging.info(circuit.qasm())
        logging.info('result_if_false={0}'.format(result_if_false))
        self.assertTrue(result_if_true['data']['counts']['111'] == 100)
        self.assertTrue(result_if_false['data']['counts']['001'] == 100)

    def test_teleport(self):
        """test teleportation as in tutorials"""

        logging.info('test_teleport')
        pi = np.pi
        shots = 1000
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 3)
        cr0 = qp.create_classical_register('cr0', 1)
        cr1 = qp.create_classical_register('cr1', 1)
        cr2 = qp.create_classical_register('cr2', 1)
        circuit = qp.create_circuit('teleport', [qr],
                                    [cr0, cr1, cr2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.ry(pi/4, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr0[0])
        circuit.measure(qr[1], cr1[0])
        circuit.z(qr[2]).c_if(cr0, 1)
        circuit.x(qr[2]).c_if(cr1, 1)
        circuit.measure(qr[2], cr2[0])
        backend = 'local_qasm_simulator'
        qobj = qp.compile('teleport', backend=backend, shots=shots,
                   seed=self.seed)
        results = qp.run(qobj)
        data = results.get_counts('teleport')
        alice = {}
        bob = {}
        alice['00'] = data['0 0 0'] + data['1 0 0']
        alice['01'] = data['0 1 0'] + data['1 1 0']
        alice['10'] = data['0 0 1'] + data['1 0 1']
        alice['11'] = data['0 1 1'] + data['1 1 1']
        bob['0'] = data['0 0 0'] + data['0 1 0'] +  data['0 0 1'] + data['0 1 1']
        bob['1'] = data['1 0 0'] + data['1 1 0'] +  data['1 0 1'] + data['1 1 1']
        logging.info('test_telport: circuit:')
        logging.info( circuit.qasm() )
        logging.info('test_teleport: data {0}'.format(data))
        logging.info('test_teleport: alice {0}'.format(alice))
        logging.info('test_teleport: bob {0}'.format(bob))
        alice_ratio = 1/np.tan(pi/8)**2
        bob_ratio = bob['0']/float(bob['1'])
        error = abs(alice_ratio - bob_ratio) / alice_ratio
        logging.info('test_teleport: relative error = {0:.4f}'.format(error))
        self.assertLess(error, 0.05)

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
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
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
                qobj = qp.compile(cnames, backend=backend, shots=shots,
                                  seed=seed)
                start = time.perf_counter()
                results = qp.run(qobj)
                stop = time.perf_counter()
                elapsedTime[j] = stop - start
                if elapsedTime[j] > maxTime:
                    timedOut = True
                logging.info(fmtStr1.format(nQubits, backend, elapsedTime[j]))
                if backend is not 'local_unitary_simulator':
                    for name in cnames:
                        logging.info(fmtStr2.format(
                            backend, name, len(qp.get_circuit(name)),
                            results.get_data(name)))
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
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
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
                qobj = qp.compile(cnames, backend=backend, shots=shots, seed=seed)
                start = time.perf_counter()
                results = qp.run(qobj)
                stop = time.perf_counter()
                elapsedTime[j] = stop - start
                if elapsedTime[j] > maxTime:
                    timedOut = True
                logging.info(fmtStr1.format(nQubits, backend, elapsedTime[j]))
                if backend is not 'local_unitary_simulator':
                    for name in cnames:
                        logging.info(fmtStr2.format(
                            backend, name, len(qp.get_circuit(name)),
                            results.get_data(name)))
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

if __name__ == '__main__':
    unittest.main()
