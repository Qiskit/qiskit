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

import cProfile
import io
import pstats
import shutil
import time
import unittest

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from qiskit import qasm, unroll, QuantumProgram, QuantumJob
from qiskit.backends._qasmsimulator import QasmSimulator

from ._random_qasm_generator import RandomQasmGenerator
from .common import QiskitTestCase


class LocalQasmSimulatorTest(QiskitTestCase):
    """Test local qasm simulator."""

    @classmethod
    def setUpClass(cls):
        super(LocalQasmSimulatorTest, cls).setUpClass()
        cls.pdf = PdfPages(cls.moduleName + '.pdf')

    @classmethod
    def tearDownClass(cls):
        cls.pdf.close()

    def setUp(self):
        self.seed = 88
        self.qasmFileName = self._get_resource_path('qasm/example.qasm')
        self.qp = QuantumProgram()
        self.qp.load_qasm_file(self.qasmFileName, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm('example')).parse(),
                      unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        circuit_config = {'coupling_map': None,
                          'basis_gates': 'u1,u2,u3,cx,id',
                          'layout': None,
                          'seed': self.seed}
        resources = {'max_credits': 3,
                     'wait': 5,
                     'timeout': 120}
        self.qobj = {'id': 'test_sim_single_shot',
                     'config': {
                         'max_credits': resources['max_credits'],
                         'shots': 1024,
                         'backend': 'local_qasm_simulator',
                     },
                     'circuits': [
                         {
                             'name': 'test',
                             'compiled_circuit': circuit,
                             'compiled_circuit_qasm': None,
                             'config': circuit_config
                         }
                     ]
        }
        self.q_job = QuantumJob(self.qobj,
                                backend='local_qasm_simulator',
                                circuit_config=circuit_config,
                                seed=self.seed,
                                resources=resources,
                                preformatted=True
                                )
                                

    def tearDown(self):
        pass

    def test_qasm_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        self.qobj['config']['shots'] = shots
        result = QasmSimulator().run(self.q_job)
        self.assertEqual(result.get_status(), 'COMPLETED')

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = QasmSimulator().run(self.q_job)
        expected = {'100 100': 137, '011 011': 131, '101 101': 117, '111 111': 127,
                    '000 000': 131, '010 010': 141, '110 110': 116, '001 001': 124}
        self.assertEqual(result.get_counts('test'), expected)

    def test_if_statement(self):
        self.log.info('test_if_statement_x')
        shots = 100
        max_qubits = 3
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', max_qubits)
        cr = qp.create_classical_register('cr', max_qubits)
        circuit_if_true = qp.create_circuit('test_if_true', [qr], [cr])
        circuit_if_true.x(qr[0])
        circuit_if_true.x(qr[1])
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.x(qr[2]).c_if(cr, 0x3)
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.measure(qr[2], cr[2])
        circuit_if_false = qp.create_circuit('test_if_false', [qr], [cr])
        circuit_if_false.x(qr[0])
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.x(qr[2]).c_if(cr, 0x3)
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.measure(qr[2], cr[2])
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=qp.get_qasm('test_if_true')).parse(),
            unroll.JsonBackend(basis_gates))
        ucircuit_true = unroller.execute()
        unroller = unroll.Unroller(
            qasm.Qasm(data=qp.get_qasm('test_if_false')).parse(),
            unroll.JsonBackend(basis_gates))
        ucircuit_false = unroller.execute()
        config = {'shots': shots, 'seed': self.seed}
        qobj = {'id': 'test_if_qobj',
                'config': {
                    'max_credits': 3,
                    'shots': shots,
                    'backend': 'local_qasm_simulator',
                },
                'circuits': [
                    {
                        'name': 'test_if_true',
                        'compiled_circuit': ucircuit_true,
                        'compiled_circuit_qasm': None,
                        'config': {'coupling_map': None,
                                   'basis_gates': 'u1,u2,u3,cx,id',
                                   'layout': None,
                                   'seed': None
                                   }
                    },
                    {
                        'name': 'test_if_false',
                        'compiled_circuit': ucircuit_false,
                        'compiled_circuit_qasm': None,
                        'config': {'coupling_map': None,
                                   'basis_gates': 'u1,u2,u3,cx,id',
                                   'layout': None,
                                   'seed': None
                                   }
                    }
                ]
        }
        q_job = QuantumJob(qobj, preformatted=True)
        result = QasmSimulator().run(q_job)
        result_if_true = result.get_data('test_if_true')
        self.log.info('result_if_true circuit:')
        self.log.info(circuit_if_true.qasm())
        self.log.info('result_if_true={0}'.format(result_if_true))

        result_if_false = result.get_data('test_if_false')        
        self.log.info('result_if_false circuit:')
        self.log.info(circuit_if_false.qasm())
        self.log.info('result_if_false={0}'.format(result_if_false))
        self.assertTrue(result_if_true['counts']['111'] == 100)
        self.assertTrue(result_if_false['counts']['001'] == 100)

    def test_teleport(self):
        """test teleportation as in tutorials"""

        self.log.info('test_teleport')
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
        self.log.info('test_telport: circuit:')
        self.log.info( circuit.qasm() )
        self.log.info('test_teleport: data {0}'.format(data))
        self.log.info('test_teleport: alice {0}'.format(alice))
        self.log.info('test_teleport: bob {0}'.format(bob))
        alice_ratio = 1/np.tan(pi/8)**2
        bob_ratio = bob['0']/float(bob['1'])
        error = abs(alice_ratio - bob_ratio) / alice_ratio
        self.log.info('test_teleport: relative error = {0:.4f}'.format(error))
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
        self.log.info('------- start profiling QasmSimulator -----------')
        ps.print_stats()
        self.log.info(sout.getvalue())
        self.log.info('------- stop profiling QasmSimulator -----------')
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
            self.log.info('profile_nqubit_speed::\"qasm_simulator\" executable not in path...skipping')
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
                self.log.info(fmtStr1.format(nQubits, backend, elapsedTime[j]))
                if backend is not 'local_unitary_simulator':
                    for name in cnames:
                        self.log.info(fmtStr2.format(
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
            self.log.info('profile_nqubit_speed::\"qasm_simulator\" executable not in path...skipping')
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
                self.log.info(fmtStr1.format(nQubits, backend, elapsedTime[j]))
                if backend is not 'local_unitary_simulator':
                    for name in cnames:
                        self.log.info(fmtStr2.format(
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
