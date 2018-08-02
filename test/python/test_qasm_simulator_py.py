# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring
from sys import version_info
import cProfile
import io
import pstats
import shutil
import time
import unittest

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from qiskit import (qasm, unroll, transpiler,
                    ClassicalRegister, QuantumRegister, QuantumCircuit)
from qiskit.backends.local.qasm_simulator_py import QasmSimulatorPy
from qiskit.qobj import Qobj, QobjHeader, QobjItem, QobjConfig, QobjExperiment

from ._random_qasm_generator import RandomQasmGenerator
from .common import QiskitTestCase


do_profiling = False


class TestLocalQasmSimulatorPy(QiskitTestCase):
    """Test local_qasm_simulator_py."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if do_profiling:
            cls.pdf = PdfPages(cls.moduleName + '.pdf')

    @classmethod
    def tearDownClass(cls):
        if do_profiling:
            cls.pdf.close()

    def setUp(self):
        self.qp = None
        self.seed = 88
        qasm_filename = self._get_resource_path('qasm/example.qasm')
        unroller = unroll.Unroller(qasm.Qasm(filename=qasm_filename).parse(),
                                   unroll.JsonBackend([]))
        circuit = QobjExperiment.from_dict(unroller.execute())
        circuit.config = QobjItem(coupling_map=None,
                                  basis_gates='u1,u2,u3,cx,id',
                                  layout=None,
                                  seed=self.seed)
        circuit.header.name = 'test'

        self.qobj = Qobj(qobj_id='test_sim_single_shot',
                         config=QobjConfig(shots=1024,
                                           memory_slots=6,
                                           max_credits=3),
                         experiments=[circuit],
                         header=QobjHeader(
                             backend_name='local_qasm_simulator_py'))

    def test_qasm_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        self.qobj.config.shots = shots
        result = QasmSimulatorPy().run(self.qobj).result()
        self.assertEqual(result.get_status(), 'COMPLETED')

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = QasmSimulatorPy().run(self.qobj).result()
        shots = 1024
        threshold = 0.04 * shots
        counts = result.get_counts('test')
        target = {'100 100': shots / 8, '011 011': shots / 8,
                  '101 101': shots / 8, '111 111': shots / 8,
                  '000 000': shots / 8, '010 010': shots / 8,
                  '110 110': shots / 8, '001 001': shots / 8}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_if_statement(self):
        self.log.info('test_if_statement_x')
        shots = 100
        max_qubits = 3
        qr = QuantumRegister(max_qubits, 'qr')
        cr = ClassicalRegister(max_qubits, 'cr')
        circuit_if_true = QuantumCircuit(qr, cr, name='test_if_true')
        circuit_if_true.x(qr[0])
        circuit_if_true.x(qr[1])
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.x(qr[2]).c_if(cr, 0x3)
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.measure(qr[2], cr[2])
        circuit_if_false = QuantumCircuit(qr, cr, name='test_if_false')
        circuit_if_false.x(qr[0])
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.x(qr[2]).c_if(cr, 0x3)
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.measure(qr[2], cr[2])
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=circuit_if_true.qasm()).parse(),
            unroll.JsonBackend(basis_gates))
        ucircuit_true = QobjExperiment.from_dict(unroller.execute())
        unroller = unroll.Unroller(
            qasm.Qasm(data=circuit_if_false.qasm()).parse(),
            unroll.JsonBackend(basis_gates))
        ucircuit_false = QobjExperiment.from_dict(unroller.execute())

        # Customize the experiments and create the qobj.
        ucircuit_true.config = QobjItem(coupling_map=None,
                                        basis_gates='u1,u2,u3,cx,id',
                                        layout=None,
                                        seed=None)
        ucircuit_true.header.name = 'test_if_true'
        ucircuit_false.config = QobjItem(coupling_map=None,
                                         basis_gates='u1,u2,u3,cx,id',
                                         layout=None,
                                         seed=None)
        ucircuit_false.header.name = 'test_if_false'

        qobj = Qobj(qobj_id='test_if_qobj',
                    config=QobjConfig(max_credits=3,
                                      shots=shots,
                                      memory_slots=max_qubits),
                    experiments=[ucircuit_true, ucircuit_false],
                    header=QobjHeader(backend_name='local_qasm_simulator_py'))

        result = QasmSimulatorPy().run(qobj).result()
        result_if_true = result.get_data('test_if_true')
        self.log.info('result_if_true circuit:')
        self.log.info(circuit_if_true.qasm())
        self.log.info('result_if_true=%s', result_if_true)

        result_if_false = result.get_data('test_if_false')
        self.log.info('result_if_false circuit:')
        self.log.info(circuit_if_false.qasm())
        self.log.info('result_if_false=%s', result_if_false)
        self.assertTrue(result_if_true['counts']['111'] == 100)
        self.assertTrue(result_if_false['counts']['001'] == 100)

    @unittest.skipIf(version_info.minor == 5,
                     "Due to gate ordering issues with Python 3.5 "
                     "we have to disable this test until fixed")
    def test_teleport(self):
        """test teleportation as in tutorials"""
        self.log.info('test_teleport')
        pi = np.pi
        shots = 1000
        qr = QuantumRegister(3, 'qr')
        cr0 = ClassicalRegister(1, 'cr0')
        cr1 = ClassicalRegister(1, 'cr1')
        cr2 = ClassicalRegister(1, 'cr2')
        circuit = QuantumCircuit(qr, cr0, cr1, cr2, name='teleport')
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
        backend = QasmSimulatorPy()
        qobj = transpiler.compile(circuit, backend=backend, shots=shots,
                                  seed=self.seed)
        results = backend.run(qobj).result()
        data = results.get_counts('teleport')
        alice = {
            '00': data['0 0 0'] + data['1 0 0'],
            '01': data['0 1 0'] + data['1 1 0'],
            '10': data['0 0 1'] + data['1 0 1'],
            '11': data['0 1 1'] + data['1 1 1']
        }
        bob = {
            '0': data['0 0 0'] + data['0 1 0'] + data['0 0 1'] + data['0 1 1'],
            '1': data['1 0 0'] + data['1 1 0'] + data['1 0 1'] + data['1 1 1']
        }
        self.log.info('test_teleport: circuit:')
        self.log.info('test_teleport: circuit:')
        self.log.info(circuit.qasm())
        self.log.info('test_teleport: data %s', data)
        self.log.info('test_teleport: alice %s', alice)
        self.log.info('test_teleport: bob %s', bob)
        alice_ratio = 1/np.tan(pi/8)**2
        bob_ratio = bob['0']/float(bob['1'])
        error = abs(alice_ratio - bob_ratio) / alice_ratio
        self.log.info('test_teleport: relative error = %s', error)
        self.assertLess(error, 0.05)

    @unittest.skipIf(not do_profiling, "skipping simulator profiling.")
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
        n_circuits = 100
        min_depth = 1
        max_depth = 40
        min_qubits = 1
        max_qubits = 5
        pr = cProfile.Profile()
        random_circuits = RandomQasmGenerator(seed,
                                              min_qubits=min_qubits,
                                              max_qubits=max_qubits,
                                              min_depth=min_depth,
                                              max_depth=max_depth)
        random_circuits.add_circuits(n_circuits)
        self.qp = random_circuits.get_program()
        pr.enable()
        self.qp.execute(self.qp.get_circuit_names(),
                        backend='local_qasm_simulator_py',
                        shots=shots)
        pr.disable()
        sout = io.StringIO()
        ps = pstats.Stats(pr, stream=sout).sort_stats('cumulative')
        self.log.info('------- start profiling QasmSimulatorPy -----------')
        ps.print_stats()
        self.log.info(sout.getvalue())
        self.log.info('------- stop profiling QasmSimulatorPy -----------')
        sout.close()
        pr.dump_stats(self.moduleName + '.prof')

    @unittest.skipIf(not do_profiling, "skipping simulator profiling.")
    def profile_nqubit_speed_grow_depth(self):
        """simulation time vs the number of qubits

        where the circuit depth is 10x the number of simulated
        qubits. Also creates a pdf file with this module name showing a
        plot of the results. Compilation is not included in speed.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        qubit_range_max = 15
        n_qubit_list = range(1, qubit_range_max + 1)
        n_circuits = 10
        shots = 1024
        seed = 88
        max_time = 30  # seconds; timing stops when simulation time exceeds this number
        fmt_str1 = 'profile_nqubit_speed::nqubits:{0}, backend:{1}, elapsed_time:{2:.2f}'
        fmt_str2 = 'backend:{0}, circuit:{1}, numOps:{2}, result:{3}'
        fmt_str3 = 'minDepth={minDepth}, maxDepth={maxDepth}, num circuits={nCircuits},' \
                   'shots={shots}'
        backend_list = ['local_qasm_simulator_py', 'local_unitary_simulator_py']
        if shutil.which('qasm_simulator'):
            backend_list.append('local_qasm_simulator_cpp')
        else:
            self.log.info('profile_nqubit_speed::\"qasm_simulator\" executable'
                          'not in path...skipping')
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_axes((0.1, 0.25, 0.8, 0.6))
        for _, backend in enumerate(backend_list):
            elapsed_time = np.zeros(len(n_qubit_list))
            if backend == 'local_unitary_simulator_py':
                do_measure = False
            else:
                do_measure = True
            j, timed_out = 0, False
            while j < qubit_range_max and not timed_out:
                n_qubits = n_qubit_list[j]
                random_circuits = RandomQasmGenerator(seed,
                                                      min_qubits=n_qubits,
                                                      max_qubits=n_qubits,
                                                      min_depth=n_qubits * 10,
                                                      max_depth=n_qubits * 10)
                random_circuits.add_circuits(n_circuits, do_measure=do_measure)
                qp = random_circuits.get_program()
                c_names = qp.get_circuit_names()
                qobj = qp.compile(c_names, backend=backend, shots=shots,
                                  seed=seed)
                start = time.perf_counter()
                results = qp.run(qobj)
                stop = time.perf_counter()
                elapsed_time[j] = stop - start
                if elapsed_time[j] > max_time:
                    timed_out = True
                self.log.info(fmt_str1.format(n_qubits, backend, elapsed_time[j]))
                if backend != 'local_unitary_simulator_py':
                    for name in c_names:
                        log_str = fmt_str2.format(
                            backend, name, len(qp.get_circuit(name)),
                            results.get_data(name))
                        self.log.info(log_str)
                j += 1
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if backend == 'local_unitary_simulator_py':
                ax.plot(n_qubit_list[:j], elapsed_time[:j], label=backend, marker='o')
            else:
                ax.plot(n_qubit_list[:j], elapsed_time[:j]/shots, label=backend,
                        marker='o')
            ax.set_yscale('log', basey=10)
            ax.set_xlabel('number of qubits')
            ax.set_ylabel('process time/shot')
            ax.set_title('profile_nqubit_speed_grow_depth')
            fig.text(0.1, 0.05,
                     fmt_str3.format(minDepth='10*nQubits', maxDepth='10*nQubits',
                                     nCircuits=n_circuits, shots=shots))
            ax.legend()
        self.pdf.savefig(fig)

    @unittest.skipIf(not do_profiling, "skipping simulator profiling.")
    def profile_nqubit_speed_constant_depth(self):
        """simulation time vs the number of qubits

        where the circuit depth is fixed at 40. Also creates a pdf file
        with this module name showing a plot of the results. Compilation
        is not included in speed.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        qubit_range_max = 15
        n_qubit_list = range(1, qubit_range_max + 1)
        max_depth = 40
        min_depth = 40
        n_circuits = 10
        shots = 1024
        seed = 88
        max_time = 30  # seconds; timing stops when simulation time exceeds this number
        fmt_str1 = 'profile_nqubit_speed::nqubits:{0}, backend:{1},' \
                   'elapsed_time:{2:.2f}'
        fmt_str2 = 'backend:{0}, circuit:{1}, numOps:{2}, result:{3}'
        fmt_str3 = 'minDepth={minDepth}, maxDepth={maxDepth},' \
                   'num circuits={nCircuits}, shots={shots}'
        backend_list = ['local_qasm_simulator_py', 'local_unitary_simulator_py']
        if shutil.which('qasm_simulator'):
            backend_list.append('local_qasm_simulator_cpp')
        else:
            self.log.info('profile_nqubit_speed::\"qasm_simulator\" executable'
                          'not in path...skipping')
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.6))
        for _, backend in enumerate(backend_list):
            elapsedTime = np.zeros(len(n_qubit_list))
            if backend == 'local_unitary_simulator_py':
                doMeasure = False
            else:
                doMeasure = True
            j, timedOut = 0, False
            while j < qubit_range_max and not timedOut:
                nQubits = n_qubit_list[j]
                randomCircuits = RandomQasmGenerator(seed,
                                                     min_qubits=nQubits,
                                                     max_qubits=nQubits,
                                                     min_depth=min_depth,
                                                     max_depth=max_depth)
                randomCircuits.add_circuits(n_circuits, do_measure=doMeasure)
                qp = randomCircuits.get_program()
                cnames = qp.get_circuit_names()
                qobj = qp.compile(cnames, backend=backend, shots=shots, seed=seed)
                start = time.perf_counter()
                results = qp.run(qobj)
                stop = time.perf_counter()
                elapsedTime[j] = stop - start
                if elapsedTime[j] > max_time:
                    timedOut = True
                self.log.info(fmt_str1.format(nQubits, backend, elapsedTime[j]))
                if backend != 'local_unitary_simulator_py':
                    for name in cnames:
                        log_str = fmt_str2.format(
                            backend, name, len(qp.get_circuit(name)),
                            results.get_data(name))
                        self.log.info(log_str)
                j += 1
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if backend == 'local_unitary_simulator_py':
                ax.plot(n_qubit_list[:j], elapsedTime[:j], label=backend, marker='o')
            else:
                ax.plot(n_qubit_list[:j], elapsedTime[:j]/shots, label=backend,
                        marker='o')
            ax.set_yscale('log', basey=10)
            ax.set_xlabel('number of qubits')
            ax.set_ylabel('process time/shot')
            ax.set_title('profile_nqubit_speed_constant_depth')
            fig.text(0.1, 0.05,
                     fmt_str3.format(minDepth=min_depth, maxDepth=max_depth,
                                     nCircuits=n_circuits, shots=shots))
            ax.legend()
        self.pdf.savefig(fig)


if __name__ == '__main__':
    unittest.main()
