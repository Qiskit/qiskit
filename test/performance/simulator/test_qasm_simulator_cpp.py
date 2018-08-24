# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import cProfile
import io
import pstats

from matplotlib.backends.backend_pdf import PdfPages
from qiskit import qasm, unroll
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import compile
from qiskit.backends.local.qasm_simulator_py import QasmSimulatorPy

from test.performance._random_circuit_generator import RandomCircuitGenerator
from test.python.common import QiskitTestCase


class TestPerformanceQasmSimulatorCpp(QiskitTestCase):
    """
    Test qasm simulator module.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pdf = PdfPages(cls.moduleName + '.pdf')

    @classmethod
    def tearDownClass(cls):
        cls.pdf.close()

    def test_performance_nqubit_speed_grow_depth(self):
        """Simulation time vs the number of qubits

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
        backend = get_backend('local_unitary_simulator_py')
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
        axes = fig.add_axes((0.1, 0.25, 0.8, 0.6))
        for _, backend in enumerate(backend_list):
            elapsed_time = np.zeros(len(n_qubit_list))
            if backend == 'local_unitary_simulator_py':
                do_measure = False
            else:
                do_measure = True
            j, timed_out = 0, False
            while j < qubit_range_max and not timed_out:
                n_qubits = n_qubit_list[j]
                random_circuit_generator = RandomCircuitGenerator(seed=seed,
                                                               max_depth=n_qubits * 10,
                                                               min_depth=n_qubits * 10,
                                                               max_qubits=n_qubits,
                                                               min_qubits=n_qubits)
                random_circuits = random_circuit_generator.get_circuits()
                random_circuits.add_circuits(n_circuits, do_measure=do_measure)
                qobj = compile(random_circuits, backend=backend, shots=shots, seed=seed)
                start = time.perf_counter()
                results = backend.run(qobj)
                stop = time.perf_counter()
                elapsed_time[j] = stop - start
                if elapsed_time[j] > max_time:
                    timed_out = True
                self.log.info(fmt_str1.format(n_qubits, backend, elapsed_time[j]))
                if not isinstance(backend, LocalQasmSimulatorPy()):
                    for name in c_names:
                        log_str = fmt_str2.format(
                            backend, name, len(program.get_circuit(name)),
                            results.get_data(name))
                        self.log.info(log_str)
                j += 1
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            if isinstance(backend, LocalQasmSimulatorPy()):
                axes.plot(n_qubit_list[:j], elapsed_time[:j], label=backend, marker='o')
            else:
                axes.plot(n_qubit_list[:j], elapsed_time[:j]/shots, label=backend,
                          marker='o')
            axes.set_yscale('log', basey=10)
            axes.set_xlabel('number of qubits')
            axes.set_ylabel('process time/shot')
            axes.set_title('profile_nqubit_speed_grow_depth')
            fig.text(0.1, 0.05,
                     fmt_str3.format(minDepth='10*nQubits', maxDepth='10*nQubits',
                                     nCircuits=n_circuits, shots=shots))
            axes.legend()
        self.pdf.savefig(fig)

    def test_performance_nqubit_speed_constant_depth(self):
        """Simulation time vs the number of qubits

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
        axes = fig.add_axes((0.1, 0.2, 0.8, 0.6))
        for _, backend in enumerate(backend_list):
            elapsed_time = np.zeros(len(n_qubit_list))
            if backend == 'local_unitary_simulator_py':
                do_measure = False
            else:
                do_measure = True
            j, timed_out = 0, False
            while j < qubit_range_max and not timed_out:
                n_qubits = n_qubit_list[j]
                random_circuits = RandomCircuitGenerator(seed,
                                                      min_qubits=n_qubits,
                                                      max_qubits=n_qubits,
                                                      min_depth=min_depth,
                                                      max_depth=max_depth)
                random_circuits.add_circuits(n_circuits, do_measure=do_measure)
                qprogram = random_circuits.get_program()
                cnames = qprogram.get_circuit_names()
                qobj = qprogram.compile(cnames, backend=backend, shots=shots, seed=seed)
                start = time.perf_counter()
                results = qprogram.run(qobj)
                stop = time.perf_counter()
                elapsed_time[j] = stop - start
                if elapsed_time[j] > max_time:
                    timed_out = True
                self.log.info(fmt_str1.format(n_qubits, backend, elapsed_time[j]))
                if backend != 'local_unitary_simulator_py':
                    for name in cnames:
                        log_str = fmt_str2.format(
                            backend, name, len(qprogram.get_circuit(name)),
                            results.get_data(name))
                        self.log.info(log_str)
                j += 1
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            if backend == 'local_unitary_simulator_py':
                axes.plot(n_qubit_list[:j], elapsed_time[:j], label=backend, marker='o')
            else:
                axes.plot(n_qubit_list[:j], elapsed_time[:j]/shots, label=backend,
                          marker='o')
            axes.set_yscale('log', basey=10)
            axes.set_xlabel('number of qubits')
            axes.set_ylabel('process time/shot')
            axes.set_title('profile_nqubit_speed_constant_depth')
            fig.text(0.1, 0.05,
                     fmt_str3.format(minDepth=min_depth, maxDepth=max_depth,
                                     nCircuits=n_circuits, shots=shots))
            axes.legend()
        self.pdf.savefig(fig)
