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


class TestPerformanceQasmSimulatorPy(QiskitTestCase):
    """
    Test performance of QasmSimulatorPy module.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pdf = PdfPages(cls.moduleName + '.pdf')

    @classmethod
    def tearDownClass(cls):
        cls.pdf.close()

    def test_performance_qasm_simulator_py(self):
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
        backend = QasmSimulatorPy()
        profile = cProfile.Profile()
        random_circuit_generator = RandomCircuitGenerator(seed=seed,
                                                          max_depth=max_depth,
                                                          min_depth=min_depth,
                                                          max_qubits=max_qubits,
                                                          min_qubits=min_qubits)
        random_circuit_generator.add_circuits(n_circuits, do_measure=True)
        random_circuits = random_circuit_generator.get_circuits()
        qobj = compile(random_circuits, backend, shots=shots)
        profile.enable()
        backend.run(qobj)
        profile.disable()
        sout = io.StringIO()
        stats = pstats.Stats(profile, stream=sout).sort_stats('cumulative')
        self.log.info('------- start profiling QasmSimulatorPy -----------')
        stats.print_stats()
        self.log.info(sout.getvalue())
        self.log.info('------- stop profiling QasmSimulatorPy -----------')
        sout.close()
        profile.dump_stats(self.moduleName + '.prof')
