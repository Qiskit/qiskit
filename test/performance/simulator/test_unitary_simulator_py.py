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


class TestPerformanceUnitarySimulatorPy(QiskitTestCase):
    """Profile performance of python unitary simulator."""

    def test_performance_unitary_simulator_py(self):
        """Profile randomly generated circuits.

        Writes profile results to <this_module>.prof as well as recording
        to the log file.

        number of circuits = 100.
        number of operations/circuit in [1, 40]
        number of qubits in [1, 5]
        """
        n_circuits = 100
        max_depth = 40
        max_qubits = 5
        profile = cProfile.Profile()
        random_circuits = RandomQasmGenerator(seed=self.seed,
                                              max_depth=max_depth,
                                              max_qubits=max_qubits)
        random_circuits.add_circuits(n_circuits, do_measure=False)
        qprogram = random_circuits.get_program()
        profile.enable()
        qprogram.execute(qprogram.get_circuit_names(),
                         backend=UnitarySimulatorPy())
        profile.disable()
        sout = io.StringIO()
        profile_stats = pstats.Stats(profile, stream=sout).sort_stats('cumulative')
        self.log.info('------- start profiling UnitarySimulatorPy -----------')
        profile_stats.print_stats()
        self.log.info(sout.getvalue())
        self.log.info('------- stop profiling UnitarySimulatorPy -----------')
        sout.close()
        profile.dump_stats(self.moduleName + '.prof')
