# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring
# pylint: disable=redefined-builtin

import cProfile
import io
import pstats
import unittest
import numpy as np

from qiskit import (qasm, unroll, QuantumCircuit,
                    QuantumRegister, ClassicalRegister, compile)
from qiskit.backends.local.unitary_simulator_py import UnitarySimulatorPy
from qiskit.qobj import Qobj, QobjItem, QobjExperiment, QobjConfig, QobjHeader
from ._random_qasm_generator import RandomQasmGenerator
from .common import QiskitTestCase


class LocalUnitarySimulatorTest(QiskitTestCase):
    """Test local unitary simulator."""

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/example.qasm')

    def test_unitary_simulator(self):
        """test generation of circuit unitary"""
        unroller = unroll.Unroller(
            qasm.Qasm(filename=self.qasm_filename).parse(),
            unroll.JsonBackend([]))
        circuit = unroller.execute()
        # strip measurements from circuit to avoid warnings
        circuit['instructions'] = [op for op in circuit['instructions']
                                   if op['name'] != 'measure']
        circuit = QobjExperiment.from_dict(circuit)
        circuit.config = QobjItem(coupling_map=None,
                                  basis_gates=None,
                                  layout=None,
                                  seed=self.seed)
        circuit.header.name = 'test'

        qobj = Qobj(qobj_id='unitary',
                    config=QobjConfig(shots=1,
                                      memory_slots=6,
                                      max_credits=None),
                    experiments=[circuit],
                    header=QobjHeader(
                        backend_name='local_unitary_simulator_py'))
        # numpy.savetxt currently prints complex numbers in a way
        # loadtxt can't read. To save file do,
        # fmtstr=['% .4g%+.4gj' for i in range(numCols)]
        # np.savetxt('example_unitary_matrix.dat', numpyMatrix, fmt=fmtstr,
        # delimiter=',')
        expected = np.loadtxt(self._get_resource_path('example_unitary_matrix.dat'),
                              dtype='complex', delimiter=',')

        result = UnitarySimulatorPy().run(qobj).result()
        self.assertTrue(np.allclose(result.get_unitary('test'),
                                    expected,
                                    rtol=1e-3))

    def test_two_unitary_simulator(self):
        """test running two circuits

        This test is similar to one in test_quantumprogram but doesn't use
        multiprocessing.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr)
        qc2.cx(qr[0], qr[1])
        backend = UnitarySimulatorPy()
        qobj = compile([qc1, qc2], backend=backend)
        job = backend.run(qobj)
        unitary1 = job.result().get_unitary(qc1)
        unitary2 = job.result().get_unitary(qc2)
        unitaryreal1 = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
                                 [0.5, 0.5, -0.5, -0.5],
                                 [0.5, -0.5, -0.5, 0.5]])
        unitaryreal2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1],
                                 [0., 0, 1, 0], [0, 1, 0, 0]])
        norm1 = np.trace(np.dot(np.transpose(np.conj(unitaryreal1)), unitary1))
        norm2 = np.trace(np.dot(np.transpose(np.conj(unitaryreal2)), unitary2))
        self.assertAlmostEqual(norm1, 4)
        self.assertAlmostEqual(norm2, 4)

    def profile_unitary_simulator(self):
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


if __name__ == '__main__':
    unittest.main()
