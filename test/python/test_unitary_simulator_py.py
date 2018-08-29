# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring
# pylint: disable=redefined-builtin

import unittest
import numpy as np

from qiskit import qasm, unroll
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import compile
from qiskit.backends.local.unitary_simulator_py import UnitarySimulatorPy
from qiskit.qobj import Qobj, QobjItem, QobjExperiment, QobjConfig, QobjHeader
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

    def test_local_unitary_simulator(self):
        """Test unitary simulator.

        If all correct should return the hxh and cx.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
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

    def test_local_unitary_simulator_single_thread(self):
        """test running two circuits

        Identical to the above test, but does not use multiprocessing.
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


if __name__ == '__main__':
    unittest.main()
