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
                                      max_credits=10),
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
        """Test unitary simulator."""
        circuits = self._test_circuits()
        backend = UnitarySimulatorPy()
        qobj = compile(circuits, backend=backend)
        job = backend.run(qobj)
        sim_unitaries = [job.result().get_unitary(circ) for circ in circuits]
        reference_unitaries = self._reference_unitaries()
        norms = [np.trace(np.dot(np.transpose(np.conj(target)), actual))
                 for target, actual in zip(reference_unitaries, sim_unitaries)]
        for norm in norms:
            self.assertAlmostEqual(norm, 8)

    def _test_circuits(self):
        """Return test circuits for unitary simulator"""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc3 = QuantumCircuit(qr, cr)
        qc4 = QuantumCircuit(qr, cr)
        qc5 = QuantumCircuit(qr, cr)
        # Test circuit 1:  HxHxH
        qc1.h(qr)
        # Test circuit 2: IxCX
        qc2.cx(qr[0], qr[1])
        # Test circuit 3:  CXxY
        qc3.y(qr[0])
        qc3.cx(qr[1], qr[2])
        # Test circuit 4: (CX.I).(IxCX).(IxIxX)
        qc4.h(qr[0])
        qc4.cx(qr[0], qr[1])
        qc4.cx(qr[1], qr[2])
        # Test circuit 5 (X.Z)x(Z.Y)x(Y.X)
        qc5.x(qr[0])
        qc5.y(qr[0])
        qc5.y(qr[1])
        qc5.z(qr[1])
        qc5.z(qr[2])
        qc5.x(qr[2])
        return [qc1, qc2, qc3, qc4, qc5]

    def _reference_unitaries(self):
        """Return reference unitaries for test circuits"""
        # Gate matrices
        gate_h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        gate_x = np.array([[0, 1], [1, 0]])
        gate_y = np.array([[0, -1j], [1j, 0]])
        gate_z = np.array([[1, 0], [0, -1]])
        gate_cx = np.array([[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0., 0, 1, 0],
                            [0, 1, 0, 0]])
        # Unitary matrices
        target_unitary1 = np.kron(np.kron(gate_h, gate_h), gate_h)
        target_unitary2 = np.kron(np.eye(2), gate_cx)
        target_unitary3 = np.kron(gate_cx, gate_y)
        target_unitary4 = np.dot(np.kron(gate_cx, np.eye(2)),
                                 np.dot(np.kron(np.eye(2), gate_cx),
                                        np.kron(np.eye(4), gate_h)))
        target_unitary5 = np.kron(np.kron(np.dot(gate_x, gate_z),
                                          np.dot(gate_z, gate_y)),
                                  np.dot(gate_y, gate_x))
        return [target_unitary1, target_unitary2, target_unitary3,
                target_unitary4, target_unitary5]


if __name__ == '__main__':
    unittest.main()
