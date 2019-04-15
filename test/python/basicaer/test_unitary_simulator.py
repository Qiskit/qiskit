# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for unitary simulator."""

import unittest

import numpy as np

from qiskit import execute
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.test import ReferenceCircuits
from qiskit.test import providers


class BasicAerUnitarySimulatorPyTest(providers.BackendTestCase):
    """Test BasicAer unitary simulator."""

    backend_cls = UnitarySimulatorPy
    circuit = ReferenceCircuits.bell_no_measure()

    def test_basicaer_unitary_simulator_py(self):
        """Test unitary simulator."""
        circuits = self._test_circuits()
        job = execute(circuits, backend=self.backend)
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
