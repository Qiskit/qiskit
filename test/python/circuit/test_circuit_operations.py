# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's QuantumCircuit class."""

import os
import tempfile
import unittest

import qiskit.extensions.simulator
from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QISKitError
from qiskit.quantum_info import state_fidelity
from ..common import QiskitTestCase


class TestCircuitOperations(QiskitTestCase):
    """QuantumCircuit Operations tests."""

    def test_combine_circuit_common(self):
        """Test combining two circuits with same registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_combine_circuit_different(self):
        """Test combining two circuits with different registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        qc1.x(qr)
        qc2 = QuantumCircuit(qr, cr)
        qc2.measure(qr, cr)
        new_circuit = qc1 + qc2
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'11': shots}
        self.assertEqual(counts, target)

    def test_combine_circuit_fail(self):
        """Test combining two circuits fails if registers incompatible.

        If two circuits have same name register of different size or type
        it should raise a QISKitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(QISKitError, qc1.__add__, qc2)
        self.assertRaises(QISKitError, qc1.__add__, qcr3)

    def test_combine_circuit_extension_instructions(self):
        """Test combining circuits containing barrier, initializer, snapshot
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qc1.initialize(desired_vector, qr)
        qc1.barrier()
        qc2 = QuantumCircuit(qr, cr)
        qc2.snapshot(slot='1')
        qc2.measure(qr, cr)
        new_circuit = qc1 + qc2
        backend = Aer.get_backend('qasm_simulator_py')
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed=78).result()
        snapshot_vectors = result.get_snapshot()
        fidelity = state_fidelity(snapshot_vectors[0], desired_vector)
        self.assertGreater(fidelity, 0.99)

        counts = result.get_counts()
        target = {'00': shots/4, '01': shots/4, '10': shots/4, '11': shots/4}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_extend_circuit(self):
        """Test extending a circuit with same registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc1 += qc2
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_extend_circuit_different_registers(self):
        """Test extending a circuit with different registers.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        qc1.x(qr)
        qc2 = QuantumCircuit(qr, cr)
        qc2.measure(qr, cr)
        qc1 += qc2
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed=78).result()
        counts = result.get_counts()
        target = {'11': shots}
        self.assertEqual(counts, target)

    def test_extend_circuit_fail(self):
        """Test extending a circuits fails if registers incompatible.

        If two circuits have same name register of different size or type
        it should raise a QISKitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(QISKitError, qc1.__iadd__, qc2)
        self.assertRaises(QISKitError, qc1.__iadd__, qcr3)

    def test_extend_circuit_extension_instructions(self):
        """Test extending circuits containing barrier, initializer, snapshot
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr)
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qc1.initialize(desired_vector, qr)
        qc1.barrier()
        qc2 = QuantumCircuit(qr, cr)
        qc2.snapshot(slot='1')
        qc2.measure(qr, cr)
        qc1 += qc2
        backend = Aer.get_backend('qasm_simulator_py')
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed=78).result()

        snapshot_vectors = result.get_snapshot('1')
        fidelity = state_fidelity(snapshot_vectors[0], desired_vector)
        self.assertGreater(fidelity, 0.99)

        counts = result.get_counts()
        target = {'00': shots/4, '01': shots/4, '10': shots/4, '11': shots/4}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)
