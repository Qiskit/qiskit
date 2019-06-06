# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's QuantumCircuit class."""

from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.test import QiskitTestCase


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
        backend = BasicAer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed_simulator=78).result()
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
        backend = BasicAer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots,
                         seed_simulator=78).result()
        counts = result.get_counts()
        target = {'11': shots}
        self.assertEqual(counts, target)

    def test_combine_circuit_fail(self):
        """Test combining two circuits fails if registers incompatible.

        If two circuits have same name register of different size or type
        it should raise a QiskitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(QiskitError, qc1.__add__, qc2)
        self.assertRaises(QiskitError, qc1.__add__, qcr3)

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
        backend = BasicAer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots,
                         seed_simulator=78).result()
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
        backend = BasicAer.get_backend('qasm_simulator')
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots,
                         seed_simulator=78).result()
        counts = result.get_counts()
        target = {'11': shots}
        self.assertEqual(counts, target)

    def test_extend_circuit_fail(self):
        """Test extending a circuits fails if registers incompatible.

        If two circuits have same name register of different size or type
        it should raise a QiskitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(QiskitError, qc1.__iadd__, qc2)
        self.assertRaises(QiskitError, qc1.__iadd__, qcr3)

    def test_measure_args_type_cohesion(self):
        """Test for proper args types for measure function.
        """
        quantum_reg = QuantumRegister(3)
        classical_reg_0 = ClassicalRegister(1)
        classical_reg_1 = ClassicalRegister(2)
        quantum_circuit = QuantumCircuit(quantum_reg, classical_reg_0,
                                         classical_reg_1)
        quantum_circuit.h(quantum_reg)

        with self.assertRaises(QiskitError) as ctx:
            quantum_circuit.measure(quantum_reg, classical_reg_1)
        self.assertEqual(ctx.exception.message,
                         'register size error')

    def test_copy_circuit(self):
        """ Test copy method makes a copy"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        self.assertEqual(qc, qc.copy())


class TestCircuitBuilding(QiskitTestCase):
    """QuantumCircuit tests."""

    def test_append_dimension_mismatch(self):
        """Test appending to incompatible wires.
        """
