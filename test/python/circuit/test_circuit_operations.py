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
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase


class TestCircuitOperations(QiskitTestCase):
    """QuantumCircuit Operations tests."""

    def test_adding_self(self):
        """Test that qc += qc finishes, which can be prone to infinite while-loops.

        This can occur e.g. when a user tries
        >>> other_qc = qc
        >>> other_qc += qc  # or qc2.extend(qc)
        """
        qc = QuantumCircuit(1)
        qc.x(0)  # must contain at least one operation to end up in a infinite while-loop

        # attempt addition, times out if qc is added via reference
        qc += qc

        # finally, qc should contain two X gates
        self.assertEqual(['x', 'x'], [x[0].name for x in qc.data])

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
        it should raise a CircuitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(CircuitError, qc1.__add__, qc2)
        self.assertRaises(CircuitError, qc1.__add__, qcr3)

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
        it should raise a CircuitError.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")
        qc1 = QuantumCircuit(qr1)
        qc2 = QuantumCircuit(qr2)
        qcr3 = QuantumCircuit(cr1)

        self.assertRaises(CircuitError, qc1.__iadd__, qc2)
        self.assertRaises(CircuitError, qc1.__iadd__, qcr3)

    def test_measure_args_type_cohesion(self):
        """Test for proper args types for measure function.
        """
        quantum_reg = QuantumRegister(3)
        classical_reg_0 = ClassicalRegister(1)
        classical_reg_1 = ClassicalRegister(2)
        quantum_circuit = QuantumCircuit(quantum_reg, classical_reg_0,
                                         classical_reg_1)
        quantum_circuit.h(quantum_reg)

        with self.assertRaises(CircuitError) as ctx:
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

    def test_copy_copies_registers(self):
        """Test copy copies the registers not via reference."""
        qc = QuantumCircuit(1, 1)
        copied = qc.copy()

        copied.add_register(QuantumRegister(1, 'additional_q'))
        copied.add_register(ClassicalRegister(1, 'additional_c'))

        self.assertEqual(len(qc.qregs), 1)
        self.assertEqual(len(copied.qregs), 2)

        self.assertEqual(len(qc.cregs), 1)
        self.assertEqual(len(copied.cregs), 2)

    def test_measure_active(self):
        """Test measure_active
        Applies measurements only to non-idle qubits. Creates a ClassicalRegister of size equal to
        the amount of non-idle qubits to store the measured values.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(2, 'measure')

        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[2])
        circuit.measure_active()

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[2])
        expected.add_register(cr)
        expected.barrier()
        expected.measure([qr[0], qr[2]], [cr[0], cr[1]])

        self.assertEqual(expected, circuit)

    def test_measure_active_copy(self):
        """Test measure_active copy
        Applies measurements only to non-idle qubits. Creates a ClassicalRegister of size equal to
        the amount of non-idle qubits to store the measured values.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(2, 'measure')

        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[2])
        new_circuit = circuit.measure_active(inplace=False)

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[2])
        expected.add_register(cr)
        expected.barrier()
        expected.measure([qr[0], qr[2]], [cr[0], cr[1]])

        self.assertEqual(expected, new_circuit)
        self.assertFalse('measure' in circuit.count_ops().keys())

    def test_measure_active_repetition(self):
        """Test measure_active in a circuit with a 'measure' creg.
        measure_active should be aware that the creg 'measure' might exists.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, 'measure')

        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.measure_active()

        self.assertEqual(len(circuit.cregs), 2)  # Two cregs
        self.assertEqual(len(circuit.cregs[0]), 2)  # Both length 2
        self.assertEqual(len(circuit.cregs[1]), 2)

    def test_measure_all(self):
        """Test measure_all applies measurements to all qubits.
        Creates a ClassicalRegister of size equal to the total amount of qubits to
        store those measured values.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, 'meas')

        circuit = QuantumCircuit(qr)
        circuit.measure_all()

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, circuit)

    def test_measure_all_copy(self):
        """Test measure_all with inplace=False
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, 'meas')

        circuit = QuantumCircuit(qr)
        new_circuit = circuit.measure_all(inplace=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, new_circuit)
        self.assertFalse('measure' in circuit.count_ops().keys())

    def test_measure_all_repetition(self):
        """Test measure_all in a circuit with a 'measure' creg.
        measure_all should be aware that the creg 'measure' might exists.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, 'measure')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure_all()

        self.assertEqual(len(circuit.cregs), 2)  # Two cregs
        self.assertEqual(len(circuit.cregs[0]), 2)  # Both length 2
        self.assertEqual(len(circuit.cregs[1]), 2)

    def test_remove_final_measurements(self):
        """Test remove_final_measurements
        Removes all measurements at end of circuit.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, 'meas')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.remove_final_measurements()

        expected = QuantumCircuit(qr)

        self.assertEqual(expected, circuit)

    def test_remove_final_measurements_copy(self):
        """Test remove_final_measurements on copy
        Removes all measurements at end of circuit.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, 'meas')

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        new_circuit = circuit.remove_final_measurements(inplace=False)

        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)
        self.assertTrue('measure' in circuit.count_ops().keys())

    def test_remove_final_measurements_multiple_measures(self):
        """Test remove_final_measurements only removes measurements at the end of the circuit
        remove_final_measurements should not remove measurements in the beginning or middle of the
        circuit.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr[0], cr)
        circuit.h(0)
        circuit.measure(qr[0], cr)
        circuit.h(0)
        circuit.measure(qr[0], cr)
        circuit.remove_final_measurements()

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr)
        expected.h(0)
        expected.measure(qr[0], cr)
        expected.h(0)

        self.assertEqual(expected, circuit)

    def test_mirror(self):
        """Test mirror method reverses but does not invert."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc.x(0)
        qc.y(1)

        expected = QuantumCircuit(2, 2)
        expected.y(1)
        expected.x(0)
        expected.measure([0, 1], [0, 1])
        expected.cx(0, 1)
        expected.s(1)
        expected.h(0)

        self.assertEqual(qc.mirror(), expected)

    def test_repeat(self):
        """Test repeating the circuit works."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.h(0).c_if(cr, 1)

        with self.subTest('repeat 0 times'):
            rep = qc.repeat(0)
            self.assertEqual(rep, QuantumCircuit(qr, cr))

        with self.subTest('repeat 3 times'):
            inst = qc.to_instruction()
            ref = QuantumCircuit(qr, cr)
            for _ in range(3):
                ref.append(inst, ref.qubits, ref.clbits)

            rep = qc.repeat(3)
            self.assertEqual(rep, ref)


class TestCircuitBuilding(QiskitTestCase):
    """QuantumCircuit tests."""

    def test_append_dimension_mismatch(self):
        """Test appending to incompatible wires.
        """
