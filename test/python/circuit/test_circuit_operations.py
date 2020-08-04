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

from ddt import ddt, data
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.standard_gates import SGate


@ddt
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

    def test_reverse(self):
        """Test reverse method reverses but does not invert."""
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

        self.assertEqual(qc.reverse_ops(), expected)

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

    def test_power(self):
        """Test taking the circuit to a power works."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(0.2, 1)

        gate = qc.to_gate()

        with self.subTest('power(int >= 0) equals repeat'):
            self.assertEqual(qc.power(4), qc.repeat(4))

        with self.subTest('explicit matrix power'):
            self.assertEqual(qc.power(4, matrix_power=True).data[0][0],
                             gate.power(4))

        with self.subTest('float power'):
            self.assertEqual(qc.power(1.23).data[0][0], gate.power(1.23))

        with self.subTest('negative power'):
            self.assertEqual(qc.power(-2).data[0][0], gate.power(-2))

    def test_power_parameterized_circuit(self):
        """Test taking a parameterized circuit to a power."""
        theta = Parameter('th')
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(theta, 1)

        with self.subTest('power(int >= 0) equals repeat'):
            self.assertEqual(qc.power(4), qc.repeat(4))

        with self.subTest('cannot to matrix power if parameterized'):
            with self.assertRaises(CircuitError):
                _ = qc.power(0.5)

    def test_control(self):
        """Test controlling the circuit."""
        qc = QuantumCircuit(2, name='my_qc')
        qc.cry(0.2, 0, 1)

        c_qc = qc.control()
        with self.subTest('return type is circuit'):
            self.assertIsInstance(c_qc, QuantumCircuit)

        with self.subTest('test name'):
            self.assertEqual(c_qc.name, 'c_my_qc')

        with self.subTest('repeated control'):
            cc_qc = c_qc.control()
            self.assertEqual(cc_qc.num_qubits, c_qc.num_qubits + 1)

        with self.subTest('controlled circuit has same parameter'):
            param = Parameter('p')
            qc.rx(param, 0)
            c_qc = qc.control()
            self.assertEqual(qc.parameters, c_qc.parameters)

        with self.subTest('non-unitary operation raises'):
            qc.reset(0)
            with self.assertRaises(CircuitError):
                _ = qc.control()

    def test_control_implementation(self):
        """Run a test case for controlling the circuit, which should use ``Gate.control``."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cry(0.2, 0, 1)
        qc.t(0)
        qc.append(SGate().control(2), [0, 1, 2])
        qc.iswap(2, 0)

        c_qc = qc.control(2, ctrl_state='10')

        cgate = qc.to_gate().control(2, ctrl_state='10')
        ref = QuantumCircuit(*c_qc.qregs)
        ref.append(cgate, ref.qubits)

        self.assertEqual(ref, c_qc)

    @data('gate', 'instruction')
    def test_repeat_appended_type(self, subtype):
        """Test repeat appends Gate if circuit contains only gates and Instructions otherwise."""
        sub = QuantumCircuit(2)
        sub.x(0)

        if subtype == 'gate':
            sub = sub.to_gate()
        else:
            sub = sub.to_instruction()

        qc = QuantumCircuit(2)
        qc.append(sub, [0, 1])
        rep = qc.repeat(3)

        if subtype == 'gate':
            self.assertTrue(all(isinstance(op[0], Gate) for op in rep.data))
        else:
            self.assertTrue(all(isinstance(op[0], Instruction) for op in rep.data))

    def test_reverse_bits(self):
        """Test reversing order of bits."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.measure(0, 1)
        qc.x(0)
        qc.y(1)

        expected = QuantumCircuit(3, 2)
        expected.h(2)
        expected.s(1)
        expected.cx(2, 1)
        expected.measure(2, 0)
        expected.x(2)
        expected.y(1)

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_boxed(self):
        """Test reversing order of bits in a hierarchiecal circuit."""
        wide_cx = QuantumCircuit(3)
        wide_cx.cx(0, 1)
        wide_cx.cx(1, 2)

        wide_cxg = wide_cx.to_gate()
        cx_box = QuantumCircuit(3)
        cx_box.append(wide_cxg, [0, 1, 2])

        expected = QuantumCircuit(3)
        expected.cx(2, 1)
        expected.cx(1, 0)

        self.assertEqual(cx_box.reverse_bits().decompose(), expected)
        self.assertEqual(cx_box.decompose().reverse_bits(), expected)

        # box one more layer to be safe.
        cx_box_g = cx_box.to_gate()
        cx_box_box = QuantumCircuit(4)
        cx_box_box.append(cx_box_g, [0, 1, 2])
        cx_box_box.cx(0, 3)

        expected2 = QuantumCircuit(4)
        expected2.cx(3, 2)
        expected2.cx(2, 1)
        expected2.cx(3, 0)

        self.assertEqual(cx_box_box.reverse_bits().decompose().decompose(), expected2)

    def test_reverse_bits_with_registers(self):
        """Test reversing order of bits when registers are present."""
        qr1 = QuantumRegister(3, 'a')
        qr2 = QuantumRegister(2, 'b')
        qc = QuantumCircuit(qr1, qr2)
        qc.h(qr1[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
        qc.cx(qr2[0], qr2[1])

        expected = QuantumCircuit(qr2, qr1)
        expected.h(qr1[2])
        expected.cx(qr1[2], qr1[1])
        expected.cx(qr1[1], qr1[0])
        expected.cx(qr1[0], qr2[1])
        expected.cx(qr2[1], qr2[0])

        self.assertEqual(qc.reverse_bits(), expected)

    def test_cnot_alias(self):
        """Test that the cnot method alias adds a cx gate."""
        qc = QuantumCircuit(2)
        qc.cnot(0, 1)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(qc, expected)

    def test_inverse(self):
        """Test inverse circuit."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr, global_phase=0.5)
        qc.h(0)
        qc.barrier(qr)
        qc.t(1)

        expected = QuantumCircuit(qr)
        expected.tdg(1)
        expected.barrier(qr)
        expected.h(0)
        expected.global_phase = -0.5
        self.assertEqual(qc.inverse(), expected)


class TestCircuitBuilding(QiskitTestCase):
    """QuantumCircuit tests."""

    def test_append_dimension_mismatch(self):
        """Test appending to incompatible wires.
        """
