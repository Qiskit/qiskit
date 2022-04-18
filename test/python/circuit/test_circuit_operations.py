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
import numpy as np
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit.circuit import Gate, Instruction, Parameter, Measure
from qiskit.circuit.bit import Bit
from qiskit.circuit.classicalregister import Clbit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumcircuit import BitLocations
from qiskit.circuit.quantumregister import AncillaQubit, AncillaRegister, Qubit
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.standard_gates import SGate
from qiskit.quantum_info import Operator


@ddt
class TestCircuitOperations(QiskitTestCase):
    """QuantumCircuit Operations tests."""

    @data(0, 1, -1, -2)
    def test_append_resolves_integers(self, index):
        """Test that integer arguments to append are correctly resolved."""
        # We need to assume that appending ``Bit`` instances will always work, so we have something
        # to test against.
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        test.append(Measure(), [index], [index])
        expected = QuantumCircuit(qubits, clbits)
        expected.append(Measure(), [qubits[index]], [clbits[index]])
        self.assertEqual(test, expected)

    @data(np.int32(0), np.int8(-1), np.uint64(1))
    def test_append_resolves_numpy_integers(self, index):
        """Test that Numpy's integers can be used to reference qubits and clbits."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        test.append(Measure(), [index], [index])
        expected = QuantumCircuit(qubits, clbits)
        expected.append(Measure(), [qubits[int(index)]], [clbits[int(index)]])
        self.assertEqual(test, expected)

    @data(
        slice(0, 2),
        slice(None, 1),
        slice(1, None),
        slice(None, None),
        slice(0, 2, 2),
        slice(2, -1, -1),
        slice(1000, 1003),
    )
    def test_append_resolves_slices(self, index):
        """Test that slices can be used to reference qubits and clbits with the same semantics that
        they have on lists."""
        qregs = [QuantumRegister(2), QuantumRegister(1)]
        cregs = [ClassicalRegister(1), ClassicalRegister(2)]
        test = QuantumCircuit(*qregs, *cregs)
        test.append(Measure(), [index], [index])
        expected = QuantumCircuit(*qregs, *cregs)
        for qubit, clbit in zip(expected.qubits[index], expected.clbits[index]):
            expected.append(Measure(), [qubit], [clbit])
        self.assertEqual(test, expected)

    def test_append_resolves_scalar_numpy_array(self):
        """Test that size-1 Numpy arrays can be used to index arguments.  These arrays can be passed
        to ``int``, which means they sometimes might be involved in spurious casts."""
        test = QuantumCircuit(1, 1)
        test.append(Measure(), [np.array([0])], [np.array([0])])

        expected = QuantumCircuit(1, 1)
        expected.measure(0, 0)

        self.assertEqual(test, expected)

    @data([3], [-3], [0, 1, 3])
    def test_append_rejects_out_of_range_input(self, specifier):
        """Test that append rejects an integer that's out of range."""
        test = QuantumCircuit(2, 2)
        with self.subTest("qubit"), self.assertRaisesRegex(CircuitError, "out of range"):
            opaque = Instruction("opaque", len(specifier), 1, [])
            test.append(opaque, specifier, [0])
        with self.subTest("clbit"), self.assertRaisesRegex(CircuitError, "out of range"):
            opaque = Instruction("opaque", 1, len(specifier), [])
            test.append(opaque, [0], specifier)

    def test_append_rejects_bits_not_in_circuit(self):
        """Test that append rejects bits that are not in the circuit."""
        test = QuantumCircuit(2, 2)
        with self.subTest("qubit"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [Qubit()], [test.clbits[0]])
        with self.subTest("clbit"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [test.qubits[0]], [Clbit()])
        with self.subTest("qubit list"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [[test.qubits[0], Qubit()]], [test.clbits])
        with self.subTest("clbit list"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [test.qubits], [[test.clbits[0], Clbit()]])

    def test_append_rejects_bit_of_wrong_type(self):
        """Test that append rejects bits of the wrong type in an argument list."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        with self.subTest("c to q"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [clbits[0]], [clbits[1]])
        with self.subTest("q to c"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [qubits[0]], [qubits[1]])

        with self.subTest("none to q"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [Bit()], [clbits[0]])
        with self.subTest("none to c"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [qubits[0]], [Bit()])
        with self.subTest("none list"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [[qubits[0], Bit()]], [[clbits[0], Bit()]])

    @data(0.0, 1.0, 1.0 + 0.0j, "0")
    def test_append_rejects_wrong_types(self, specifier):
        """Test that various bad inputs are rejected, both given loose or in sublists."""
        test = QuantumCircuit(2, 2)
        # Use a default Instruction to be sure that there's not overridden broadcasting.
        opaque = Instruction("opaque", 1, 1, [])
        with self.subTest("q"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [specifier], [0])
        with self.subTest("c"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [0], [specifier])
        with self.subTest("q list"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [[specifier]], [[0]])
        with self.subTest("c list"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [[0]], [[specifier]])

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
        self.assertEqual(["x", "x"], [x[0].name for x in qc.data])

    def test_combine_circuit_common(self):
        """Test combining two circuits with same registers (inplace=False)."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        new_circuit = qc1.combine(qc2)

        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_combine_circuit_common_plus(self):
        """Test combining two circuits with same registers (as plus)."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(new_circuit, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

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
        """Test extending a circuit with same registers (in place add)."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc1.extend(qc2)
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 2})  # changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_extend_circuit_iadd(self):
        """Test extending a circuit with same registers (in place add)."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc1 += qc2
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 2})  # changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_extend_circuit_fail(self):
        """Test extending a circuit fails if registers incompatible.

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

    def test_extend_circuit_adds_qubits(self):
        """Test extending a circuits with differing registers adds the qubits."""
        qr = QuantumRegister(1, "q")
        qc = QuantumCircuit(qr)
        empty = QuantumCircuit()
        empty.extend(qc)

        self.assertListEqual(empty.qubits, qr[:])

    def test_compose_circuit(self):
        """Test composing two circuits"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc3 = qc1.compose(qc2)
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc3, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compose_circuit_and(self):
        """Test composing two circuits using & operator"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc3 = qc1 & qc2
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc3, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compose_circuit_iand(self):
        """Test composing circuits using &= operator (in place)"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc1 &= qc2
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 2})  # changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compose_circuit_fail_circ_size(self):
        """Test composing circuit fails when number of wires in circuit is not enough"""
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(4)

        # Creating our circuits
        qc1 = QuantumCircuit(qr1)
        qc1.x(0)
        qc1.h(1)

        qc2 = QuantumCircuit(qr2)
        qc2.h([1, 2])
        qc2.cx(2, 3)

        # Composing will fail because qc2 requires 4 wires
        self.assertRaises(CircuitError, qc1.compose, qc2)

    def test_compose_circuit_fail_arg_size(self):
        """Test composing circuit fails when arg size does not match number of wires"""
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)

        qc1 = QuantumCircuit(qr1)
        qc1.h(0)

        qc2 = QuantumCircuit(qr2)
        qc2.cx(0, 1)

        self.assertRaises(CircuitError, qc1.compose, qc2, qubits=[0])

    def test_tensor_circuit(self):
        """Test tensoring two circuits"""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 1)

        qc2.h(0)
        qc2.measure(0, 0)
        qc1.measure(0, 0)

        qc3 = qc1.tensor(qc2)
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc3, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc2.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc1.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_tensor_circuit_xor(self):
        """Test tensoring two circuits using ^ operator"""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 1)

        qc2.h(0)
        qc2.measure(0, 0)
        qc1.measure(0, 0)

        qc3 = qc1 ^ qc2
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc3, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc2.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc1.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_tensor_circuit_ixor(self):
        """Test tensoring two circuits using ^= operator"""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 1)

        qc2.h(0)
        qc2.measure(0, 0)
        qc1.measure(0, 0)

        qc1 ^= qc2
        backend = BasicAer.get_backend("qasm_simulator")
        shots = 1024
        result = execute(qc1, backend=backend, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 2})  # changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_measure_args_type_cohesion(self):
        """Test for proper args types for measure function."""
        quantum_reg = QuantumRegister(3)
        classical_reg_0 = ClassicalRegister(1)
        classical_reg_1 = ClassicalRegister(2)
        quantum_circuit = QuantumCircuit(quantum_reg, classical_reg_0, classical_reg_1)
        quantum_circuit.h(quantum_reg)

        with self.assertRaises(CircuitError) as ctx:
            quantum_circuit.measure(quantum_reg, classical_reg_1)
        self.assertEqual(ctx.exception.message, "register size error")

    def test_copy_circuit(self):
        """Test copy method makes a copy"""
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

        copied.add_register(QuantumRegister(1, "additional_q"))
        copied.add_register(ClassicalRegister(1, "additional_c"))

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
        cr = ClassicalRegister(2, "measure")

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
        cr = ClassicalRegister(2, "measure")

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
        self.assertFalse("measure" in circuit.count_ops().keys())

    def test_measure_active_repetition(self):
        """Test measure_active in a circuit with a 'measure' creg.
        measure_active should be aware that the creg 'measure' might exists.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "measure")

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
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr)
        circuit.measure_all()

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, circuit)

    def test_measure_all_not_add_bits_equal(self):
        """Test measure_all applies measurements to all qubits.
        Does not create a new ClassicalRegister if the existing one is big enough.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure_all(add_bits=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, circuit)

    def test_measure_all_not_add_bits_bigger(self):
        """Test measure_all applies measurements to all qubits.
        Does not create a new ClassicalRegister if the existing one is big enough.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(3, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure_all(add_bits=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr[0:2])

        self.assertEqual(expected, circuit)

    def test_measure_all_not_add_bits_smaller(self):
        """Test measure_all applies measurements to all qubits.
        Raises an error if there are not enough classical bits to store the measurements.
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)

        with self.assertRaisesRegex(CircuitError, "The number of classical bits"):
            circuit.measure_all(add_bits=False)

    def test_measure_all_copy(self):
        """Test measure_all with inplace=False"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr)
        new_circuit = circuit.measure_all(inplace=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, new_circuit)
        self.assertFalse("measure" in circuit.count_ops().keys())

    def test_measure_all_repetition(self):
        """Test measure_all in a circuit with a 'measure' creg.
        measure_all should be aware that the creg 'measure' might exists.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "measure")

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
        cr = ClassicalRegister(2, "meas")

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
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        new_circuit = circuit.remove_final_measurements(inplace=False)

        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)
        self.assertTrue("measure" in circuit.count_ops().keys())

    def test_remove_final_measurements_copy_with_parameters(self):
        """Test remove_final_measurements doesn't corrupt ParameterTable

        See https://github.com/Qiskit/qiskit-terra/issues/6108 for more details
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")
        theta = Parameter("theta")

        circuit = QuantumCircuit(qr, cr)
        circuit.rz(theta, qr)
        circuit.measure(qr, cr)
        circuit.remove_final_measurements()
        copy = circuit.copy()

        self.assertEqual(copy, circuit)

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

    def test_remove_final_measurements_5802(self):
        """Test remove_final_measurements removes classical bits
        https://github.com/Qiskit/qiskit-terra/issues/5802.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.remove_final_measurements()

        self.assertEqual(circuit.cregs, [])
        self.assertEqual(circuit.clbits, [])

    def test_remove_final_measurements_7089(self):
        """Test remove_final_measurements removes resulting unused registers
        even if not all bits were measured into.
        https://github.com/Qiskit/qiskit-terra/issues/7089.
        """
        circuit = QuantumCircuit(2, 5)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        circuit.remove_final_measurements(inplace=True)

        self.assertEqual(circuit.cregs, [])
        self.assertEqual(circuit.clbits, [])

    def test_remove_final_measurements_bit_locations(self):
        """Test remove_final_measurements properly recalculates clbit indicies
        and preserves order of remaining cregs and clbits.
        """
        c0 = ClassicalRegister(1)
        c1_0 = Clbit()
        c2 = ClassicalRegister(1)
        c3 = ClassicalRegister(1)

        # add an individual bit that's not in any register of this circuit
        circuit = QuantumCircuit(QuantumRegister(1), c0, [c1_0], c2, c3)

        circuit.measure(0, c1_0)
        circuit.measure(0, c2[0])

        # assert cregs and clbits before measure removal
        self.assertEqual(circuit.cregs, [c0, c2, c3])
        self.assertEqual(circuit.clbits, [c0[0], c1_0, c2[0], c3[0]])

        # assert clbit indices prior to measure removal
        self.assertEqual(circuit.find_bit(c0[0]), BitLocations(0, [(c0, 0)]))
        self.assertEqual(circuit.find_bit(c1_0), BitLocations(1, []))
        self.assertEqual(circuit.find_bit(c2[0]), BitLocations(2, [(c2, 0)]))
        self.assertEqual(circuit.find_bit(c3[0]), BitLocations(3, [(c3, 0)]))

        circuit.remove_final_measurements()

        # after measure removal, creg c2 should be gone, as should lone bit c1_0
        # and c0 should still come before c3
        self.assertEqual(circuit.cregs, [c0, c3])
        self.assertEqual(circuit.clbits, [c0[0], c3[0]])

        # there should be no gaps in clbit indices
        # e.g. c3[0] is now the second clbit
        self.assertEqual(circuit.find_bit(c0[0]), BitLocations(0, [(c0, 0)]))
        self.assertEqual(circuit.find_bit(c3[0]), BitLocations(1, [(c3, 0)]))

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

        with self.subTest("repeat 0 times"):
            rep = qc.repeat(0)
            self.assertEqual(rep, QuantumCircuit(qr, cr))

        with self.subTest("repeat 3 times"):
            inst = qc.to_instruction()
            ref = QuantumCircuit(qr, cr)
            for _ in range(3):
                ref.append(inst, ref.qubits, ref.clbits)
            rep = qc.repeat(3)
            self.assertEqual(rep, ref)

    @data(0, 1, 4)
    def test_repeat_global_phase(self, num):
        """Test the global phase is properly handled upon repeat."""
        phase = 0.123
        qc = QuantumCircuit(1, global_phase=phase)
        expected = np.exp(1j * phase * num) * np.identity(2)
        np.testing.assert_array_almost_equal(Operator(qc.repeat(num)).data, expected)

    def test_bind_global_phase(self):
        """Test binding global phase."""
        x = Parameter("x")
        circuit = QuantumCircuit(1, global_phase=x)
        self.assertEqual(circuit.parameters, {x})

        bound = circuit.bind_parameters({x: 2})
        self.assertEqual(bound.global_phase, 2)
        self.assertEqual(bound.parameters, set())

    def test_bind_parameter_in_phase_and_gate(self):
        """Test binding a parameter present in the global phase and the gates."""
        x = Parameter("x")
        circuit = QuantumCircuit(1, global_phase=x)
        circuit.rx(x, 0)
        self.assertEqual(circuit.parameters, {x})

        ref = QuantumCircuit(1, global_phase=2)
        ref.rx(2, 0)

        bound = circuit.bind_parameters({x: 2})
        self.assertEqual(bound, ref)
        self.assertEqual(bound.parameters, set())

    def test_power(self):
        """Test taking the circuit to a power works."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(0.2, 1)

        gate = qc.to_gate()

        with self.subTest("power(int >= 0) equals repeat"):
            self.assertEqual(qc.power(4), qc.repeat(4))

        with self.subTest("explicit matrix power"):
            self.assertEqual(qc.power(4, matrix_power=True).data[0][0], gate.power(4))

        with self.subTest("float power"):
            self.assertEqual(qc.power(1.23).data[0][0], gate.power(1.23))

        with self.subTest("negative power"):
            self.assertEqual(qc.power(-2).data[0][0], gate.power(-2))

    def test_power_parameterized_circuit(self):
        """Test taking a parameterized circuit to a power."""
        theta = Parameter("th")
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(theta, 1)

        with self.subTest("power(int >= 0) equals repeat"):
            self.assertEqual(qc.power(4), qc.repeat(4))

        with self.subTest("cannot to matrix power if parameterized"):
            with self.assertRaises(CircuitError):
                _ = qc.power(0.5)

    def test_control(self):
        """Test controlling the circuit."""
        qc = QuantumCircuit(2, name="my_qc")
        qc.cry(0.2, 0, 1)

        c_qc = qc.control()
        with self.subTest("return type is circuit"):
            self.assertIsInstance(c_qc, QuantumCircuit)

        with self.subTest("test name"):
            self.assertEqual(c_qc.name, "c_my_qc")

        with self.subTest("repeated control"):
            cc_qc = c_qc.control()
            self.assertEqual(cc_qc.num_qubits, c_qc.num_qubits + 1)

        with self.subTest("controlled circuit has same parameter"):
            param = Parameter("p")
            qc.rx(param, 0)
            c_qc = qc.control()
            self.assertEqual(qc.parameters, c_qc.parameters)

        with self.subTest("non-unitary operation raises"):
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

        c_qc = qc.control(2, ctrl_state="10")

        cgate = qc.to_gate().control(2, ctrl_state="10")
        ref = QuantumCircuit(*c_qc.qregs)
        ref.append(cgate, ref.qubits)

        self.assertEqual(ref, c_qc)

    @data("gate", "instruction")
    def test_repeat_appended_type(self, subtype):
        """Test repeat appends Gate if circuit contains only gates and Instructions otherwise."""
        sub = QuantumCircuit(2)
        sub.x(0)

        if subtype == "gate":
            sub = sub.to_gate()
        else:
            sub = sub.to_instruction()

        qc = QuantumCircuit(2)
        qc.append(sub, [0, 1])
        rep = qc.repeat(3)

        if subtype == "gate":
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
        qc.global_phase = -1

        expected = QuantumCircuit(3, 2)
        expected.h(2)
        expected.s(1)
        expected.cx(2, 1)
        expected.measure(2, 0)
        expected.x(2)
        expected.y(1)
        expected.global_phase = -1

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_boxed(self):
        """Test reversing order of bits in a hierarchical circuit."""
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
        qr1 = QuantumRegister(3, "a")
        qr2 = QuantumRegister(2, "b")
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

    def test_compare_two_equal_circuits(self):
        """Test to compare that 2 circuits are equal."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)

        self.assertTrue(qc1 == qc2)

    def test_compare_two_different_circuits(self):
        """Test to compare that 2 circuits are different."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)

        qc2 = QuantumCircuit(2, 2)
        qc2.x(0)

        self.assertFalse(qc1 == qc2)

    def test_compare_a_circuit_with_none(self):
        """Test to compare that a circuit is different to None."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)

        qc2 = None

        self.assertFalse(qc1 == qc2)

    def test_overlapped_add_bits_and_add_register(self):
        """Test add registers whose bits have already been added by add_bits."""
        qc = QuantumCircuit()
        for bit_type, reg_type in (
            [Qubit, QuantumRegister],
            [Clbit, ClassicalRegister],
            [AncillaQubit, AncillaRegister],
        ):
            bits = [bit_type() for _ in range(10)]
            reg = reg_type(bits=bits)
            qc.add_bits(bits)
            qc.add_register(reg)

        self.assertEqual(qc.num_qubits, 20)
        self.assertEqual(qc.num_clbits, 10)
        self.assertEqual(qc.num_ancillas, 10)

    def test_overlapped_add_register_and_add_register(self):
        """Test add registers whose bits have already been added by add_register."""
        qc = QuantumCircuit()
        for bit_type, reg_type in (
            [Qubit, QuantumRegister],
            [Clbit, ClassicalRegister],
            [AncillaQubit, AncillaRegister],
        ):
            bits = [bit_type() for _ in range(10)]
            reg1 = reg_type(bits=bits)
            reg2 = reg_type(bits=bits)
            qc.add_register(reg1)
            qc.add_register(reg2)

        self.assertEqual(qc.num_qubits, 20)
        self.assertEqual(qc.num_clbits, 10)
        self.assertEqual(qc.num_ancillas, 10)

    def test_deprecated_measure_function(self):
        """Test that the deprecated version of the loose 'measure' function works correctly."""
        from qiskit.circuit.measure import measure

        test = QuantumCircuit(1, 1)
        with self.assertWarnsRegex(DeprecationWarning, r".*Qiskit Terra 0\.19.*"):
            measure(test, 0, 0)

        expected = QuantumCircuit(1, 1)
        expected.measure(0, 0)

        self.assertEqual(test, expected)

    def test_deprecated_reset_function(self):
        """Test that the deprecated version of the loose 'reset' function works correctly."""
        from qiskit.circuit.reset import reset

        test = QuantumCircuit(1, 1)
        with self.assertWarnsRegex(DeprecationWarning, r".*Qiskit Terra 0\.19.*"):
            reset(test, 0)

        expected = QuantumCircuit(1, 1)
        expected.reset(0)

        self.assertEqual(test, expected)


class TestCircuitPrivateOperations(QiskitTestCase):
    """Direct tests of some of the private methods of QuantumCircuit.  These do not represent
    functionality that we want to expose to users, but there are some cases where private methods
    are used internally (similar to "protected" access in .NET or "friend" access in C++), and we
    want to make sure they work in those cases."""

    def test_previous_instruction_in_scope_failures(self):
        """Test the failure paths of the peek and pop methods for retrieving the most recent
        instruction in a scope."""
        test = QuantumCircuit(1, 1)
        with self.assertRaisesRegex(CircuitError, r"This circuit contains no instructions\."):
            test._peek_previous_instruction_in_scope()
        with self.assertRaisesRegex(CircuitError, r"This circuit contains no instructions\."):
            test._pop_previous_instruction_in_scope()
        with test.for_loop(range(2)):
            with self.assertRaisesRegex(CircuitError, r"This scope contains no instructions\."):
                test._peek_previous_instruction_in_scope()
            with self.assertRaisesRegex(CircuitError, r"This scope contains no instructions\."):
                test._pop_previous_instruction_in_scope()

    def test_pop_previous_instruction_removes_parameters(self):
        """Test that the private "pop instruction" method removes parameters from the parameter
        table if that instruction is the only instance."""
        x, y = Parameter("x"), Parameter("y")
        test = QuantumCircuit(1, 1)
        test.rx(y, 0)
        last_instructions = test.u(x, y, 0, 0)
        self.assertEqual({x, y}, set(test.parameters))

        instruction, _, _ = test._pop_previous_instruction_in_scope()
        self.assertEqual(list(last_instructions), [instruction])
        self.assertEqual({y}, set(test.parameters))
