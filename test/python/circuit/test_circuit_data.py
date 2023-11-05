# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test operations on circuit.data."""
import ddt
from qiskit._accelerate.quantum_circuit import CircuitData

from qiskit.circuit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    Parameter,
    CircuitInstruction,
    Operation,
    Qubit,
)
from qiskit.circuit.library import HGate, XGate, CXGate, RXGate

from qiskit.test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError


@ddt.ddt
class TestQuantumCircuitData(QiskitTestCase):
    """CircuitData (Rust) operation tests."""

    @ddt.data(
        slice(0, 5, 1),  # Get everything.
        slice(-1, -6, -1),  # Get everything, reversed.
        slice(0, 4, 1),  # Get subslice.
        slice(0, 5, 2),  # Get every other.
        slice(-1, -6, -2),  # Get every other, reversed.
        slice(2, 2, 1),  # Get nothing.
        slice(2, 3, 1),  # Get at index 2.
        slice(4, 10, 1),  # Get index 4 to end, using excessive upper bound.
        slice(5, 0, -2),  # Get every other, reversed, excluding index 0.
        slice(-10, -5, 1),  # Get nothing.
        slice(0, 10, 1),  # Get everything.
    )
    def test_getitem_slice(self, sli):
        """Test that __getitem__ with slice is equivalent to that of list."""
        qr = QuantumRegister(5)
        data_list = [
            CircuitInstruction(XGate(), [qr[0]], []),
            CircuitInstruction(XGate(), [qr[1]], []),
            CircuitInstruction(XGate(), [qr[2]], []),
            CircuitInstruction(XGate(), [qr[3]], []),
            CircuitInstruction(XGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)
        self.assertEqual(data[sli], data_list[sli])

    @ddt.data(
        slice(0, 5, 1),  # Delete everything.
        slice(-1, -6, -1),  # Delete everything, reversed.
        slice(0, 4, 1),  # Delete subslice.
        slice(0, 5, 2),  # Delete every other.
        slice(-1, -6, -2),  # Delete every other, reversed.
        slice(2, 2, 1),  # Delete nothing.
        slice(2, 3, 1),  # Delete at index 2.
        slice(4, 10, 1),  # Delete index 4 to end, excessive upper bound.
        slice(5, 0, -2),  # Delete every other, reversed, excluding index 0.
        slice(-10, -5, 1),  # Delete nothing.
        slice(0, 10, 1),  # Delete everything, excessive upper bound.
    )
    def test_delitem_slice(self, sli):
        """Test that __delitem__ with slice is equivalent to that of list."""
        qr = QuantumRegister(5)
        data_list = [
            CircuitInstruction(XGate(), [qr[0]], []),
            CircuitInstruction(XGate(), [qr[1]], []),
            CircuitInstruction(XGate(), [qr[2]], []),
            CircuitInstruction(XGate(), [qr[3]], []),
            CircuitInstruction(XGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)

        del data_list[sli]
        del data[sli]
        if data_list[sli] != data[sli]:
            print(f"data_list: {data_list}")
            print(f"data: {list(data)}")

        self.assertEqual(data[sli], data_list[sli])

    @ddt.data(
        (slice(0, 5, 1), 5),  # Replace entire slice.
        (slice(-1, -6, -1), 5),  # Replace entire slice, reversed.
        (slice(0, 4, 1), 4),  # Replace subslice.
        (slice(0, 4, 1), 10),  # Replace subslice with bigger sequence.
        (slice(0, 5, 2), 3),  # Replace every other.
        (slice(-1, -6, -2), 3),  # Replace every other, reversed.
        (slice(2, 2, 1), 1),  # Insert at index 2.
        (slice(2, 3, 1), 1),  # Replace at index 2.
        (slice(2, 3, 1), 10),  # Replace at index 2 with bigger sequence.
        (slice(4, 10, 1), 2),  # Replace index 4 with bigger sequence, excessive upper bound.
        (slice(5, 10, 1), 10),  # Append sequence.
        (slice(4, 0, -1), 4),  # Replace subslice at end, reversed.
    )
    @ddt.unpack
    def test_setitem_slice(self, sli, value_length):
        """Test that __setitem__ with slice is equivalent to that of list."""
        reg_size = 20
        assert value_length <= reg_size
        qr = QuantumRegister(reg_size)
        default_bit = Qubit()
        data_list = [
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
        ]
        data = CircuitData(qubits=list(qr) + [default_bit], data=data_list)

        value = [CircuitInstruction(XGate(), [qr[i]]) for i in range(value_length)]
        data_list[sli] = value
        data[sli] = value
        self.assertEqual(data, data_list)

    @ddt.data(
        (slice(0, 5, 2), 2),  # Replace smaller, with gaps.
        (slice(0, 5, 2), 4),  # Replace larger, with gaps.
        (slice(4, 0, -1), 10),  # Replace larger, reversed.
        (slice(-1, -6, -1), 6),  # Replace larger, reversed, negative notation.
        (slice(4, 3, -1), 10),  # Replace at index 4 with bigger sequence, reversed.
    )
    @ddt.unpack
    def test_setitem_slice_negative(self, sli, value_length):
        """Test that __setitem__ with slice is equivalent to that of list."""
        reg_size = 20
        assert value_length <= reg_size
        qr = QuantumRegister(reg_size)
        default_bit = Qubit()
        data_list = [
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
        ]
        data = CircuitData(qubits=list(qr) + [default_bit], data=data_list)

        value = [CircuitInstruction(XGate(), [qr[i]]) for i in range(value_length)]
        with self.assertRaises(ValueError):
            data_list[sli] = value
        with self.assertRaises(ValueError):
            data[sli] = value
        self.assertEqual(data, data_list)

    def test_unregistered_bit_error_new(self):
        """Test using foreign bits is not allowed."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        with self.assertRaisesRegex(KeyError, "not been added to this circuit"):
            CircuitData(qr, cr, [CircuitInstruction(XGate(), [Qubit()], [])])

    def test_unregistered_bit_error_append(self):
        """Test using foreign bits is not allowed."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        data = CircuitData(qr, cr)
        with self.assertRaisesRegex(KeyError, "not been added to this circuit"):
            qr_foreign = QuantumRegister(1)
            data.append(CircuitInstruction(XGate(), [qr_foreign[0]], []))

    def test_unregistered_bit_error_set(self):
        """Test using foreign bits is not allowed."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        data = CircuitData(qr, cr, [CircuitInstruction(XGate(), [qr[0]], [])])
        with self.assertRaisesRegex(KeyError, "not been added to this circuit"):
            qr_foreign = QuantumRegister(1)
            data[0] = CircuitInstruction(XGate(), [qr_foreign[0]], [])


class TestQuantumCircuitInstructionData(QiskitTestCase):
    """QuantumCircuit.data operation tests."""

    # N.B. Most of the cases here are not expected use cases of circuit.data
    # but are included as tests to maintain compatability with the previous
    # list interface of circuit.data.

    def test_getitem_by_insertion_order(self):
        """Verify one can get circuit.data items in insertion order."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        data = qc.data

        self.assertEqual(data[0], CircuitInstruction(HGate(), [qr[0]], []))
        self.assertEqual(data[1], CircuitInstruction(CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(data[2], CircuitInstruction(HGate(), [qr[1]], []))

    def test_count_gates(self):
        """Verify circuit.data can count inst/qarg/carg tuples."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.x(0)
        qc.h(1)
        qc.h(0)

        data = qc.data

        self.assertEqual(data.count(CircuitInstruction(HGate(), [qr[0]], [])), 2)

    def test_len(self):
        """Verify finding the length of circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        self.assertEqual(len(qc.data), 0)
        qc.h(0)
        self.assertEqual(len(qc.data), 1)
        qc.cx(0, 1)
        self.assertEqual(len(qc.data), 2)

    def test_contains(self):
        """Verify checking if a inst/qarg/carg tuple is in circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)

        self.assertTrue(CircuitInstruction(HGate(), [qr[0]], []) in qc.data)
        self.assertFalse(CircuitInstruction(HGate(), [qr[1]], []) in qc.data)
        self.assertFalse(CircuitInstruction(XGate(), [qr[0]], []) in qc.data)

    def test_index_gates(self):
        """Verify finding the index of a inst/qarg/carg tuple in circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        self.assertEqual(qc.data.index(CircuitInstruction(HGate(), [qr[0]], [])), 0)
        self.assertEqual(qc.data.index(CircuitInstruction(CXGate(), [qr[0], qr[1]], [])), 1)
        self.assertEqual(qc.data.index(CircuitInstruction(HGate(), [qr[1]], [])), 2)

    def test_iter(self):
        """Verify circuit.data can behave as an iterator."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        iter_ = iter(qc.data)
        self.assertEqual(next(iter_), CircuitInstruction(HGate(), [qr[0]], []))
        self.assertEqual(next(iter_), CircuitInstruction(CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(next(iter_), CircuitInstruction(HGate(), [qr[1]], []))
        self.assertRaises(StopIteration, next, iter_)

    def test_slice(self):
        """Verify circuit.data can be sliced."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.cx(1, 0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)

        h_slice = qc.data[::2]
        cx_slice = qc.data[1:-1:2]

        self.assertEqual(
            h_slice,
            [
                CircuitInstruction(HGate(), [qr[0]], []),
                CircuitInstruction(HGate(), [qr[1]], []),
                CircuitInstruction(HGate(), [qr[1]], []),
                CircuitInstruction(HGate(), [qr[0]], []),
            ],
        )
        self.assertEqual(
            cx_slice,
            [
                CircuitInstruction(CXGate(), [qr[0], qr[1]], []),
                CircuitInstruction(CXGate(), [qr[1], qr[0]], []),
                CircuitInstruction(CXGate(), [qr[0], qr[1]], []),
            ],
        )

    def test_copy(self):
        """Verify one can create a shallow copy circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        data_copy = qc.data.copy()

        self.assertEqual(data_copy, qc.data)

    def test_repr(self):
        """Verify circuit.data repr."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(repr(qc.data), repr([g1[0], g2[0], g3[0]]))

    def test_str(self):
        """Verify circuit.data string representation."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(str(qc.data), str([g1[0], g2[0], g3[0]]))

    def test_remove_gate(self):
        """Verify removing a gate via circuit.data.remove."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        qc.data.remove(CircuitInstruction(HGate(), [qr[0]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.h(0)

        self.assertEqual(qc, expected_qc)

    def test_del(self):
        """Verify removing a gate via circuit.data.delattr."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        del qc.data[0]

        expected_qc = QuantumCircuit(qr)

        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.h(0)

        self.assertEqual(qc, expected_qc)

    def test_pop_gate(self):
        """Verify removing a gate via circuit.data.pop."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        last_h = qc.data.pop()

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)

        self.assertEqual(qc, expected_qc)
        self.assertEqual(last_h, CircuitInstruction(HGate(), [qr[1]], []))

    def test_clear_gates(self):
        """Verify emptying a circuit via circuit.data.clear."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        qc.data.clear()

        self.assertEqual(qc.data, [])

    def test_reverse_gates(self):
        """Verify reversing a circuit via circuit.data.reverse."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        qc.data.reverse()

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(1)
        expected_qc.cx(0, 1)
        expected_qc.h(0)

        self.assertEqual(qc, expected_qc)

    def test_repeating_a_circuit_via_mul(self):
        """Verify repeating a circuit via circuit.data.__mul__."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        qc.data *= 2

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

    def test_add_radd(self):
        """Verify adding lists of gates via circuit.data.__add__."""
        qr = QuantumRegister(2)
        qc1 = QuantumCircuit(qr)

        qc1.h(0)
        qc1.cx(0, 1)
        qc1.h(1)

        qc2 = QuantumCircuit(qr)
        qc2.cz(0, 1)

        qc1.data += qc2.data

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.cz(0, 1)

        self.assertEqual(qc1, expected_qc)

    def test_append_is_validated(self):
        """Verify appended gates via circuit.data are broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.append(CircuitInstruction(HGate(), [qr[0]], []))
        qc.data.append(CircuitInstruction(CXGate(), [0, 1], []))
        qc.data.append(CircuitInstruction(HGate(), [qr[1]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(
            CircuitError, qc.data.append, CircuitInstruction(HGate(), [qr[0], qr[1]], [])
        )
        self.assertRaises(CircuitError, qc.data.append, CircuitInstruction(HGate(), [], [qr[0]]))

    def test_insert_is_validated(self):
        """Verify inserting gates via circuit.data are broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.insert(0, CircuitInstruction(HGate(), [qr[0]], []))
        qc.data.insert(1, CircuitInstruction(CXGate(), [0, 1], []))
        qc.data.insert(2, CircuitInstruction(HGate(), [qr[1]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(
            CircuitError, qc.data.insert, 0, CircuitInstruction(HGate(), [qr[0], qr[1]], [])
        )
        self.assertRaises(CircuitError, qc.data.insert, 0, CircuitInstruction(HGate(), [], [qr[0]]))

    def test_extend_is_validated(self):
        """Verify extending circuit.data is broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.extend(
            [
                CircuitInstruction(HGate(), [qr[0]], []),
                CircuitInstruction(CXGate(), [0, 1], []),
                CircuitInstruction(HGate(), [qr[1]], []),
            ]
        )

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(
            CircuitError, qc.data.extend, [CircuitInstruction(HGate(), [qr[0], qr[1]], [])]
        )
        self.assertRaises(CircuitError, qc.data.extend, [CircuitInstruction(HGate(), [], [qr[0]])])

    def test_setting_data_is_validated(self):
        """Verify setting circuit.data is broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data = [
            CircuitInstruction(HGate(), [qr[0]], []),
            CircuitInstruction(CXGate(), [0, 1], []),
            CircuitInstruction(HGate(), [qr[1]], []),
        ]

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        with self.assertRaises(CircuitError):
            qc.data = [CircuitInstruction(HGate(), [qr[0], qr[1]], [])]
        with self.assertRaises(CircuitError):
            qc.data = [CircuitInstruction(HGate(), [], [qr[0]])]

    def test_setting_data_coerces_to_instruction(self):
        """Verify that the `to_instruction` coercion also happens when setting data using the legacy
        3-tuple format."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        class NotAnInstruction:
            # pylint: disable=missing-class-docstring,missing-function-docstring
            def to_instruction(self):
                return CXGate()

        qc.data[0] = (NotAnInstruction(), qc.qubits, [])

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(qc, expected)

    def test_setting_data_allows_operation(self):
        """Test that using the legacy 3-tuple setter to the data allows arbitrary `Operation`
        classes to be used, not just `Instruction`."""

        class MyOp(Operation):
            # pylint: disable=missing-class-docstring,missing-function-docstring

            @property
            def name(self):
                return "myop"

            @property
            def num_qubits(self):
                return 2

            @property
            def num_clbits(self):
                return 0

            def __eq__(self, other):
                return isinstance(other, MyOp)

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.data[0] = (MyOp(), qc.qubits, [])

        expected = QuantumCircuit(2)
        expected.append(MyOp(), [0, 1], [])
        self.assertEqual(qc, expected)

    def test_param_gate_instance(self):
        """Verify that the same Parameter gate instance is not being used in
        multiple circuits."""
        a, b = Parameter("a"), Parameter("b")
        rx = RXGate(a)
        qc0, qc1 = QuantumCircuit(1), QuantumCircuit(1)
        qc0.append(rx, [0])
        qc1.append(rx, [0])
        qc0.assign_parameters({a: b}, inplace=True)
        qc0_instance = next(iter(qc0._parameter_table[b]))[0]
        qc1_instance = next(iter(qc1._parameter_table[a]))[0]
        self.assertNotEqual(qc0_instance, qc1_instance)
