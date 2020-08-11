# -*- coding: utf-8 -*-

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

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, XGate, CXGate

from qiskit.test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError


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

        self.assertEqual(data[0], (HGate(), [qr[0]], []))
        self.assertEqual(data[1], (CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(data[2], (HGate(), [qr[1]], []))

    def test_count_gates(self):
        """Verify circuit.data can count inst/qarg/carg tuples."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.x(0)
        qc.h(1)
        qc.h(0)

        data = qc.data

        self.assertEqual(data.count((HGate(), [qr[0]], [])), 2)

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

        self.assertTrue((HGate(), [qr[0]], []) in qc.data)
        self.assertFalse((HGate(), [qr[1]], []) in qc.data)
        self.assertFalse((XGate(), [qr[0]], []) in qc.data)

    def test_index_gates(self):
        """Verify finding the index of a inst/qarg/carg tuple in circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        self.assertEqual(qc.data.index((HGate(), [qr[0]], [])), 0)
        self.assertEqual(qc.data.index((CXGate(), [qr[0], qr[1]], [])), 1)
        self.assertEqual(qc.data.index((HGate(), [qr[1]], [])), 2)

    def test_iter(self):
        """Verify circuit.data can behave as an iterator."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        iter_ = iter(qc.data)
        self.assertEqual(next(iter_), (HGate(), [qr[0]], []))
        self.assertEqual(next(iter_), (CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(next(iter_), (HGate(), [qr[1]], []))
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

        self.assertEqual(h_slice, [
            (HGate(), [qr[0]], []),
            (HGate(), [qr[1]], []),
            (HGate(), [qr[1]], []),
            (HGate(), [qr[0]], []),
        ])
        self.assertEqual(cx_slice, [
            (CXGate(), [qr[0], qr[1]], []),
            (CXGate(), [qr[1], qr[0]], []),
            (CXGate(), [qr[0], qr[1]], []),
        ])

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

        # pylint: disable=invalid-name
        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(
            repr(qc.data),
            "[({}, {}, {}), ({}, {}, {}), ({}, {}, {})]".format(
                repr(g1.instructions[0]), repr(g1.qargs[0]), repr(g1.cargs[0]),
                repr(g2.instructions[0]), repr(g2.qargs[0]), repr(g2.cargs[0]),
                repr(g3.instructions[0]), repr(g3.qargs[0]), repr(g3.cargs[0])))

    def test_str(self):
        """Verify circuit.data string representation."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        # pylint: disable=invalid-name
        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(
            str(qc.data),
            "[({}, {}, {}), ({}, {}, {}), ({}, {}, {})]".format(
                repr(g1.instructions[0]), repr(g1.qargs[0]), repr(g1.cargs[0]),
                repr(g2.instructions[0]), repr(g2.qargs[0]), repr(g2.cargs[0]),
                repr(g3.instructions[0]), repr(g3.qargs[0]), repr(g3.cargs[0])))

    def test_remove_gate(self):
        """Verify removing a gate via circuit.data.remove."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        qc.data.remove((HGate(), [qr[0]], []))

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
        self.assertEqual(last_h, (HGate(), [qr[1]], []))

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

        qc.data.append((HGate(), [qr[0]], []))
        qc.data.append((CXGate(), [0, 1], []))
        qc.data.append((HGate(), [qr[1]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(CircuitError, qc.data.append, (HGate(), [qr[0], qr[1]], []))
        self.assertRaises(CircuitError, qc.data.append, (HGate(), [], [qr[0]]))

    def test_insert_is_validated(self):
        """Verify inserting gates via circuit.data are broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.insert(0, (HGate(), [qr[0]], []))
        qc.data.insert(1, (CXGate(), [0, 1], []))
        qc.data.insert(2, (HGate(), [qr[1]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(CircuitError, qc.data.insert, 0, (HGate(), [qr[0], qr[1]], []))
        self.assertRaises(CircuitError, qc.data.insert, 0, (HGate(), [], [qr[0]]))

    def test_extend_is_validated(self):
        """Verify extending circuit.data is broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.extend([(HGate(), [qr[0]], []),
                        (CXGate(), [0, 1], []),
                        (HGate(), [qr[1]], [])])

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(CircuitError, qc.data.extend, [(HGate(), [qr[0], qr[1]], [])])
        self.assertRaises(CircuitError, qc.data.extend, [(HGate(), [], [qr[0]])])

    def test_setting_data_is_validated(self):
        """Verify setting circuit.data is broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data = [(HGate(), [qr[0]], []),
                   (CXGate(), [0, 1], []),
                   (HGate(), [qr[1]], [])]

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        with self.assertRaises(CircuitError):
            qc.data = [(HGate(), [qr[0], qr[1]], [])]
        with self.assertRaises(CircuitError):
            qc.data = [(HGate(), [], [qr[0]])]
