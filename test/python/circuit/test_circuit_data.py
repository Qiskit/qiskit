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

from ddt import ddt, data, unpack

from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, XGate, CXGate, RXGate

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

        qc_data = qc.data

        self.assertEqual(qc_data[0], (HGate(), [qr[0]], []))
        self.assertEqual(qc_data[1], (CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(qc_data[2], (HGate(), [qr[1]], []))

    def test_count_gates(self):
        """Verify circuit.data can count inst/qarg/carg tuples."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.x(0)
        qc.h(1)
        qc.h(0)

        qc_data = qc.data

        self.assertEqual(qc_data.count((HGate(), [qr[0]], [])), 2)

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

        self.assertEqual(
            h_slice,
            [
                (HGate(), [qr[0]], []),
                (HGate(), [qr[1]], []),
                (HGate(), [qr[1]], []),
                (HGate(), [qr[0]], []),
            ],
        )
        self.assertEqual(
            cx_slice,
            [
                (CXGate(), [qr[0], qr[1]], []),
                (CXGate(), [qr[1], qr[0]], []),
                (CXGate(), [qr[0], qr[1]], []),
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

        self.assertEqual(
            repr(qc.data),
            "[({}, {}, {}), ({}, {}, {}), ({}, {}, {})]".format(
                repr(g1.instructions[0]),
                repr(g1.qargs[0]),
                repr(g1.cargs[0]),
                repr(g2.instructions[0]),
                repr(g2.qargs[0]),
                repr(g2.cargs[0]),
                repr(g3.instructions[0]),
                repr(g3.qargs[0]),
                repr(g3.cargs[0]),
            ),
        )

    def test_str(self):
        """Verify circuit.data string representation."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(
            str(qc.data),
            "[({}, {}, {}), ({}, {}, {}), ({}, {}, {})]".format(
                repr(g1.instructions[0]),
                repr(g1.qargs[0]),
                repr(g1.cargs[0]),
                repr(g2.instructions[0]),
                repr(g2.qargs[0]),
                repr(g2.cargs[0]),
                repr(g3.instructions[0]),
                repr(g3.qargs[0]),
                repr(g3.cargs[0]),
            ),
        )

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

        qc.data.extend([(HGate(), [qr[0]], []), (CXGate(), [0, 1], []), (HGate(), [qr[1]], [])])

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

        qc.data = [(HGate(), [qr[0]], []), (CXGate(), [0, 1], []), (HGate(), [qr[1]], [])]

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        with self.assertRaises(CircuitError):
            qc.data = [(HGate(), [qr[0], qr[1]], [])]
        with self.assertRaises(CircuitError):
            qc.data = [(HGate(), [], [qr[0]])]

    def test_param_gate_instance(self):
        """Verify that the same Parameter gate instance is not being used in
        multiple circuits."""
        a, b = Parameter("a"), Parameter("b")
        rx = RXGate(a)
        qc0, qc1 = QuantumCircuit(1), QuantumCircuit(1)
        qc0.append(rx, [0])
        qc1.append(rx, [0])
        qc0.assign_parameters({a: b}, inplace=True)
        qc0_instance = qc0._parameter_table[b][0][0]
        qc1_instance = qc1._parameter_table[a][0][0]
        self.assertNotEqual(qc0_instance, qc1_instance)


@ddt
class TestQuantumCircuitRegisterData(QiskitTestCase):
    """QuantumCircuit.{q,c}regs operation tests."""

    # N.B. Most of the cases here are not expected use cases of circuit.{q,c}regs
    # but are included as tests to maintain compatability with the previous
    # list interface of circuit.{q,c}regs.

    @data(["qregs", QuantumRegister], ["cregs", ClassicalRegister])
    @unpack
    def test_getitem_by_insertion_order(self, reg_prop, reg_type):
        """Verify we can fetch registers by their insertion order."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")
        reg4 = reg_type(3, "reg4")

        qc = QuantumCircuit(reg1, reg2, reg3, reg4)
        circ_regs = getattr(qc, reg_prop)

        # __eq__
        self.assertEqual(circ_regs, [reg1, reg2, reg3, reg4])

        # __len__
        self.assertEqual(len(circ_regs), 4)

        # __getitem__
        self.assertEqual(circ_regs[0], reg1)
        self.assertEqual(circ_regs[1], reg2)
        self.assertEqual(circ_regs[2], reg3)
        self.assertEqual(circ_regs[3], reg4)

        with self.assertRaises(IndexError):
            _ = circ_regs[4]

        # __contains__
        self.assertTrue(reg1 in circ_regs)
        self.assertTrue(reg2 in circ_regs)
        self.assertTrue(reg3 in circ_regs)
        self.assertTrue(reg4 in circ_regs)

        reg5 = reg_type(5, "reg5")
        self.assertFalse(reg5 in circ_regs)

        # .index
        self.assertEqual(circ_regs.index(reg1), 0)
        self.assertEqual(circ_regs.index(reg2), 1)
        self.assertEqual(circ_regs.index(reg3), 2)
        self.assertEqual(circ_regs.index(reg4), 3)

        with self.assertRaises(ValueError):
            _ = circ_regs.index(reg5)

        # __iter__
        iter_ = iter(circ_regs)
        self.assertEqual(next(iter_), reg1)
        self.assertEqual(next(iter_), reg2)
        self.assertEqual(next(iter_), reg3)
        self.assertEqual(next(iter_), reg4)
        with self.assertRaises(StopIteration):
            next(iter_)

        # slice
        self.assertEqual(circ_regs[::2], [reg1, reg3])
        self.assertEqual(circ_regs[1::2], [reg2, reg4])

        self.assertEqual(circ_regs[-1:1:-1], [reg4, reg3])
        self.assertEqual(circ_regs[1:-1:-1], [])

        # copy
        self.assertEqual(circ_regs.copy(), circ_regs)
        self.assertTrue(circ_regs.copy() is not circ_regs)

        # repr
        self.assertEqual(repr(circ_regs), repr([reg1, reg2, reg3, reg4]))

        # str
        self.assertEqual(str(circ_regs), str([reg1, reg2, reg3, reg4]))

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_remove_gate(self, reg_prop, bit_prop, reg_type):
        """Verify removing a register via circuit.{q,c}regs.remove."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        getattr(qc, reg_prop).remove(reg2)

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg3])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_del(self, reg_prop, bit_prop, reg_type):
        """Verify removing a register via circuit.{q,c}regs.delattr."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        del getattr(qc, reg_prop)[1]

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg3])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_pop_reg(self, reg_prop, bit_prop, reg_type):
        """Verify removing a register via circuit.{q,c}regs.pop."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        last_reg = getattr(qc, reg_prop).pop()

        self.assertEqual(reg3, last_reg)
        self.assertEqual(getattr(qc, reg_prop), [reg1, reg2])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_clear_regs(self, reg_prop, bit_prop, reg_type):
        """Verify emptying circuit registers via circuit.{q,c}regs.clear."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        getattr(qc, reg_prop).clear()

        self.assertEqual(getattr(qc, reg_prop), [])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_reverse_regs(self, reg_prop, bit_prop, reg_type):
        """Verify reversing circuit registers via circuit.{q,c}regs.reverse."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        getattr(qc, reg_prop).reverse()

        self.assertEqual(getattr(qc, reg_prop), [reg3, reg2, reg1])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

    @data(["qregs", QuantumRegister], ["cregs", ClassicalRegister])
    @unpack
    def test_repeating_a_circuit_via_mul(self, reg_prop, reg_type):
        """Verify repeating registers via circuit.{q,c}regs.__mul__ raises a CircuitError."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        with self.assertRaisesRegex(CircuitError, r"register name.*already exists"):
            setattr(qc, reg_prop, getattr(qc, reg_prop) * 3)

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_add_radd(self, reg_prop, bit_prop, reg_type):
        """Verify adding lists of registers via circuit.{q,c}regs.__add__."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        new_reg1 = reg_type(3, "new_reg1")
        new_reg2 = reg_type(3, "new_reg2")
        new_regs = [new_reg1, new_reg2]

        setattr(qc, reg_prop, getattr(qc, reg_prop) + new_regs)

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg2, reg3, new_reg1, new_reg2])
        self.assertEqual(
            getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:] + new_reg1[:] + new_reg2[:]
        )

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_append_is_validated(self, reg_prop, bit_prop, reg_type):
        """Verify appended regs via circuit.{q,c}regs are validated."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        with self.assertRaisesRegex(CircuitError, r"expected a register"):
            getattr(qc, reg_prop).append("not a register")

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg2, reg3])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

        new_reg = reg_type(3, "new_reg")

        getattr(qc, reg_prop).append(new_reg)

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg2, reg3, new_reg])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:] + new_reg[:])

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_insert_is_validated(self, reg_prop, bit_prop, reg_type):
        """Verify inserting regs via circuit.{q,c}regs are validated."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        with self.assertRaisesRegex(CircuitError, r"expected a register"):
            getattr(qc, reg_prop).insert(0, "not a register")

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg2, reg3])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:])

        new_reg = reg_type(3, "new_reg")

        getattr(qc, reg_prop).insert(0, new_reg)

        self.assertEqual(getattr(qc, reg_prop), [new_reg, reg1, reg2, reg3])
        self.assertEqual(getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:] + new_reg[:])

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_extend_is_validated(self, reg_prop, bit_prop, reg_type):
        """Verify extending registers via circuit.{q,c}regs is validated."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        with self.assertRaisesRegex(CircuitError, r"expected a register"):
            getattr(qc, reg_prop).extend(["not a register"])

        new_reg1 = reg_type(3, "new_reg1")
        new_reg2 = reg_type(3, "new_reg2")
        new_regs = [new_reg1, new_reg2]

        getattr(qc, reg_prop).extend(new_regs)

        self.assertEqual(getattr(qc, reg_prop), [reg1, reg2, reg3, new_reg1, new_reg2])
        self.assertEqual(
            getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:] + new_reg1[:] + new_reg2[:]
        )

    @data(["qregs", "qubits", QuantumRegister], ["cregs", "clbits", ClassicalRegister])
    @unpack
    def test_setting_data_is_validated(self, reg_prop, bit_prop, reg_type):
        """Verify setting circuit.data is broadcast and validated."""
        reg1 = reg_type(3, "reg1")
        reg2 = reg_type(3, "reg2")
        reg3 = reg_type(3, "reg3")

        qc = QuantumCircuit(reg1, reg2, reg3)

        with self.assertRaisesRegex(CircuitError, r"expected a register"):
            setattr(qc, reg_prop, ["not a register"])

        new_reg1 = reg_type(3, "new_reg1")
        new_reg2 = reg_type(3, "new_reg2")

        setattr(qc, reg_prop, [new_reg1, reg2, new_reg2])

        self.assertEqual(getattr(qc, reg_prop), [new_reg1, reg2, new_reg2])
        self.assertEqual(
            getattr(qc, bit_prop), reg1[:] + reg2[:] + reg3[:] + new_reg1[:] + new_reg2[:]
        )
