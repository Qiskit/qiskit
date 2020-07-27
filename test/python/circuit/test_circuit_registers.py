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

import numpy as np

from qiskit.circuit import (
    QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit, Qubit, Clbit, AncillaQubit,
    Gate
)
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase


class TestCircuitRegisters(QiskitTestCase):
    """QuantumCircuit Registers tests."""

    def test_qregs(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(10, "q")
        self.assertEqual(qr1.name, "q")
        self.assertEqual(qr1.size, 10)
        self.assertEqual(type(qr1), QuantumRegister)

    def test_cregs(self):
        """Test getting classical registers from circuit.
        """
        cr1 = ClassicalRegister(10, "c")
        self.assertEqual(cr1.name, "c")
        self.assertEqual(cr1.size, 10)
        self.assertEqual(type(cr1), ClassicalRegister)

    def test_aregs(self):
        """Test getting ancilla registers from circuit.
        """
        ar1 = AncillaRegister(10, "a")
        self.assertEqual(ar1.name, "a")
        self.assertEqual(ar1.size, 10)
        self.assertEqual(type(ar1), AncillaRegister)

    def test_qarg_negative_size(self):
        """Test attempt to create a negative size QuantumRegister.
        """
        self.assertRaises(CircuitError, QuantumRegister, -1)

    def test_qarg_string_size(self):
        """Test attempt to create a non-integer size QuantumRegister.
        """
        self.assertRaises(CircuitError, QuantumRegister, 'string')

    def test_qarg_numpy_int_size(self):
        """Test castable to integer size QuantumRegister.
        """
        np_int = np.dtype('int').type(10)
        qr1 = QuantumRegister(np_int, "q")
        self.assertEqual(qr1.name, "q")
        self.assertEqual(qr1.size, 10)
        self.assertEqual(type(qr1), QuantumRegister)

    def test_register_int_types(self):
        """Test attempt to pass different types of integer as indices
        of QuantumRegister and ClassicalRegister
        """
        ints = [int(2), np.int(2), np.int32(2), np.int64(2)]
        for index in ints:
            with self.subTest(index=index):
                qr = QuantumRegister(4)
                cr = ClassicalRegister(4)
                self.assertEqual(qr[index], qr[2])
                self.assertEqual(cr[index], cr[2])

    def test_numpy_array_of_registers(self):
        """Test numpy array of Registers .
        See https://github.com/Qiskit/qiskit-terra/issues/1898
        """
        qrs = [QuantumRegister(2, name='q%s' % i) for i in range(5)]
        qreg_array = np.array([], dtype=object, ndmin=1)
        qreg_array = np.append(qreg_array, qrs)

        expected = [qrs[0][0], qrs[0][1],
                    qrs[1][0], qrs[1][1],
                    qrs[2][0], qrs[2][1],
                    qrs[3][0], qrs[3][1],
                    qrs[4][0], qrs[4][1]]

        self.assertEqual(len(qreg_array), 10)
        self.assertEqual(qreg_array.tolist(), expected)

    def test_negative_index(self):
        """Test indexing from the back
        """
        qr1 = QuantumRegister(10, "q")
        cr1 = ClassicalRegister(10, "c")
        self.assertEqual(qr1[-1], qr1[9])
        self.assertEqual(qr1[-3:-1], [qr1[7], qr1[8]])
        self.assertEqual(len(cr1[0:-2]), 8)
        self.assertEqual(qr1[[-1, -3, -5]], [qr1[9], qr1[7], qr1[5]])

    def test_reg_equal(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")

        self.assertEqual(qr1, qr1)
        self.assertNotEqual(qr1, qr2)
        self.assertNotEqual(qr1, cr1)

    def test_qubits(self):
        """Test qubits() method.
        """
        qr1 = QuantumRegister(1, "q1")
        cr1 = ClassicalRegister(3, "c1")
        qr2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(qr2, cr1, qr1)

        qubits = qc.qubits

        self.assertEqual(qubits[0], qr2[0])
        self.assertEqual(qubits[1], qr2[1])
        self.assertEqual(qubits[2], qr1[0])

    def test_clbits(self):
        """Test clbits() method.
        """
        qr1 = QuantumRegister(1, "q1")
        cr1 = ClassicalRegister(2, "c1")
        qr2 = QuantumRegister(2, "q2")
        cr2 = ClassicalRegister(1, "c2")
        qc = QuantumCircuit(qr2, cr2, qr1, cr1)

        clbits = qc.clbits

        self.assertEqual(clbits[0], cr2[0])
        self.assertEqual(clbits[1], cr1[0])
        self.assertEqual(clbits[2], cr1[1])

    def test_ancillas(self):
        """Test ancillas() method.
        """
        qr1 = QuantumRegister(1, "q1")
        cr1 = ClassicalRegister(2, "c1")
        ar1 = AncillaRegister(2, "a1")
        qr2 = QuantumRegister(2, "q2")
        cr2 = ClassicalRegister(1, "c2")
        ar2 = AncillaRegister(1, "a2")
        qc = QuantumCircuit(qr2, cr2, ar2, qr1, cr1, ar1)

        ancillas = qc.ancillas

        self.assertEqual(qc.num_ancillas, 3)

        self.assertEqual(ancillas[0], ar2[0])
        self.assertEqual(ancillas[1], ar1[0])
        self.assertEqual(ancillas[2], ar1[1])

    def test_ancilla_qubit(self):
        """Test ancilla type and that it can be accessed as ordinary qubit."""
        qr, ar = QuantumRegister(2), AncillaRegister(2)
        qc = QuantumCircuit(qr, ar)

        with self.subTest('num ancillas and qubits'):
            self.assertEqual(qc.num_ancillas, 2)
            self.assertEqual(qc.num_qubits, 4)

        with self.subTest('ancilla is a qubit'):
            for ancilla in qc.ancillas:
                self.assertIsInstance(ancilla, AncillaQubit)
                self.assertIsInstance(ancilla, Qubit)

        with self.subTest('qubit is not an ancilla'):
            action_qubits = [qubit for qubit in qc.qubits if not isinstance(qubit, AncillaQubit)]
            self.assertEqual(len(action_qubits), 2)

    def test_qregs_circuit(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        q_regs = qc.qregs
        self.assertEqual(len(q_regs), 2)
        self.assertEqual(q_regs[0], qr1)
        self.assertEqual(q_regs[1], qr2)

    def test_cregs_circuit(self):
        """Test getting classical registers from circuit.
        """
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(3)
        qc = QuantumCircuit(cr1, cr2, cr3)
        c_regs = qc.cregs
        self.assertEqual(len(c_regs), 3)
        self.assertEqual(c_regs[0], cr1)
        self.assertEqual(c_regs[1], cr2)

    def test_basic_slice(self):
        """simple slice test"""
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        self.assertEqual(len(qr[0:3]), 3)
        self.assertEqual(len(cr[0:3]), 3)

    def test_apply_gate_to_slice(self):
        """test applying gate to register slice"""
        sli = slice(0, 9, 2)
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0:9:2])
        for i, index in enumerate(range(*sli.indices(sli.stop))):
            self.assertEqual(qc.data[i][1][0].index, index)

    def test_apply_barrier_to_slice(self):
        """test applying barrier to register slice"""
        num_qubits = 10
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        qc = QuantumCircuit(qr, cr)
        qc.barrier(qr)
        # barrier works a little different than normal gates for expansion
        # test full register
        self.assertEqual(len(qc.data), 1)
        self.assertEqual(qc.data[0][0].name, 'barrier')
        self.assertEqual(len(qc.data[0][1]), num_qubits)
        for i, bit in enumerate(qc.data[0][1]):
            self.assertEqual(bit.index, i)
        # test slice
        num_qubits = 2
        qc = QuantumCircuit(qr, cr)
        qc.barrier(qr[0:num_qubits])
        self.log.info(qc.qasm())
        self.assertEqual(len(qc.data), 1)
        self.assertEqual(qc.data[0][0].name, 'barrier')
        self.assertEqual(len(qc.data[0][1]), num_qubits)
        for i in range(num_qubits):
            self.assertEqual(qc.data[0][1][i].index, i)

    def test_apply_ccx_to_slice(self):
        """test applying ccx to register slice"""
        qcontrol = QuantumRegister(10)
        qcontrol2 = QuantumRegister(10)
        qtarget = QuantumRegister(5)
        qtarget2 = QuantumRegister(10)
        qc = QuantumCircuit(qcontrol, qtarget)
        # test slice with skip and full register target
        qc.ccx(qcontrol[1::2], qcontrol[0::2], qtarget)
        self.assertEqual(len(qc.data), 5)
        for i, ictl, (gate, qargs, _) in zip(range(len(qc.data)), range(0, 10, 2), qc.data):
            self.assertEqual(gate.name, 'ccx')
            self.assertEqual(len(qargs), 3)
            self.assertIn(qargs[0].index, [ictl, ictl + 1])
            self.assertIn(qargs[1].index, [ictl, ictl + 1])
            self.assertEqual(qargs[2].index, i)
        # test decrementing slice
        qc = QuantumCircuit(qcontrol, qtarget)
        qc.ccx(qcontrol[2:0:-1], qcontrol[4:6], qtarget[0:2])
        self.assertEqual(len(qc.data), 2)
        for (gate, qargs, _), ictl1, ictl2, itgt in zip(qc.data, range(2, 0, -1),
                                                        range(4, 6), range(0, 2)):
            self.assertEqual(gate.name, 'ccx')
            self.assertEqual(len(qargs), 3)
            self.assertEqual(qargs[0].index, ictl1)
            self.assertEqual(qargs[1].index, ictl2)
            self.assertEqual(qargs[2].index, itgt)
        # test register expansion in ccx
        qc = QuantumCircuit(qcontrol, qcontrol2, qtarget2)
        qc.ccx(qcontrol, qcontrol2, qtarget2)
        for i, (gate, qargs, _) in enumerate(qc.data):
            self.assertEqual(gate.name, 'ccx')
            self.assertEqual(len(qargs), 3)
            self.assertEqual(qargs[0].index, i)
            self.assertEqual(qargs[1].index, i)
            self.assertEqual(qargs[2].index, i)

    def test_cswap_on_slice(self):
        """test applying cswap to register slice"""
        qr1 = QuantumRegister(10)
        qr2 = QuantumRegister(5)
        qc = QuantumCircuit(qr1, qr2)
        qc.cswap(qr2[3::-1], qr1[1:9:2], qr1[2:9:2])
        qc.cswap(qr2[0], qr1[1], qr1[2])
        qc.cswap([qr2[0]], [qr1[1]], [qr1[2]])
        self.assertRaises(CircuitError, qc.cswap, qr2[4::-1],
                          qr1[1:9:2], qr1[2:9:2])

    def test_apply_ccx_to_empty_slice(self):
        """test applying ccx to non-register raises"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, qc.ccx, qr[2:0], qr[4:2], qr[7:5])

    def test_apply_cx_to_non_register(self):
        """test applying ccx to non-register raises"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, qc.cx, qc[0:2], qc[2:4])

    def test_apply_ch_to_slice(self):
        """test applying ch to slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        # test slice
        qc = QuantumCircuit(qr, cr)
        ctl_slice = slice(0, 2)
        tgt_slice = slice(2, 4)
        qc.ch(qr[ctl_slice], qr[tgt_slice])
        for (gate, qargs, _), ictrl, itgt in zip(qc.data, range(0, 2), range(2, 4)):
            self.assertEqual(gate.name, 'ch')
            self.assertEqual(len(qargs), 2)
            self.assertEqual(qargs[0].index, ictrl)
            self.assertEqual(qargs[1].index, itgt)
        # test single qubit args
        qc = QuantumCircuit(qr, cr)
        qc.ch(qr[0], qr[1])
        self.assertEqual(len(qc.data), 1)
        op, qargs, _ = qc.data[0]
        self.assertEqual(op.name, 'ch')
        self.assertEqual(qargs[0].index, 0)
        self.assertEqual(qargs[1].index, 1)

    def test_measure_slice(self):
        """test measure slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        cr2 = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr[0:2], cr[2:4])
        for (gate, qargs, cargs), ictrl, itgt in zip(qc.data, range(0, 2), range(2, 4)):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(qargs), 1)
            self.assertEqual(len(cargs), 1)
            self.assertEqual(qargs[0].index, ictrl)
            self.assertEqual(cargs[0].index, itgt)
        # test single element slice
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr[0:1], cr[2:3])
        for (gate, qargs, cargs), ictrl, itgt in zip(qc.data, range(0, 1), range(2, 3)):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(qargs), 1)
            self.assertEqual(len(cargs), 1)
            self.assertEqual(qargs[0].index, ictrl)
            self.assertEqual(cargs[0].index, itgt)
        # test tuple
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr[0], cr[2])
        self.assertEqual(len(qc.data), 1)
        op, qargs, cargs = qc.data[0]
        self.assertEqual(op.name, 'measure')
        self.assertEqual(len(qargs), 1)
        self.assertEqual(len(cargs), 1)
        self.assertTrue(isinstance(qargs[0], Qubit))
        self.assertTrue(isinstance(cargs[0], Clbit))
        self.assertEqual(qargs[0].index, 0)
        self.assertEqual(cargs[0].index, 2)
        # test full register
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr, cr)
        for (gate, qargs, cargs), ictrl, itgt in zip(qc.data, range(len(qr)), range(len(cr))):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(qargs), 1)
            self.assertEqual(len(cargs), 1)
            self.assertEqual(qargs[0].index, ictrl)
            self.assertEqual(cargs[0].index, itgt)
        # test mix slice full register
        qc = QuantumCircuit(qr, cr2)
        qc.measure(qr[::2], cr2)
        for (gate, qargs, cargs), ictrl, itgt in zip(qc.data, range(0, 10, 2), range(len(cr2))):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(qargs), 1)
            self.assertEqual(len(cargs), 1)
            self.assertEqual(qargs[0].index, ictrl)
            self.assertEqual(cargs[0].index, itgt)

    def test_measure_slice_raises(self):
        """test raising exception for strange measures"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        with self.assertRaises(CircuitError):
            qc.measure(qr[0:2], cr[2])
        # this is ok
        qc.measure(qr[0], cr[0:2])

    def test_list_indexing(self):
        """test list indexing"""
        qr = QuantumRegister(10)
        cr = QuantumRegister(10)
        qc = QuantumCircuit(qr, cr)
        ind = [0, 1, 8, 9]
        qc.h(qr[ind])
        self.assertEqual(len(qc.data), len(ind))
        for (gate, qargs, _), index in zip(qc.data, ind):
            self.assertEqual(gate.name, 'h')
            self.assertEqual(len(qargs), 1)
            self.assertEqual(qargs[0].index, index)
        qc = QuantumCircuit(qr, cr)
        ind = [0, 1, 8, 9]
        qc.cx(qr[ind], qr[2:6])
        for (gate, qargs, _), ind1, ind2 in zip(qc.data, ind, range(2, 6)):
            self.assertEqual(gate.name, 'cx')
            self.assertEqual(len(qargs), 2)
            self.assertEqual(qargs[0].index, ind1)
            self.assertEqual(qargs[1].index, ind2)

    def test_bit_index_mix_list(self):
        """Test mix of bit and index in list indexing"""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        expected = QuantumCircuit(qr)
        expected.h([qr[0], qr[1]])

        qc.h([qr[0], 1])
        self.assertEqual(qc, expected)

    def test_4_args_custom_gate_trivial_expansion(self):
        """test 'expansion' of 4 args in custom gate.
        See https://github.com/Qiskit/qiskit-terra/issues/2508"""
        qr = QuantumRegister(4)
        circ = QuantumCircuit(qr)
        circ.append(Gate("mcx", 4, []), [qr[0], qr[1], qr[2], qr[3]])

        self.assertEqual(len(circ.data), 1)
        (gate, qargs, _) = circ.data[0]
        self.assertEqual(gate.name, 'mcx')
        self.assertEqual(len(qargs), 4)

    def test_4_args_unitary_trivial_expansion(self):
        """test 'expansion' of 4 args in unitary gate.
        See https://github.com/Qiskit/qiskit-terra/issues/2508"""
        qr = QuantumRegister(4)
        circ = QuantumCircuit(qr)
        circ.unitary(np.eye(2 ** 4), [qr[0], qr[1], qr[2], qr[3]])

        self.assertEqual(len(circ.data), 1)
        (gate, qargs, _) = circ.data[0]
        self.assertEqual(gate.name, 'unitary')
        self.assertEqual(len(qargs), 4)

    def test_4_args_unitary_zip_expansion(self):
        """test zip expansion of 4 args in unitary gate.
        See https://github.com/Qiskit/qiskit-terra/issues/2508"""
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(4)
        qr3 = QuantumRegister(4)
        qr4 = QuantumRegister(4)

        circ = QuantumCircuit(qr1, qr2, qr3, qr4)
        circ.unitary(np.eye(2 ** 4), [qr1, qr2, qr3, qr4])

        self.assertEqual(len(circ.data), 4)
        for (gate, qargs, _) in circ.data:
            self.assertEqual(gate.name, 'unitary')
            self.assertEqual(len(qargs), 4)

    def test_quantumregister_hash_upate_name(self):
        """Test QuantumRegister hash changes on name update."""
        test_reg = QuantumRegister(2)
        orig_hash = hash(test_reg)
        orig_bit_hashes = [hash(x) for x in test_reg]
        test_reg.name = 'test_quantum'
        new_hash = hash(test_reg)
        new_bit_hashes = [hash(x) for x in test_reg]
        self.assertNotEqual(orig_hash, new_hash)
        for x in range(2):
            self.assertNotEqual(orig_bit_hashes[x], new_bit_hashes[x])

    def test_quantumregister_hash_upate_size(self):
        """Test QuantumRegister hash changes on size update."""
        test_reg = QuantumRegister(2)
        orig_hash = hash(test_reg)
        test_reg.size = 3
        new_hash = hash(test_reg)
        self.assertNotEqual(orig_hash, new_hash)

    def test_classicalregister_hash_upate_name(self):
        """Test ClassicalRegister hash changes on name update."""
        test_reg = ClassicalRegister(2)
        orig_hash = hash(test_reg)
        orig_bit_hashes = [hash(x) for x in test_reg]
        test_reg.name = 'test_classical'
        new_hash = hash(test_reg)
        new_bit_hashes = [hash(x) for x in test_reg]
        self.assertNotEqual(orig_hash, new_hash)
        for x in range(2):
            self.assertNotEqual(orig_bit_hashes[x], new_bit_hashes[x])

    def test_classicalregister_hash_upate_size(self):
        """Test ClassicalRegister hash changes on size update."""
        test_reg = ClassicalRegister(2)
        orig_hash = hash(test_reg)
        test_reg.size = 3
        new_hash = hash(test_reg)
        self.assertNotEqual(orig_hash, new_hash)
