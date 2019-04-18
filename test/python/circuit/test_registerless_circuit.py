# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test registerless QuantumCircuit and Gates on wires"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, QiskitError
from qiskit.test import QiskitTestCase


class TestRegisterlessCircuit(QiskitTestCase):
    """Test registerless QuantumCircuit."""

    def test_circuit_constructor_qwires(self):
        """Create a QuantumCircuit directly with quantum wires
        """
        circuit = QuantumCircuit(2)

        expected = QuantumCircuit(QuantumRegister(2, 'q'))

        self.assertEqual(circuit, expected)

    def test_circuit_constructor_wires_wrong(self):
        """Create a registerless QuantumCircuit wrongly
        """
        self.assertRaises(QiskitError, QuantumCircuit, 1, 2, 3)  # QuantumCircuit(1, 2, 3)

    def test_circuit_constructor_wires_wrong_mix(self):
        """Create an almost-registerless QuantumCircuit
        """
        # QuantumCircuit(1, ClassicalRegister(2))
        self.assertRaises(QiskitError, QuantumCircuit, 1, ClassicalRegister(2))


class TestGatesOnWires(QiskitTestCase):
    """Test gates on wires."""

    def test_circuit_single_wire_h(self):
        """Test circuit on wire (H gate).
        """
        qreg = QuantumRegister(2)
        circ = QuantumCircuit(qreg)
        circ.h(1)

        expected = QuantumCircuit(qreg)
        expected.h(qreg[1])

        self.assertEqual(circ, expected)

    def test_circuit_two_wire_cx(self):
        """Test circuit two wires (CX gate).
        """
        qreg = QuantumRegister(2)
        expected = QuantumCircuit(qreg)
        expected.cx(qreg[0], qreg[1])

        circ = QuantumCircuit(qreg)
        circ.cx(0, 1)

        self.assertEqual(circ, expected)

    def test_circuit_single_wire_measure(self):
        """Test circuit on wire (measure gate).
        """
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)

        circ = QuantumCircuit(qreg, creg)
        circ.measure(1, 1)

        expected = QuantumCircuit(qreg, creg)
        expected.measure(qreg[1], creg[1])

        self.assertEqual(circ, expected)

    def test_circuit_multi_qregs_h(self):
        """Test circuit multi qregs and wires (H gate).
        """
        qreg0 = QuantumRegister(2)
        qreg1 = QuantumRegister(2)

        circ = QuantumCircuit(qreg0, qreg1)
        circ.h(0)
        circ.h(2)

        expected = QuantumCircuit(qreg0, qreg1)
        expected.h(qreg0[0])
        expected.h(qreg1[0])

        self.assertEqual(circ, expected)

    def test_circuit_multi_qreg_cregs_measure(self):
        """Test circuit multi qregs/cregs and wires (measure).
        """
        qreg0 = QuantumRegister(2)
        creg0 = ClassicalRegister(2)
        qreg1 = QuantumRegister(2)
        creg1 = ClassicalRegister(2)

        circ = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circ.measure(0, 2)
        circ.measure(2, 1)

        expected = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        expected.measure(qreg0[0], creg1[0])
        expected.measure(qreg1[0], creg0[1])

        self.assertEqual(circ, expected)

    def test_circuit_barrier(self):
        """Test barrier on wires.
        """
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)

        circ = QuantumCircuit(qreg01, qreg23)
        circ.barrier(0)
        circ.barrier(2)

        expected = QuantumCircuit(qreg01, qreg23)
        expected.barrier(qreg01[0])
        expected.barrier(qreg23[0])

        self.assertEqual(circ, expected)

    # def test_circuit_conditional(self):
    #     """Test conditional on wires.
    #     """
    #     qreg = QuantumRegister(2)
    #     creg = ClassicalRegister(4)
    #     circ = QuantumCircuit(qreg, creg)
    #     circ.h(0).c_if(creg, 3)
    #
    #     expected = QuantumCircuit(qreg, creg)
    #     expected.h(qreg[0]).c_if(creg, 3)
    #
    #     self.assertEqual(circ, expected)
    #
    # def test_circuit_qwire_out_of_range(self):
    #     """Fail if quantum wire is out of range.
    #     """
    #     qreg = QuantumRegister(2)
    #
    #     circ = QuantumCircuit(qreg)
    #     self.assertRaises(QiskitError, circ.h, 99)  # circ.h(99)
    #
    # def test_circuit_cwire_out_of_range(self):
    #     """Fail if classical wire is out of range.
    #     """
    #     qreg = QuantumRegister(2)
    #     creg = ClassicalRegister(2)
    #
    #     circ = QuantumCircuit(qreg, creg)
    #     self.assertRaises(QiskitError, circ.measure, 1, 99)  # circ.measure(1, 99)

class TestGatesOnWireRange(QiskitTestCase):
    """Test gates on wire range."""

    def test_circuit_initialize(self):
        """Test initialize on wires.
        """
        init_vector = [0.5, 0.5, 0.5, 0.5]
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)

        circuit = QuantumCircuit(qreg01, qreg23, name='circuit')
        circuit.initialize(init_vector, [0, 2])

        expected = QuantumCircuit(qreg01, qreg23, name='circuit')
        expected.initialize(init_vector, [qreg01[0], qreg23[0]])

        self.assertEqual(circuit, expected)

# class TestGatesOnWireSlice(QiskitTestCase):
#     """Test gates on wire slice."""
#
#     def test_wire_slice(self):
#         """Test gate wire slice
#         """
#         qreg = QuantumRegister(4)
#
#         # circ = QuantumCircuit(qreg)
#         # circ.h(slice(1, 2))
#
#         expected = QuantumCircuit(qreg)
#         expected.h(qreg[0:2])
#
#         # self.assertEqual(circ, expected)
#
#     # def test_circuit_multi_qregs_h(self):
#     #     """Test circuit multi qregs and wires (H gate).
#     #     """
#     #     qreg0 = QuantumRegister(2)
#     #     qreg1 = QuantumRegister(2)
#     #
#     #     circ = QuantumCircuit(qreg0, qreg1)
#     #     circ.h(0)
#     #     circ.h(2)
#     #
#     #     expected = QuantumCircuit(qreg0, qreg1)
#     #     expected.h(qreg0[0])
#     #     expected.h(qreg1[0])
#     #
#     #     self.assertEqual(circ, expected)
#     #
#     # def test_circuit_multi_qreg_cregs_measure(self):
#     #     """Test circuit multi qregs/cregs and wires (measure).
#     #     """
#     #     qreg0 = QuantumRegister(2)
#     #     creg0 = ClassicalRegister(2)
#     #     qreg1 = QuantumRegister(2)
#     #     creg1 = ClassicalRegister(2)
#     #
#     #     circ = QuantumCircuit(qreg0, qreg1, creg0, creg1)
#     #     circ.measure(0, 2)
#     #     circ.measure(2, 1)
#     #
#     #     expected = QuantumCircuit(qreg0, qreg1, creg0, creg1)
#     #     expected.measure(qreg0[0], creg1[0])
#     #     expected.measure(qreg1[0], creg0[1])
#     #
#     #     self.assertEqual(circ, expected)
#     #
#     # def test_circuit_barrier(self):
#     #     """Test barrier on wires.
#     #     """
#     #     qreg01 = QuantumRegister(2)
#     #     qreg23 = QuantumRegister(2)
#     #
#     #     circ = QuantumCircuit(qreg01, qreg23)
#     #     circ.barrier(0)
#     #     circ.barrier(2)
#     #
#     #     expected = QuantumCircuit(qreg01, qreg23)
#     #     expected.barrier(qreg01[0])
#     #     expected.barrier(qreg23[0])
#     #
#     #     self.assertEqual(circ, expected)
#     #
#     # def test_circuit_initialize(self):
#     #     """Test initialize on wires.
#     #     """
#     #     init_vector = [0.5, 0.5, 0.5, 0.5]
#     #     qreg01 = QuantumRegister(2)
#     #     qreg23 = QuantumRegister(2)
#     #
#     #     circuit = QuantumCircuit(qreg01, qreg23, name='circuit')
#     #     circuit.initialize(init_vector, [0, 2])
#     #
#     #     expected = QuantumCircuit(qreg01, qreg23, name='circuit')
#     #     expected.initialize(init_vector, [qreg01[0], qreg23[0]])
#     #
#     #     self.assertEqual(circuit, expected)
#     #
#     # def test_circuit_conditional(self):
#     #     """Test conditional on wires.
#     #     """
#     #     qreg = QuantumRegister(2)
#     #     creg = ClassicalRegister(4)
#     #     circ = QuantumCircuit(qreg, creg)
#     #     circ.h(0).c_if(creg, 3)
#     #
#     #     expected = QuantumCircuit(qreg, creg)
#     #     expected.h(qreg[0]).c_if(creg, 3)
#     #
#     #     self.assertEqual(circ, expected)
#     #
#     # def test_circuit_qwire_out_of_range(self):
#     #     """Fail if quantum wire is out of range.
#     #     """
#     #     qreg = QuantumRegister(2)
#     #
#     #     circ = QuantumCircuit(qreg)
#     #     self.assertRaises(QiskitError, circ.h, 99)  # circ.h(99)
#     #
#     # def test_circuit_cwire_out_of_range(self):
#     #     """Fail if classical wire is out of range.
#     #     """
#     #     qreg = QuantumRegister(2)
#     #     creg = ClassicalRegister(2)
#     #
#     #     circ = QuantumCircuit(qreg, creg)
#     #     self.assertRaises(QiskitError, circ.measure, 1, 99)  # circ.measure(1, 99)
