# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for comparing the outputs of text drawing of circuit with expected ones."""

import unittest
from math import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from .common import QiskitTestCase
from qiskit.tools.visualization.text import text_drawer


class TestCircuitTextDrawer(QiskitTestCase):
    def test_text_sample_circuit(self):
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.z(qr[0])
        circuit.barrier(qr)
        circuit.h(qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[0])
        circuit.t(qr[0])
        circuit.tdg(qr[0])
        circuit.iden(qr[0])
        circuit.reset(qr[0])
        circuit.rx(pi, qr[0])
        circuit.ry(pi, qr[0])
        circuit.rz(pi, qr[0])
        circuit.u0(pi, qr[0])
        circuit.u1(pi, qr[0])
        circuit.u2(pi, pi, qr[0])
        circuit.u3(pi, pi, pi, qr[0])
        circuit.swap(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.ch(qr[0], qr[1])
        circuit.cu1(pi, qr[0], qr[1])
        circuit.cu3(pi, pi, pi, qr[0], qr[1])
        circuit.crz(pi, qr[0], qr[1])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.measure(qr, cr)
        text_drawer(circuit)

    def test_text_pager(self):
        qr = QuantumRegister(1, 'q')
        circuit = QuantumCircuit(qr)
        no_instructions = 50
        for _ in range(no_instructions):
            circuit.x(qr[0])
        self.assertEqual(text_drawer(circuit, line_length=10).count('\n'), no_instructions * 3 + 2)

    def test_text_measure_1(self):
        expected = '\n'.join([
            "              ┌─┐",
            "q_0: |0>──────┤M├",
            "           ┌─┐└╥┘",
            "q_1: |0>───┤M├─╫─",
            "        ┌─┐└╥┘ ║ ",
            "q_2: |0>┤M├─╫──╫─",
            "        └╥┘ ║  ║ ",
            " c_0: 0 ═╬══╬══╩═",
            "         ║  ║    ",
            " c_1: 0 ═╬══╩════",
            "         ║       ",
            " c_2: 0 ═╩═══════",
            "                 "])
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        self.assertEqual(text_drawer(circuit), expected)

    def test_text_measure_2(self):
        expected = '\n'.join([
            "               ",
            "q1_0: |0>──────",
            "               ",
            "q1_1: |0>──────",
            "            ┌─┐",
            "q2_0: |0>───┤M├",
            "         ┌─┐└╥┘",
            "q2_1: |0>┤M├─╫─",
            "         └╥┘ ║ ",
            " c1_0: 0 ═╬══╬═",
            "          ║  ║ ",
            " c1_1: 0 ═╬══╬═",
            "          ║  ║ ",
            " c2_0: 0 ═╬══╩═",
            "          ║    ",
            " c2_1: 0 ═╩════",
            "               "])
        qr1 = QuantumRegister(2, 'q1')
        cr1 = ClassicalRegister(2, 'c1')
        qr2 = QuantumRegister(2, 'q2')
        cr2 = ClassicalRegister(2, 'c2')
        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.measure(qr2, cr2)
        self.assertEqual(text_drawer(circuit), expected)

    def test_text_swap(self):
        expected = '\n'.join([
            "               ",
            "q1_0: |0>────X─",
            "             │ ",
            "q1_1: |0>─X──┼─",
            "          │  │ ",
            "q2_0: |0>─┼──X─",
            "          │    ",
            "q2_1: |0>─X────",
            "               "])
        qr1 = QuantumRegister(2, 'q1')
        qr2 = QuantumRegister(2, 'q2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.swap(qr1, qr2)
        self.assertEqual(text_drawer(circuit), expected)


if __name__ == '__main__':
    unittest.main()
