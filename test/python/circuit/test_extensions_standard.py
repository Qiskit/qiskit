# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

import unittest
import warnings
from inspect import signature
from ddt import ddt, data, unpack

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit.qasm import pi
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase
from qiskit.circuit import Gate, ControlledGate, ParameterVector
from qiskit import BasicAer
from qiskit.quantum_info.operators.predicates import matrix_equal, is_unitary_matrix

from qiskit.circuit.library import (
    HGate, CHGate, IGate, RGate, RXGate, CRXGate, RYGate, CRYGate, RZGate,
    CRZGate, SGate, SdgGate, CSwapGate, TGate, TdgGate, U1Gate, CU1Gate,
    U2Gate, U3Gate, CU3Gate, XGate, CXGate, CCXGate, YGate, CYGate,
    ZGate, CZGate
)


class TestStandard1Q(QiskitTestCase):
    """Standard Extension Test. Gates with a single Qubit"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)

    def test_barrier(self):
        self.circuit.barrier(self.qr[1])
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[1]])

    def test_barrier_wires(self):
        self.circuit.barrier(1)
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[1]])

    def test_barrier_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.barrier, self.cr[0])
        self.assertRaises(CircuitError, qc.barrier, self.cr)
        self.assertRaises(CircuitError, qc.barrier, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.barrier, .0)

    def test_conditional_barrier_invalid(self):
        qc = self.circuit
        barrier = qc.barrier(self.qr)
        self.assertRaises(QiskitError, barrier.c_if, self.cr, 0)

    def test_barrier_reg(self):
        self.circuit.barrier(self.qr)
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_barrier_none(self):
        self.circuit.barrier()
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2],
                                 self.qr2[0], self.qr2[1], self.qr2[2]])

    def test_ccx(self):
        self.circuit.ccx(self.qr[0], self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ccx')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_ccx_wires(self):
        self.circuit.ccx(0, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ccx')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_ccx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.ccx, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.ccx, self.qr[0], self.qr[0], self.qr[2])
        self.assertRaises(CircuitError, qc.ccx, 0.0, self.qr[0], self.qr[2])
        self.assertRaises(CircuitError, qc.ccx, self.cr, self.qr, self.qr)
        self.assertRaises(CircuitError, qc.ccx, 'a', self.qr[1], self.qr[2])

    def test_ch(self):
        self.circuit.ch(self.qr[0], self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ch')
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_ch_wires(self):
        self.circuit.ch(0, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ch')
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_ch_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.ch, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.ch, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.ch, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.ch, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.ch, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.ch, 'a', self.qr[1])

    def test_crz(self):
        self.circuit.crz(1, self.qr[0], self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'crz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_cry(self):
        self.circuit.cry(1, self.qr[0], self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cry')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_crx(self):
        self.circuit.crx(1, self.qr[0], self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'crx')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_crz_wires(self):
        self.circuit.crz(1, 0, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'crz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_cry_wires(self):
        self.circuit.cry(1, 0, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cry')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_crx_wires(self):
        self.circuit.crx(1, 0, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'crx')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_crz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.crz, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.crz, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.crz, 0, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.crz, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.crz, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.crz, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.crz, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.crz, 'a', self.qr[1], self.qr[2])

    def test_cry_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cry, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.cry, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cry, 0, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cry, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.cry, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cry, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.cry, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.cry, 'a', self.qr[1], self.qr[2])

    def test_crx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.crx, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.crx, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.crx, 0, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.crx, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.crx, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.crx, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.crx, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.crx, 'a', self.qr[1], self.qr[2])

    def test_cswap(self):
        self.circuit.cswap(self.qr[0], self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cswap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_cswap_wires(self):
        self.circuit.cswap(0, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cswap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_cswap_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cswap, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cswap, self.qr[1], self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cswap, self.qr[1], .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cswap, self.cr[0], self.cr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.cswap, self.qr[0], self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, .0, self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, (self.qr, 3), self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, self.cr, self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, 'a', self.qr[1], self.qr[2])

    def test_cu1(self):
        self.circuit.cu1(1, self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu1_wires(self):
        self.circuit.cu1(1, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu1_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cu1, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cu1, 1, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cu1, self.qr[1], 0, self.qr[0])
        self.assertRaises(CircuitError, qc.cu1, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.cu1, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cu1, 0, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cu1, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.cu1, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cu1, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.cu1, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.cu1, 'a', self.qr[1], self.qr[2])

    def test_cu3(self):
        self.circuit.cu3(1, 2, 3, self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu3_wires(self):
        self.circuit.cu3(1, 2, 3, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu3_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cu3, 0, 0, self.qr[0], self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cu3, 0, 0, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cu3, 0, 0, self.qr[1], 0, self.qr[0])
        self.assertRaises(CircuitError, qc.cu3, 0, 0, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cu3, 0, 0, 0, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cu3, 0, 0, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.cu3, 0, 0, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.cu3, 0, 0, 'a', self.qr[1], self.qr[2])

    def test_cx(self):
        self.circuit.cx(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cx')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cx_wires(self):
        self.circuit.cx(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cx')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cx, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cx, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cx, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cx, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.cx, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.cx, 'a', self.qr[1])

    def test_cy(self):
        self.circuit.cy(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cy')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cy_wires(self):
        self.circuit.cy(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cy')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cy_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cy, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cy, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cy, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cy, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.cy, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.cy, 'a', self.qr[1])

    def test_cz(self):
        self.circuit.cz(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cz')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cz_wires(self):
        self.circuit.cz(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cz')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cz, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cz, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cz, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.cz, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.cz, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.cz, 'a', self.qr[1])

    def test_h(self):
        self.circuit.h(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'h')
        self.assertEqual(qargs, [self.qr[1]])

    def test_h_wires(self):
        self.circuit.h(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'h')
        self.assertEqual(qargs, [self.qr[1]])

    def test_h_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.h, self.cr[0])
        self.assertRaises(CircuitError, qc.h, self.cr)
        self.assertRaises(CircuitError, qc.h, (self.qr, 3))
        self.assertRaises(CircuitError, qc.h, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.h, .0)

    def test_h_reg(self):
        instruction_set = self.circuit.h(self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'h')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_h_reg_inv(self):
        instruction_set = self.circuit.h(self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'h')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_iden(self):
        self.circuit.i(self.qr[1])
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'id')
        self.assertEqual(op.params, [])

    def test_iden_wires(self):
        self.circuit.i(1)
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'id')
        self.assertEqual(op.params, [])

    def test_iden_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.i, self.cr[0])
        self.assertRaises(CircuitError, qc.i, self.cr)
        self.assertRaises(CircuitError, qc.i, (self.qr, 3))
        self.assertRaises(CircuitError, qc.i, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.i, .0)

    def test_iden_reg(self):
        instruction_set = self.circuit.i(self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'id')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_iden_reg_inv(self):
        instruction_set = self.circuit.i(self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'id')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_rx(self):
        self.circuit.rx(1, self.qr[1])
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'rx')
        self.assertEqual(op.params, [1])

    def test_rx_wires(self):
        self.circuit.rx(1, 1)
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'rx')
        self.assertEqual(op.params, [1])

    def test_rx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.rx, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.rx, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.rx, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.rx, 0, .0)
        self.assertRaises(CircuitError, qc.rx, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.rx, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.rx, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.rx, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.rx, 0, 'a')

    def test_rx_reg(self):
        instruction_set = self.circuit.rx(1, self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'rx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_rx_reg_inv(self):
        instruction_set = self.circuit.rx(1, self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'rx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_rx_pi(self):
        qc = self.circuit
        qc.rx(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rx')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_ry(self):
        self.circuit.ry(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ry')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_ry_wires(self):
        self.circuit.ry(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ry')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_ry_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.ry, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.ry, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.ry, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.ry, 0, .0)
        self.assertRaises(CircuitError, qc.ry, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.ry, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.ry, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.ry, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.ry, 0, 'a')

    def test_ry_reg(self):
        instruction_set = self.circuit.ry(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'ry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_ry_reg_inv(self):
        instruction_set = self.circuit.ry(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_ry_pi(self):
        qc = self.circuit
        qc.ry(pi / 2, self.qr[1])
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'ry')
        self.assertEqual(op.params, [pi / 2])

    def test_rz(self):
        self.circuit.rz(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_rz_wires(self):
        self.circuit.rz(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_rz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.rz, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.rz, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.rz, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.rz, 0, .0)
        self.assertRaises(CircuitError, qc.rz, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.rz, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.rz, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.rz, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.rz, 0, 'a')

    def test_rz_reg(self):
        instruction_set = self.circuit.rz(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'rz')
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_rz_reg_inv(self):
        instruction_set = self.circuit.rz(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'rz')
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_rz_pi(self):
        self.circuit.rz(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rz')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_rzz(self):
        self.circuit.rzz(1, self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rzz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_rzz_wires(self):
        self.circuit.rzz(1, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rzz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_rzz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.rzz, 1, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.rzz, 1, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.rzz, 1, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.rzz, 1, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.rzz, 1, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.rzz, 1, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.rzz, 0.1, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.rzz, 0.1, self.qr[0], self.qr[0])

    def test_s(self):
        self.circuit.s(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 's')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_s_wires(self):
        self.circuit.s(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 's')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_s_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.s, self.cr[0])
        self.assertRaises(CircuitError, qc.s, self.cr)
        self.assertRaises(CircuitError, qc.s, (self.qr, 3))
        self.assertRaises(CircuitError, qc.s, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.s, .0)

    def test_s_reg(self):
        instruction_set = self.circuit.s(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 's')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_s_reg_inv(self):
        instruction_set = self.circuit.s(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'sdg')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_sdg(self):
        self.circuit.sdg(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'sdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_sdg_wires(self):
        self.circuit.sdg(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'sdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_sdg_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.sdg, self.cr[0])
        self.assertRaises(CircuitError, qc.sdg, self.cr)
        self.assertRaises(CircuitError, qc.sdg, (self.qr, 3))
        self.assertRaises(CircuitError, qc.sdg, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.sdg, .0)

    def test_sdg_reg(self):
        instruction_set = self.circuit.sdg(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'sdg')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_sdg_reg_inv(self):
        instruction_set = self.circuit.sdg(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 's')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_swap(self):
        self.circuit.swap(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'swap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_swap_wires(self):
        self.circuit.swap(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'swap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_swap_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.swap, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.swap, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.swap, .0, self.qr[0])
        self.assertRaises(CircuitError, qc.swap, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.swap, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.swap, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.swap, self.qr, self.qr2[[1, 2]])
        self.assertRaises(CircuitError, qc.swap, self.qr[:2], self.qr2)

    def test_t(self):
        self.circuit.t(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 't')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_t_wire(self):
        self.circuit.t(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 't')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_t_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.t, self.cr[0])
        self.assertRaises(CircuitError, qc.t, self.cr)
        self.assertRaises(CircuitError, qc.t, (self.qr, 3))
        self.assertRaises(CircuitError, qc.t, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.t, .0)

    def test_t_reg(self):
        instruction_set = self.circuit.t(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 't')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_t_reg_inv(self):
        instruction_set = self.circuit.t(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'tdg')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_tdg(self):
        self.circuit.tdg(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'tdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_tdg_wires(self):
        self.circuit.tdg(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'tdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_tdg_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.tdg, self.cr[0])
        self.assertRaises(CircuitError, qc.tdg, self.cr)
        self.assertRaises(CircuitError, qc.tdg, (self.qr, 3))
        self.assertRaises(CircuitError, qc.tdg, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.tdg, .0)

    def test_tdg_reg(self):
        instruction_set = self.circuit.tdg(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'tdg')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_tdg_reg_inv(self):
        instruction_set = self.circuit.tdg(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 't')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_u1(self):
        self.circuit.u1(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u1_wires(self):
        self.circuit.u1(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u1_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(CircuitError, qc.u1, self.cr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.u1, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.u1, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.u1, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.u1, 0, .0)
        self.assertRaises(CircuitError, qc.u1, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.u1, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.u1, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.u1, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.u1, 0, 'a')

    def test_u1_reg(self):
        instruction_set = self.circuit.u1(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_u1_reg_inv(self):
        instruction_set = self.circuit.u1(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_u1_pi(self):
        qc = self.circuit
        qc.u1(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u1')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u2(self):
        self.circuit.u2(1, 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u2')
        self.assertEqual(op.params, [1, 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u2_wires(self):
        self.circuit.u2(1, 2, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u2')
        self.assertEqual(op.params, [1, 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u2_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.u2, 0, self.cr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.u2, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.u2, 0, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.u2, 0, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.u2, 0, 0, .0)
        self.assertRaises(CircuitError, qc.u2, 0, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.u2, 0, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.u2, 0, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.u2, 0, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.u2, 0, 0, 'a')

    def test_u2_reg(self):
        instruction_set = self.circuit.u2(1, 2, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u2')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2])

    def test_u2_reg_inv(self):
        instruction_set = self.circuit.u2(1, 2, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u2')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-pi - 2, -1 + pi])

    def test_u2_pi(self):
        self.circuit.u2(pi / 2, 0.3 * pi, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u2')
        self.assertEqual(op.params, [pi / 2, 0.3 * pi])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u3(self):
        self.circuit.u3(1, 2, 3, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u3_wires(self):
        self.circuit.u3(1, 2, 3, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u3_invalid(self):
        qc = self.circuit
        # TODO self.assertRaises(CircuitError, qc.u3, 0, self.cr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.u3, 0, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.u3, 0, 0, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.u3, 0, 0, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.u3, 0, 0, 0, .0)
        self.assertRaises(CircuitError, qc.u3, 0, 0, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.u3, 0, 0, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.u3, 0, 0, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.u3, 0, 0, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.u3, 0, 0, 0, 'a')

    def test_u3_reg(self):
        instruction_set = self.circuit.u3(1, 2, 3, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_u3_reg_inv(self):
        instruction_set = self.circuit.u3(1, 2, 3, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_u3_pi(self):
        self.circuit.u3(pi, pi / 2, 0.3 * pi, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u3')
        self.assertEqual(op.params, [pi, pi / 2, 0.3 * pi])
        self.assertEqual(qargs, [self.qr[1]])

    def test_x(self):
        self.circuit.x(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'x')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_x_wires(self):
        self.circuit.x(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'x')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_x_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.x, self.cr[0])
        self.assertRaises(CircuitError, qc.x, self.cr)
        self.assertRaises(CircuitError, qc.x, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.x, 0.0)

    def test_x_reg(self):
        instruction_set = self.circuit.x(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'x')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_x_reg_inv(self):
        instruction_set = self.circuit.x(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'x')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_y(self):
        self.circuit.y(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'y')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_y_wires(self):
        self.circuit.y(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'y')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_y_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.y, self.cr[0])
        self.assertRaises(CircuitError, qc.y, self.cr)
        self.assertRaises(CircuitError, qc.y, (self.qr, 'a'))
        self.assertRaises(CircuitError, qc.y, 0.0)

    def test_y_reg(self):
        instruction_set = self.circuit.y(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'y')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_y_reg_inv(self):
        instruction_set = self.circuit.y(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'y')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_z(self):
        self.circuit.z(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'z')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_z_wires(self):
        self.circuit.z(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'z')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_z_reg(self):
        instruction_set = self.circuit.z(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'z')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_z_reg_inv(self):
        instruction_set = self.circuit.z(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'z')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])


class TestStandard2Q(QiskitTestCase):
    """Standard Extension Test. Gates with two Qubits"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)

    def test_barrier_reg_bit(self):
        self.circuit.barrier(self.qr, self.qr2[0])
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2], self.qr2[0]])

    def test_ch_reg_reg(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_reg_reg_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_reg_bit(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_reg_bit_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_bit_reg(self):
        instruction_set = self.circuit.ch(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_crz_reg_reg(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crz_reg_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crz_reg_bit(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crz_reg_bit_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crz_bit_reg(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crz_bit_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cry_reg_reg(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cry_reg_reg_inv(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cry_reg_bit(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cry_reg_bit_inv(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cry_bit_reg(self):
        instruction_set = self.circuit.cry(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cry_bit_reg_inv(self):
        instruction_set = self.circuit.cry(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crx_reg_reg(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crx_reg_reg_inv(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crx_reg_bit(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'crx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crx_reg_bit_inv(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crx_bit_reg(self):
        instruction_set = self.circuit.crx(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crx_bit_reg_inv(self):
        instruction_set = self.circuit.crx(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu1_reg_reg(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cu1_reg_reg_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu1_reg_bit(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cu1_reg_bit_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu1_bit_reg(self):
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cu1_bit_reg_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu3_reg_reg(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_cu3_reg_reg_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_cu3_reg_bit(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_cu3_reg_bit_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_cu3_bit_reg(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_cu3_bit_reg_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_cx_reg_reg(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_reg_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_reg_bit(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_reg_bit_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_bit_reg(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_bit_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_reg(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_bit(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_bit_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_bit_reg(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_bit_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_reg(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_bit(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_bit_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_bit_reg(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_bit_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_swap_reg_reg(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'swap')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_swap_reg_reg_inv(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'swap')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])


class TestStandard3Q(QiskitTestCase):
    """Standard Extension Test. Gates with three Qubits"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.qr3 = QuantumRegister(3, "s")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.qr3, self.cr)

    def test_ccx_reg_reg_reg(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set.instructions[0].name, 'ccx')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ccx_reg_reg_inv(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ccx')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cswap_reg_reg_reg(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set.instructions[0].name, 'cswap')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cswap_reg_reg_inv(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cswap')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])


class TestStandardMethods(QiskitTestCase):
    """Standard Extension Test."""

    def test_to_matrix(self):
        """test gates implementing to_matrix generate matrix which matches
        definition."""
        from qiskit.circuit.library.standard_gates.ms import MSGate

        params = [0.1 * i for i in range(10)]
        gate_class_list = Gate.__subclasses__() + ControlledGate.__subclasses__()
        simulator = BasicAer.get_backend('unitary_simulator')
        for gate_class in gate_class_list:
            sig = signature(gate_class)
            if gate_class == MSGate:
                # due to the signature (num_qubits, theta, *, n_qubits=Noe) the signature detects
                # 3 arguments but really its only 2. This if can be removed once the deprecated
                # n_qubits argument is no longer supported.
                free_params = 2
            else:
                free_params = len(set(sig.parameters) - {'label'})
            try:
                gate = gate_class(*params[0:free_params])
            except (CircuitError, QiskitError, AttributeError):
                self.log.info(
                    'Cannot init gate with params only. Skipping %s',
                    gate_class)
                continue
            if gate.name in ['U', 'CX']:
                continue
            circ = QuantumCircuit(gate.num_qubits)
            circ.append(gate, range(gate.num_qubits))
            try:
                gate_matrix = gate.to_matrix()
            except CircuitError:
                # gate doesn't implement to_matrix method: skip
                self.log.info('to_matrix method FAILED for "%s" gate',
                              gate.name)
                continue
            definition_unitary = execute([circ], simulator).result().get_unitary()
            self.assertTrue(matrix_equal(definition_unitary, gate_matrix))
            self.assertTrue(is_unitary_matrix(gate_matrix))

    def test_to_matrix_op(self):
        """test gates implementing to_matrix generate matrix which matches
        definition using Operator."""
        from qiskit.quantum_info import Operator
        from qiskit.circuit.library.standard_gates.ms import MSGate

        params = [0.1 * i for i in range(10)]
        gate_class_list = Gate.__subclasses__() + ControlledGate.__subclasses__()
        for gate_class in gate_class_list:
            sig = signature(gate_class)
            if gate_class == MSGate:
                # due to the signature (num_qubits, theta, *, n_qubits=Noe) the signature detects
                # 3 arguments but really its only 2. This if can be removed once the deprecated
                # n_qubits argument is no longer supported.
                free_params = 2
            else:
                free_params = len(set(sig.parameters) - {'label'})
            try:
                gate = gate_class(*params[0:free_params])
            except (CircuitError, QiskitError, AttributeError):
                self.log.info(
                    'Cannot init gate with params only. Skipping %s',
                    gate_class)
                continue
            if gate.name in ['U', 'CX']:
                continue
            try:
                gate_matrix = gate.to_matrix()
            except CircuitError:
                # gate doesn't implement to_matrix method: skip
                self.log.info('to_matrix method FAILED for "%s" gate',
                              gate.name)
                continue
            if not hasattr(gate, 'definition') or not gate.definition:
                continue
            definition_unitary = Operator(gate.definition).data
            self.assertTrue(matrix_equal(definition_unitary, gate_matrix))
            self.assertTrue(is_unitary_matrix(gate_matrix))


@ddt
class TestQubitKeywordArgRenaming(QiskitTestCase):
    """Test renaming of qubit keyword args on standard instructions."""

    # pylint: disable=bad-whitespace
    @unpack
    @data(
        ('h',    HGate,    0, [('q', 'qubit')]),
        ('ch',   CHGate,   0, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('id',   IGate,    0, [('q', 'qubit')]),
        ('r',    RGate,    2, [('q', 'qubit')]),
        ('rx',   RXGate,   1, [('q', 'qubit')]),
        ('crx',  CRXGate,  1, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('ry',   RYGate,   1, [('q', 'qubit')]),
        ('cry',  CRYGate,  1, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('rz',   RZGate,   1, [('q', 'qubit')]),
        ('crz',  CRZGate,  1, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('s',    SGate,    0, [('q', 'qubit')]),
        ('sdg',  SdgGate,  0, [('q', 'qubit')]),
        ('cswap',
         CSwapGate,
         0,
         [('ctl', 'control_qubit'),
          ('tgt1', 'target_qubit1'),
          ('tgt2', 'target_qubit2')]),
        ('t',    TGate,    0, [('q', 'qubit')]),
        ('tdg',  TdgGate,  0, [('q', 'qubit')]),
        ('u1',   U1Gate,   1, [('q', 'qubit')]),
        ('cu1',  CU1Gate,  1, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('u2',   U2Gate,   2, [('q', 'qubit')]),
        ('u3',   U3Gate,   3, [('q', 'qubit')]),
        ('cu3',  CU3Gate,  3, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('x',    XGate,    0, [('q', 'qubit')]),
        ('cx',   CXGate, 0, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('ccx',
         CCXGate,
         0,
         [('ctl1', 'control_qubit1'),
          ('ctl2', 'control_qubit2'),
          ('tgt', 'target_qubit')]),
        ('y',    YGate,    0, [('q', 'qubit')]),
        ('cy',   CYGate,   0, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
        ('z',    ZGate,    0, [('q', 'qubit')]),
        ('cz',   CZGate,   0, [('ctl', 'control_qubit'), ('tgt', 'target_qubit')]),
    )
    # pylint: enable=bad-whitespace
    def test_kwarg_deprecation(self, instr_name, inst_class, n_params, kwarg_map):
        # Verify providing *args is unchanged
        num_qubits = len(kwarg_map)

        qr = QuantumRegister(num_qubits)
        qc = QuantumCircuit(qr)
        params = ParameterVector('theta', n_params)

        getattr(qc, instr_name)(*params[:], *qr[:])

        op, qargs, cargs = qc.data[0]
        self.assertIsInstance(op, inst_class)
        self.assertEqual(op.params, params[:])
        self.assertEqual(qargs, qr[:])
        self.assertEqual(cargs, [])

        # Verify providing old_arg raises a DeprecationWarning
        num_qubits = len(kwarg_map)

        qr = QuantumRegister(num_qubits)
        qc = QuantumCircuit(qr)
        params = ParameterVector('theta', n_params)

        with self.assertWarns(DeprecationWarning):
            getattr(qc, instr_name)(*params[:],
                                    **{keyword[0]: qubit
                                       for keyword, qubit
                                       in zip(kwarg_map, qr[:])})

        op, qargs, cargs = qc.data[0]
        self.assertIsInstance(op, inst_class)
        self.assertEqual(op.params, params[:])
        self.assertEqual(qargs, qr[:])
        self.assertEqual(cargs, [])

        # Verify providing new_arg does not raise a DeprecationWarning
        num_qubits = len(kwarg_map)

        qr = QuantumRegister(num_qubits)
        qc = QuantumCircuit(qr)
        params = ParameterVector('theta', n_params)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            getattr(qc, instr_name)(*params[:],
                                    **{keyword[1]: qubit
                                       for keyword, qubit
                                       in zip(kwarg_map, qr[:])})

            self.assertEqual(len(w), 0)

        op, qargs, cargs = qc.data[0]
        self.assertIsInstance(op, inst_class)
        self.assertEqual(op.params, params[:])
        self.assertEqual(qargs, qr[:])
        self.assertEqual(cargs, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
