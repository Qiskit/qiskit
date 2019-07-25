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

# pylint: disable=unused-import

"""Test Qiskit's inverse gate operation."""

import os
import tempfile
import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit import ControlledGate
from qiskit.converters.instruction_to_gate import instruction_to_gate
from qiskit.extensions.standard import CnotGate


class TestControlledGate(QiskitTestCase):
    """ControlledGate tests."""

    def test_controlled_x(self):
        """Test creation of controlled x gate"""
        from qiskit.extensions.standard import XGate
        from qiskit.extensions.standard import CnotGate
        self.assertEqual(XGate().q_if(), CnotGate())

    def test_controlled_y(self):
        """Test creation of controlled x gate"""
        from qiskit.extensions.standard import YGate
        from qiskit.extensions.standard import CyGate
        self.assertEqual(YGate().q_if(), CyGate())

    def test_controlled_z(self):
        """Test creation of controlled x gate"""
        from qiskit.extensions.standard import ZGate
        from qiskit.extensions.standard import CzGate
        self.assertEqual(ZGate().q_if(), CzGate())

    def test_controlled_u1(self):
        """Test creation of controlled x gate"""
        from qiskit.extensions.standard import U1Gate
        from qiskit.extensions.standard import Cu1Gate
        theta = 0.5
        self.assertEqual(U1Gate(theta).q_if(), Cu1Gate(theta))
        
    def test_circuit_append(self):
        """Test appending controlled gate to quantum circuit"""
        circ = QuantumCircuit(5)
        inst = CnotGate()
        circ.append(inst.q_if(), qargs=[0, 2, 1])
        circ.append(inst.q_if(2), qargs=[0, 3, 1, 2])
        circ.append(inst.q_if().q_if(), qargs=[0, 3, 1, 2])  # should be same as above
        self.assertEqual(circ.depth(), 3)
        self.assertEqual(circ[0][0].num_ctrl_qubits, 2)
        self.assertEqual(circ[1][0].num_ctrl_qubits, 3)
        self.assertEqual(circ[2][0].num_ctrl_qubits, 3)        
        self.assertEqual(circ[0][0].num_qubits, 3)
        self.assertEqual(circ[1][0].num_qubits, 4)
        self.assertEqual(circ[2][0].num_qubits, 4)        
        for instr in circ:
            gate = instr[0]
            self.assertTrue(isinstance(gate, ControlledGate))

    def test_high_order_conditional(self):
        gate = ControlledGate('high_order', 20, [], num_ctrl_qubits=15)
        self.assertEqual(gate.num_ctrl_qubits, 15)
        self.assertEqual(gate.num_qubits, 20)

    def test_multiple_control_multiple_target(self):
        qc1 = QuantumCircuit(5, name='bell')
        qc1.h(0)
        for i in range(len(qc1.qubits)-1):
            qc1.cx(i, i+1)
        bell = instruction_to_gate(qc1.to_instruction())
        cbell = bell.q_if(5)
        qc2 = QuantumCircuit(cbell.num_qubits)
        qc2.append(cbell, range(cbell.num_qubits))

        self.assertTrue
        
    def test_definition_specification(self):
        from qiskit.extensions.standard import SwapGate
        swap = SwapGate()
        cswap = ControlledGate('cswap', 3, [], num_ctrl_qubits=1,
                               definition=swap.definition)
        
