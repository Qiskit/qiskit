# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
import numpy as np

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit, execute,
                    BasicAer)
from qiskit.test import QiskitTestCase
from qiskit.circuit import ControlledGate
from qiskit.compiler import transpile
from qiskit.converters.instruction_to_gate import instruction_to_gate
from qiskit.extensions.standard import CnotGate
from qiskit.quantum_info.operators.predicates import matrix_equal


class TestControlledGate(QiskitTestCase):
    """ControlledGate tests."""

    def test_controlled_x(self):
        """Test creation of controlled x gate"""
        from qiskit.extensions.standard import XGate
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
        self.assertEqual(circ[1][0].num_ctrl_qubits, 2)
        self.assertEqual(circ[2][0].num_ctrl_qubits, 1)
        self.assertEqual(circ[0][0].num_qubits, 3)
        self.assertEqual(circ[1][0].num_qubits, 4)
        self.assertEqual(circ[2][0].num_qubits, 4)
        for instr in circ:
            gate = instr[0]
            self.assertTrue(isinstance(gate, ControlledGate))

    def test_definition_specification(self):
        """Test instantiation with explicit definition"""
        from qiskit.extensions.standard import SwapGate
        swap = SwapGate()
        cswap = ControlledGate('cswap', 3, [], num_ctrl_qubits=1,
                               definition=swap.definition)
        self.assertEqual(swap.definition, cswap.definition)

    def test_multi_controlled_composite_gate(self):
        """Test multi controlled composite gate"""
        num_ctrl = 2
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name='cgate')
        cgate.h(sub_q[0])
        cgate.cx(sub_q[0], sub_q[1])
        num_target = cgate.width()
        gate = instruction_to_gate(cgate.to_instruction())
        cont_gate = gate.q_if(num_ctrl_qubits=num_ctrl)
        for state in range(1 << num_ctrl):
            # setup starting state
            state_str = '{:0>{width}b}'.format(state, width=num_ctrl)
            with self.subTest(i=state_str):
                qr1 = QuantumRegister(num_ctrl)
                qr3 = QuantumRegister(num_target)
                cr1 = ClassicalRegister(num_ctrl)
                cr3 = ClassicalRegister(num_target)
                qc = QuantumCircuit(qr1, qr3, cr1, cr3)
                for i, bit in enumerate(state_str[::-1]):
                    if bit == '1':
                        qc.x(qr1[i])
                qc.append(cont_gate, qr1[:]+qr3[:])
                qc.measure(qr1, cr1)
                qc.measure(qr3, cr3)
                simulator = BasicAer.get_backend('qasm_simulator')
                result = execute(qc, simulator).result()
                counts = result.get_counts()
                counts_strip = {''.join(key.split()): value for key, value in counts.items()}
                if state == 2**num_ctrl - 1:
                    target = {'{:0>{width}b}'.format(2**num_target-1, width=num_target)
                              + state_str: 512,
                              '{:0>{width}b}'.format(0, width=num_target)
                              + state_str: 512}
                else:
                    target = {'{:0>{width}b}'.format(0, width=num_target)
                              + state_str: 1024}
                self.assertDictAlmostEqual(counts_strip, target, 40)

    def test_single_controlled_composite_gate(self):
        """Test singly controlled composite gate"""
        num_ctrl = 1
        num_ancil = 0
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name='cgate')
        cgate.h(sub_q[0])
        cgate.cx(sub_q[0], sub_q[1])
        num_target = cgate.width()
        width = num_ctrl + num_target
        gate = instruction_to_gate(cgate.to_instruction())
        for state in range(1 << num_ctrl):
            # setup starting state
            state_str = '{:0>{width}b}'.format(state, width=num_ctrl)
            with self.subTest(i=state_str):
                qr1 = QuantumRegister(num_ctrl)
                if num_ancil:
                    qr2 = QuantumRegister(num_ancil)
                qr3 = QuantumRegister(num_target)
                cr = ClassicalRegister(width)
                if num_ancil:
                    qc = QuantumCircuit(qr1, qr2, qr3, cr)
                else:
                    qc = QuantumCircuit(qr1, qr3, cr)
                for i, bit in enumerate(state_str[::-1]):
                    if bit == '1':
                        qc.x(qr1[i])
                cgate = gate.q_if(num_ctrl_qubits=num_ctrl)
                if num_ancil:
                    qc.append(cgate, qr1[:]+qr2[:]+qr3[:])
                else:
                    qc.append(cgate, qr1[:]+qr3[:])
                qc.measure(qr1, cr[:num_ctrl])
                qc.measure(qr3, cr[num_ctrl:])
                simulator = BasicAer.get_backend('qasm_simulator')
                result = execute(qc, simulator).result()
                counts = result.get_counts()
                if state == 2**num_ctrl - 1:
                    target = {'{:0>{width}b}'.format(2**num_target-1, width=num_target)
                              + state_str: 512,
                              '{:0>{width}b}'.format(0, width=num_target)
                              + state_str: 512}
                else:
                    target = {'{:0>{width}b}'.format(0, width=num_target)
                              + state_str: 1024}
                self.assertDictAlmostEqual(counts, target, 50)

    def test_multi_control_u3(self):
        """test multi controlled u3 gate"""
        import qiskit.extensions.standard.u3 as u3
        import qiskit.extensions.standard.cu3 as cu3
        from qiskit.tools.qi.qi import partial_trace
        np.set_printoptions(linewidth=250, precision=2)

        num_ctrl = 3
        ctrl_dim = 2**num_ctrl
        num_ancillae = 0
        # U3 gate params
        from math import pi
        alpha, beta, gamma = 0.2, 0.3, 0.4

        # cnu3 gate
        u3gate = u3.U3Gate(alpha, beta, gamma)
        cnu3 = u3gate.q_if(num_ctrl)
        width = cnu3.num_qubits
        qr = QuantumRegister(width)
        qcnu3 = QuantumCircuit(qr)
        qcnu3.append(cnu3, qr, [])

        # U3 gate
        qu3 = QuantumCircuit(1)
        qu3.u3(alpha, beta, gamma, 0)

        # CU3 gate
        qcu3 = QuantumCircuit(2)
        qcu3.cu3(alpha, beta, gamma, 0, 1)

        # c-cu3 gate
        width = 3
        qr = QuantumRegister(width)
        qc_cu3 = QuantumCircuit(qr)
        cu3gate = cu3.Cu3Gate(alpha, beta, gamma)
        #import ipdb;ipdb.set_trace()
        c_cu3 = cu3gate.q_if(1)
        qc_cu3.append(c_cu3, qr, [])

        job = execute([qcnu3, qu3, qcu3, qc_cu3], BasicAer.get_backend('unitary_simulator'),
                      basis_gates=['u1', 'u2', 'u3', 'id', 'cx'])
        result = job.result()

        # Circuit unitaries
        mat_cnu3 = result.get_unitary(0)        

        mat_u3 = result.get_unitary(1)
        mat_cu3 = result.get_unitary(2)
        mat_c_cu3 = result.get_unitary(3)

        # Target Controlled-U3 unitary
        ctrl_grnd = np.repeat([[1], [0]], [1, ctrl_dim-1])
        target_cnu3 = np.kron(mat_u3, np.diag(np.roll(ctrl_grnd, ctrl_dim-1)))
        for i in range(ctrl_dim-1):
            target_cnu3 += np.kron(np.eye(2), np.diag(np.roll(ctrl_grnd, i)))
        target_cu3 = np.kron(mat_u3, np.diag([0, 1])) + np.kron(np.eye(2), np.diag([1, 0]))
        target_c_cu3 = np.kron(mat_cu3, np.diag([0, 1])) + np.kron(np.eye(4), np.diag([1, 0]))

        tests = [('check unitary of u3.q_if against tensored unitary of u3',
                  target_cu3, mat_cu3),
                 ('check unitary of cu3.q_if against tensored unitary of cu3',
                  target_c_cu3, mat_c_cu3),
                 ('check unitary of cnu3 against tensored unitary of u3',
                  target_cnu3, mat_cnu3)]
        for itest in tests:
            info, target, decomp = itest[0], itest[1], itest[2]
            with self.subTest(i=info):
                self.log.info(info)
                print(info)
                print(target); print()
                print(decomp); print()
                self.assertTrue(matrix_equal(target, decomp, ignore_phase=True))

    def test_multi_control_u1(self):
        """Test multi controlled u1 gate"""
        import qiskit.extensions.standard.u1 as u1
        import qiskit.extensions.standard.cu1 as cu1
        np.set_printoptions(linewidth=250, precision=2)

        num_ctrl = 3
        ctrl_dim = 2**num_ctrl
        # U1 gate params
        theta = 0.2

        # cnu1 gate
        u1gate = u1.U1Gate(theta)
        cnu1 = u1gate.q_if(num_ctrl)
        width = cnu1.num_qubits
        qr = QuantumRegister(width)
        qcnu1 = QuantumCircuit(qr)
        qcnu1.append(cnu1, qr, [])

        # U1 gate
        qu1 = QuantumCircuit(1)
        qu1.u1(theta, 0)

        # CU1 gate
        qcu1 = QuantumCircuit(2)
        qcu1.cu1(theta, 0, 1)

        # c-cu1 gate
        width = 3
        qr = QuantumRegister(width)
        qc_cu1 = QuantumCircuit(qr)
        cu1gate = cu1.Cu1Gate(theta)
        c_cu1 = cu1gate.q_if(1)
        qc_cu1.append(c_cu1, qr, [])

        job = execute([qcnu1, qu1, qcu1, qc_cu1], BasicAer.get_backend('unitary_simulator'),
                      basis_gates=['u1', 'u2', 'u3', 'id', 'cx'])
        result = job.result()

        # Circuit unitaries
        mat_cnu1 = result.get_unitary(0)
        # trace out ancillae

        mat_u1 = result.get_unitary(1)
        mat_cu1 = result.get_unitary(2)
        mat_c_cu1 = result.get_unitary(3)

        # Target Controlled-U1 unitary
        ctrl_grnd = np.repeat([[1], [0]], [1, ctrl_dim-1])
        target_cnu1 = np.kron(mat_u1, np.diag(np.roll(ctrl_grnd, ctrl_dim-1)))
        for i in range(ctrl_dim-1):
            target_cnu1 += np.kron(np.eye(2), np.diag(np.roll(ctrl_grnd, i)))
        target_cu1 = np.kron(mat_u1, np.diag([0, 1])) + np.kron(np.eye(2), np.diag([1, 0]))
        target_c_cu1 = np.kron(mat_cu1, np.diag([0, 1])) + np.kron(np.eye(4), np.diag([1, 0]))

        tests = [('check unitary of u1.q_if against tensored unitary of u1',
                  target_cu1, mat_cu1),
                 ('check unitary of cu1.q_if against tensored unitary of cu1',
                  target_c_cu1, mat_c_cu1),
                 ('check unitary of cnu1 against tensored unitary of u1',
                  target_cnu1, mat_cnu1)]
        for itest in tests:
            info, target, decomp = itest[0], itest[1], itest[2]
            with self.subTest(i=info):
                self.log.info(info)
                self.assertTrue(matrix_equal(target, decomp, ignore_phase=True))
