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
from math import pi
from inspect import signature
import numpy as np
from ddt import ddt, data

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit, execute,
                    BasicAer)
from qiskit.test import QiskitTestCase
from qiskit.circuit import ControlledGate
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal, is_unitary_matrix
import qiskit.circuit.add_control as ac
from qiskit.transpiler.passes import Unroller
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.converters.instruction_to_gate import instruction_to_gate
from qiskit.quantum_info import Operator
from qiskit.extensions.standard import (CnotGate, XGate, YGate, ZGate, U1Gate,
                                        CyGate, CzGate, Cu1Gate, SwapGate,
                                        ToffoliGate, HGate, RZGate, FredkinGate,
                                        U3Gate, CHGate, CrzGate, Cu3Gate)
from qiskit.extensions.unitary import UnitaryGate


@ddt
class TestControlledGate(QiskitTestCase):
    """ControlledGate tests."""

    def test_controlled_x(self):
        """Test creation of controlled x gate"""
        self.assertEqual(XGate().q_if(), CnotGate())

    def test_controlled_y(self):
        """Test creation of controlled y gate"""
        self.assertEqual(YGate().q_if(), CyGate())

    def test_controlled_z(self):
        """Test creation of controlled z gate"""
        self.assertEqual(ZGate().q_if(), CzGate())

    def test_controlled_h(self):
        """Test creation of controlled h gate"""
        self.assertEqual(HGate().q_if(), CHGate())

    def test_controlled_u1(self):
        """Test creation of controlled u1 gate"""
        theta = 0.5
        self.assertEqual(U1Gate(theta).q_if(), Cu1Gate(theta))

    def test_controlled_rz(self):
        """Test creation of controlled rz gate"""
        theta = 0.5
        self.assertEqual(RZGate(theta).q_if(), CrzGate(theta))

    def test_controlled_u3(self):
        """Test creation of controlled u3 gate"""
        theta, phi, lamb = 0.1, 0.2, 0.3
        self.assertEqual(U3Gate(theta, phi, lamb).q_if(),
                         Cu3Gate(theta, phi, lamb))

    def test_controlled_cx(self):
        """Test creation of controlled cx gate"""
        self.assertEqual(CnotGate().q_if(), ToffoliGate())

    def test_controlled_swap(self):
        """Test creation of controlled swap gate"""
        self.assertEqual(SwapGate().q_if(), FredkinGate())

    def test_circuit_append(self):
        """Test appending controlled gate to quantum circuit"""
        circ = QuantumCircuit(5)
        inst = CnotGate()
        circ.append(inst.q_if(), qargs=[0, 2, 1])
        circ.append(inst.q_if(2), qargs=[0, 3, 1, 2])
        circ.append(inst.q_if().q_if(), qargs=[0, 3, 1, 2])  # should be same as above
        self.assertEqual(circ[1][0], circ[2][0])
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

    def test_definition_specification(self):
        """Test instantiation with explicit definition"""
        swap = SwapGate()
        cswap = ControlledGate('cswap', 3, [], num_ctrl_qubits=1,
                               definition=swap.definition)
        self.assertEqual(swap.definition, cswap.definition)

    def test_multi_controlled_composite_gate(self):
        """Test multi controlled composite gate"""
        num_ctrl = 3
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name='cgate')
        cgate.h(sub_q[0])
        cgate.crz(np.pi/2, sub_q[0], sub_q[1])
        cgate.swap(sub_q[0], sub_q[1])
        cgate.u3(0.1, 0.2, 0.3, sub_q[1])
        cgate.t(sub_q[0])
        num_target = cgate.width()
        gate = instruction_to_gate(cgate.to_instruction())
        cont_gate = gate.q_if(num_ctrl_qubits=num_ctrl)
        control = QuantumRegister(num_ctrl)
        target = QuantumRegister(num_target)
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:]+target[:])
        simulator = BasicAer.get_backend('unitary_simulator')
        op_mat = execute(cgate, simulator).result().get_unitary(0)
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = execute(qc, simulator).result().get_unitary(0)
        self.assertTrue(matrix_equal(cop_mat, ref_mat, ignore_phase=True))

    def test_single_controlled_composite_gate(self):
        """Test singly controlled composite gate"""
        num_ctrl = 1
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name='cgate')
        cgate.h(sub_q[0])
        cgate.cx(sub_q[0], sub_q[1])
        num_target = cgate.width()
        gate = instruction_to_gate(cgate.to_instruction())
        cont_gate = gate.q_if(num_ctrl_qubits=num_ctrl)
        control = QuantumRegister(num_ctrl)
        target = QuantumRegister(num_target)
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:]+target[:])
        simulator = BasicAer.get_backend('unitary_simulator')
        op_mat = execute(cgate, simulator).result().get_unitary(0)
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = execute(qc, simulator).result().get_unitary(0)
        self.assertTrue(matrix_equal(cop_mat, ref_mat, ignore_phase=True))

    def test_multi_control_u3(self):
        """test multi controlled u3 gate"""
        import qiskit.extensions.standard.u3 as u3
        import qiskit.extensions.standard.cu3 as cu3

        num_ctrl = 3
        # U3 gate params
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
        target_cnu3 = _compute_control_matrix(mat_u3, num_ctrl)
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
                self.assertTrue(matrix_equal(target, decomp, ignore_phase=True))

    def test_multi_control_u1(self):
        """Test multi controlled u1 gate"""
        import qiskit.extensions.standard.u1 as u1
        import qiskit.extensions.standard.cu1 as cu1

        num_ctrl = 3
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
        target_cnu1 = _compute_control_matrix(mat_u1, num_ctrl)
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

    def test_rotation_gates(self):
        """Test controlled rotation gates"""
        import qiskit.extensions.standard.u1 as u1
        import qiskit.extensions.standard.rx as rx
        import qiskit.extensions.standard.ry as ry
        import qiskit.extensions.standard.rz as rz
        num_ctrl = 2
        num_target = 1
        qreg = QuantumRegister(num_ctrl + num_target)

        gu1 = u1.U1Gate(pi)
        grx = rx.RXGate(pi)
        gry = ry.RYGate(pi)
        grz = rz.RZGate(pi)

        ugu1 = ac._unroll_gate(gu1, ['u1', 'u3', 'cx'])
        ugrx = ac._unroll_gate(grx, ['u1', 'u3', 'cx'])
        ugry = ac._unroll_gate(gry, ['u1', 'u3', 'cx'])
        ugrz = ac._unroll_gate(grz, ['u1', 'u3', 'cx'])

        cgu1 = ugu1.q_if(num_ctrl)
        cgrx = ugrx.q_if(num_ctrl)
        cgry = ugry.q_if(num_ctrl)
        cgrz = ugrz.q_if(num_ctrl)

        simulator = BasicAer.get_backend('unitary_simulator')
        for gate, cgate in zip([gu1, grx, gry, grz], [cgu1, cgrx, cgry, cgrz]):
            with self.subTest(i=gate.name):
                qc = QuantumCircuit(num_target)
                qc.append(gate, qc.qregs[0])
                op_mat = execute(qc, simulator).result().get_unitary(0)
                cqc = QuantumCircuit(num_ctrl + num_target)
                cqc.append(cgate, cqc.qregs[0])
                ref_mat = execute(cqc, simulator).result().get_unitary(0)
                cop_mat = _compute_control_matrix(op_mat, num_ctrl)
                self.assertTrue(matrix_equal(cop_mat, ref_mat,
                                             ignore_phase=True))
                dag = circuit_to_dag(cqc)
                unroller = Unroller(['u3', 'cx'])
                uqc = dag_to_circuit(unroller.run(dag))
                self.log.info('%s gate count: %d', cgate.name, uqc.size())
                self.log.info('\n%s', str(uqc))
                # these limits could be changed
                if gate.name == 'ry':
                    self.assertTrue(uqc.size() <= 32)
                else:
                    self.assertTrue(uqc.size() <= 20)
        qc = QuantumCircuit(qreg, name='composite')
        qc.append(grx.q_if(num_ctrl), qreg)
        qc.append(gry.q_if(num_ctrl), qreg)
        qc.append(gry, qreg[0:gry.num_qubits])
        qc.append(grz.q_if(num_ctrl), qreg)

        dag = circuit_to_dag(qc)
        unroller = Unroller(['u3', 'cx'])
        uqc = dag_to_circuit(unroller.run(dag))
        self.log.info('%s gate count: %d', uqc.name, uqc.size())
        self.assertTrue(uqc.size() <= 73)  # this limit could be changed

    @data(1, 2, 3, 4)
    def test_inverse_x(self, num_ctrl_qubits):
        """test inverting ControlledGate"""
        cnx = XGate().q_if(num_ctrl_qubits)
        inv_cnx = cnx.inverse()
        result = Operator(cnx).compose(Operator(inv_cnx))
        np.testing.assert_array_almost_equal(result.data,
                                             np.identity(result.dim[0]))

    @data(1, 2, 3)
    def test_inverse_circuit(self, num_ctrl_qubits):
        """test inverting ControlledGate"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rx(np.pi/4, [0, 1, 2])
        gate = instruction_to_gate(qc.to_instruction())
        cgate = gate.q_if(num_ctrl_qubits)
        inv_cgate = cgate.inverse()
        result = Operator(cgate).compose(Operator(inv_cgate))
        np.testing.assert_array_almost_equal(result.data,
                                             np.identity(result.dim[0]))

    @data(1, 2, 3, 4, 5)
    def test_controlled_unitary(self, num_ctrl_qubits):
        """test controlled unitary"""
        num_target = 1
        q_target = QuantumRegister(num_target)
        qc1 = QuantumCircuit(q_target)
        # for h-rx(pi/2)
        theta, phi, lamb = 1.57079632679490, 0.0, 4.71238898038469
        qc1.u3(theta, phi, lamb, q_target[0])
        base_gate = instruction_to_gate(qc1.to_instruction())
        # get UnitaryGate version of circuit
        base_op = Operator(qc1)
        base_mat = base_op.data
        cgate = base_gate.q_if(num_ctrl_qubits)
        test_op = Operator(cgate)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(is_unitary_matrix(base_mat))
        self.assertTrue(matrix_equal(cop_mat, test_op.data, ignore_phase=True))

    @data(1, 2, 3)
    def test_global_phase(self, num_ctrl_qubits):
        """test global phase"""
        mat1 = np.array([[0, 1], [1, 0]], dtype=float)
        mat2 = np.exp(1j) * mat1
        gate1 = UnitaryGate(mat1)
        gate2 = UnitaryGate(mat2)
        cgate1 = gate1.q_if(num_ctrl_qubits)
        cgate2 = gate2.q_if(num_ctrl_qubits)
        cop_mat1 = _compute_control_matrix(mat1, num_ctrl_qubits)
        cop_mat2 = _compute_control_matrix(mat2, num_ctrl_qubits, phase=1)
        self.assertTrue(is_unitary_matrix(mat1))
        self.assertTrue(is_unitary_matrix(mat2))
        self.assertTrue(is_unitary_matrix(cop_mat1))
        self.assertTrue(is_unitary_matrix(cop_mat2))
        self.assertTrue(Operator(cgate1).equiv(Operator(cgate2)))
        self.assertTrue(matrix_equal(cop_mat1, cop_mat2, ignore_phase=True))

    def test_base_gate_setting(self):
        """
        Test all gates in standard extensions which are of type ControlledGate
        have a base gate setting.
        """
        params = [0.1 * i for i in range(10)]
        for gate_class in ControlledGate.__subclasses__():
            sig = signature(gate_class.__init__)
            free_params = len(sig.parameters) - 1  # subtract "self"
            base_gate = gate_class(*params[0:free_params])
            cgate = base_gate.q_if()
            self.assertEqual(base_gate.base_gate, cgate.base_gate)


def _compute_control_matrix(base_mat, num_ctrl_qubits, phase=0):
    """
    Compute the controlled version of the input matrix with qiskit ordering.

    Args:
        base_mat (ndarray): unitary to be controlled
        num_ctrl_qubits (int): number of controls for new unitary
        phase (float): The global phase of base_mat which is promoted to the
            global phase of the controlled matrix

    Returns:
        ndarray: controlled version of base matrix.
    """
    num_target = int(np.log2(base_mat.shape[0]))
    ctrl_dim = 2**num_ctrl_qubits
    ctrl_grnd = np.repeat([[1], [0]], [1, ctrl_dim-1])
    full_mat_dim = ctrl_dim * base_mat.shape[0]
    full_mat = np.zeros((full_mat_dim, full_mat_dim), dtype=base_mat.dtype)
    for i in range(ctrl_dim-1):
        full_mat += np.kron(np.eye(2**num_target),
                            np.diag(np.roll(ctrl_grnd, i)))
    if phase != 0:
        full_mat = np.exp(1j * phase) * full_mat
    full_mat += np.kron(base_mat, np.diag(np.roll(ctrl_grnd, ctrl_dim-1)))
    return full_mat
