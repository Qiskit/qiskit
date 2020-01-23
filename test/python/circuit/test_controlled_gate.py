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


"""Test Qiskit's inverse gate operation."""

import unittest
import itertools
from inspect import signature
from test import combine
import numpy as np
from numpy import pi
from numpy.testing import assert_allclose
import scipy
from ddt import ddt, data

from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer, QiskitError
from qiskit.test import QiskitTestCase
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.operators.predicates import matrix_equal, is_unitary_matrix
import qiskit.circuit.add_control as ac
from qiskit.transpiler.passes import Unroller
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.extensions.standard import (CnotGate, XGate, YGate, ZGate, U1Gate,
                                        CyGate, CzGate, Cu1Gate, SwapGate,
                                        ToffoliGate, HGate, RZGate, RXGate,
                                        RYGate, CryGate, CrxGate, FredkinGate,
                                        U3Gate, CHGate, CrzGate, Cu3Gate,
                                        MSGate, Barrier)
from qiskit.extensions.unitary import UnitaryGate
import qiskit.extensions.standard as allGates


@ddt
class TestControlledGate(QiskitTestCase):
    """Tests for controlled gates and the ControlledGate class."""

    def test_controlled_x(self):
        """Test the creation of a controlled X gate."""
        self.assertEqual(XGate().control(), CnotGate())

    def test_controlled_y(self):
        """Test the creation of a controlled Y gate."""
        self.assertEqual(YGate().control(), CyGate())

    def test_controlled_z(self):
        """Test the creation of a controlled Z gate."""
        self.assertEqual(ZGate().control(), CzGate())

    def test_controlled_h(self):
        """Test the creation of a controlled H gate."""
        self.assertEqual(HGate().control(), CHGate())

    def test_controlled_u1(self):
        """Test the creation of a controlled U1 gate."""
        theta = 0.5
        self.assertEqual(U1Gate(theta).control(), Cu1Gate(theta))

    def test_controlled_rz(self):
        """Test the creation of a controlled RZ gate."""
        theta = 0.5
        self.assertEqual(RZGate(theta).control(), CrzGate(theta))

    def test_controlled_ry(self):
        """Test the creation of a controlled RY gate."""
        theta = 0.5
        self.assertEqual(RYGate(theta).control(), CryGate(theta))

    def test_controlled_rx(self):
        """Test the creation of a controlled RX gate."""
        theta = 0.5
        self.assertEqual(RXGate(theta).control(), CrxGate(theta))

    def test_controlled_u3(self):
        """Test the creation of a controlled U3 gate."""
        theta, phi, lamb = 0.1, 0.2, 0.3
        self.assertEqual(U3Gate(theta, phi, lamb).control(),
                         Cu3Gate(theta, phi, lamb))

    def test_controlled_cx(self):
        """Test the creation of a controlled CX gate."""
        self.assertEqual(CnotGate().control(), ToffoliGate())

    def test_controlled_swap(self):
        """Test the creation of a controlled Swap gate."""
        self.assertEqual(SwapGate().control(), FredkinGate())

    def test_circuit_append(self):
        """Test appending a controlled gate to a quantum circuit."""
        circ = QuantumCircuit(5)
        inst = CnotGate()
        circ.append(inst.control(), qargs=[0, 2, 1])
        circ.append(inst.control(2), qargs=[0, 3, 1, 2])
        circ.append(inst.control().control(), qargs=[0, 3, 1, 2])  # should be same as above
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

    def test_swap_definition_specification(self):
        """Test the instantiation of a controlled swap gate with explicit definition."""
        swap = SwapGate()
        cswap = ControlledGate('cswap', 3, [], num_ctrl_qubits=1,
                               definition=swap.definition)
        self.assertEqual(swap.definition, cswap.definition)

    def test_multi_controlled_composite_gate(self):
        """Test a multi controlled composite gate. """
        num_ctrl = 3
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name='cgate')
        cgate.h(sub_q[0])
        cgate.crz(pi / 2, sub_q[0], sub_q[1])
        cgate.swap(sub_q[0], sub_q[1])
        cgate.u3(0.1, 0.2, 0.3, sub_q[1])
        cgate.t(sub_q[0])
        num_target = cgate.width()
        gate = cgate.to_gate()
        cont_gate = gate.control(num_ctrl_qubits=num_ctrl)
        control = QuantumRegister(num_ctrl)
        target = QuantumRegister(num_target)
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:] + target[:])
        simulator = BasicAer.get_backend('unitary_simulator')
        op_mat = execute(cgate, simulator).result().get_unitary(0)
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = execute(qc, simulator).result().get_unitary(0)
        self.assertTrue(matrix_equal(cop_mat, ref_mat, ignore_phase=True))

    def test_single_controlled_composite_gate(self):
        """Test a singly controlled composite gate."""
        num_ctrl = 1
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name='cgate')
        cgate.h(sub_q[0])
        cgate.cx(sub_q[0], sub_q[1])
        num_target = cgate.width()
        gate = cgate.to_gate()
        cont_gate = gate.control(num_ctrl_qubits=num_ctrl)
        control = QuantumRegister(num_ctrl)
        target = QuantumRegister(num_target)
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:] + target[:])
        simulator = BasicAer.get_backend('unitary_simulator')
        op_mat = execute(cgate, simulator).result().get_unitary(0)
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = execute(qc, simulator).result().get_unitary(0)
        self.assertTrue(matrix_equal(cop_mat, ref_mat, ignore_phase=True))

    def test_controlled_u3_matrix(self):
        """Test the matrix representation of the controlled and controlled-controlled U3 gate."""
        import qiskit.extensions.standard.u3 as u3
        import qiskit.extensions.standard.cu3 as cu3

        num_ctrl = 3
        # U3 gate params
        alpha, beta, gamma = 0.2, 0.3, 0.4

        # cnu3 gate
        u3gate = u3.U3Gate(alpha, beta, gamma)
        cnu3 = u3gate.control(num_ctrl)
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

        c_cu3 = cu3gate.control(1)
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

        tests = [('check unitary of u3.control against tensored unitary of u3',
                  target_cu3, mat_cu3),
                 ('check unitary of cu3.control against tensored unitary of cu3',
                  target_c_cu3, mat_c_cu3),
                 ('check unitary of cnu3 against tensored unitary of u3',
                  target_cnu3, mat_cnu3)]
        for itest in tests:
            info, target, decomp = itest[0], itest[1], itest[2]
            with self.subTest(i=info):
                self.log.info(info)
                self.assertTrue(matrix_equal(target, decomp, ignore_phase=True))

    def test_controlled_u1_matrix(self):
        """Test the matrix representation of the controlled and controlled-controlled U1 gate."""
        import qiskit.extensions.standard.u1 as u1
        import qiskit.extensions.standard.cu1 as cu1

        num_ctrl = 3
        # U1 gate params
        theta = 0.2

        # cnu1 gate
        u1gate = u1.U1Gate(theta)
        cnu1 = u1gate.control(num_ctrl)
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
        c_cu1 = cu1gate.control(1)
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

        tests = [('check unitary of u1.control against tensored unitary of u1',
                  target_cu1, mat_cu1),
                 ('check unitary of cu1.control against tensored unitary of cu1',
                  target_c_cu1, mat_c_cu1),
                 ('check unitary of cnu1 against tensored unitary of u1',
                  target_cnu1, mat_cnu1)]
        for itest in tests:
            info, target, decomp = itest[0], itest[1], itest[2]
            with self.subTest(i=info):
                self.log.info(info)
                self.assertTrue(matrix_equal(target, decomp, ignore_phase=True))

    @data(1, 2, 3, 4)
    def test_multi_controlled_u1_matrix(self, num_controls):
        """Test the matrix representation of the multi-controlled CU1 gate."""

        # registers for the circuit
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)

        # iterate over all possible combinations of control qubits
        allsubsets = list(itertools.chain(*[itertools.combinations(range(num_controls), ni)
                                            for ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            lam = 0.3165354 * pi
            qc = QuantumCircuit(q_target, q_controls)
            for idx in subset:
                control_int += 2**idx
                qc.x(q_controls[idx])

            qc.h(q_target[0])
            qc.mcu1(lam, q_controls, q_target[0])
            qc.h(q_target[0])

            for idx in subset:
                qc.x(q_controls[idx])

            backend = BasicAer.get_backend('unitary_simulator')
            mat_mcu = execute(qc, backend).result().get_unitary(qc)

            dim = 2**(num_controls + 1)
            pos = dim - 2 * (control_int + 1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            fac = np.exp(1j * lam)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = [[(1 + fac) / 2, (1 - fac) / 2],
                                                         [(1 - fac) / 2, (1 + fac) / 2]]
            assert_allclose(mat_mcu, mat_groundtruth)

    @combine(num_controls=[1, 2, 3], mode=['basic'])
    def test_multi_control_toffoli_matrix_clean_ancillas(self, num_controls, mode):
        """Test the multi-control Toffoli gate with clean ancillas."""

        # set up circuit
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        qc = QuantumCircuit(q_target, q_controls)

        # add ancillas if necessary
        num_ancillas = 0 if num_controls <= 2 else num_controls - 2
        q_ancillas = None
        if num_ancillas > 0:
            q_ancillas = QuantumRegister(num_ancillas)
            qc.add_register(q_ancillas)

        # apply hadamard on control qubits and toffoli gate
        qc.h(q_controls)
        qc.mct(q_controls, q_target[0], q_ancillas, mode=mode)

        # execute the circuit and obtain statevector result
        backend = BasicAer.get_backend('statevector_simulator')
        vec_mct = execute(qc, backend).result().get_statevector(qc)

        # compare to expectation
        mat = np.eye(2 ** (num_controls + 1))
        mat[-2:, -2:] = [[0, 1], [1, 0]]
        if num_ancillas > 0:
            mat = np.kron(np.eye(2 ** num_ancillas), mat)

        vec_groundtruth = mat @ np.kron(np.kron(
            np.array([1] + [0] * (2 ** num_ancillas - 1)),
            [1 / 2 ** (num_controls / 2)] * 2 ** num_controls), [1, 0])

        s_f = state_fidelity(vec_mct, vec_groundtruth)
        self.assertAlmostEqual(s_f, 1)

    @combine(num_controls=[1, 2, 3, 4, 5], mode=['basic-dirty-ancilla', 'advanced', 'noancilla'])
    def test_multi_control_toffoli_matrix_dirty_ancillas(self, num_controls, mode):
        """Test the multi-control Toffoli gate with dirty ancillas."""
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        qc = QuantumCircuit(q_target, q_controls)

        if mode == 'basic-dirty-ancilla':
            if num_controls <= 2:
                num_ancillas = 0
            else:
                num_ancillas = num_controls - 2
        elif mode == 'noancilla':
            num_ancillas = 0
        else:
            if num_controls <= 4:
                num_ancillas = 0
            else:
                num_ancillas = 1

        q_ancillas = None
        if num_ancillas > 0:
            q_ancillas = QuantumRegister(num_ancillas)
            qc.add_register(q_ancillas)

        qc.mct(q_controls, q_target[0], q_ancillas, mode=mode)

        mat_mct = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary(qc)

        mat_groundtruth = np.eye(2 ** (num_controls + 1))
        mat_groundtruth[-2:, -2:] = [[0, 1], [1, 0]]
        if num_ancillas > 0:
            mat_groundtruth = np.kron(np.eye(2 ** num_ancillas), mat_groundtruth)

        assert_allclose(mat_mct, mat_groundtruth, atol=1e-08)

    def test_single_controlled_rotation_gates(self):
        """Test the controlled rotation gates controlled on one qubit."""
        import qiskit.extensions.standard.u1 as u1
        import qiskit.extensions.standard.rx as rx
        import qiskit.extensions.standard.ry as ry
        import qiskit.extensions.standard.rz as rz
        num_ctrl = 2
        num_target = 1
        qreg = QuantumRegister(num_ctrl + num_target)

        theta = pi
        gu1 = u1.U1Gate(theta)
        grx = rx.RXGate(theta)
        gry = ry.RYGate(theta)
        grz = rz.RZGate(theta)

        ugu1 = ac._unroll_gate(gu1, ['u1', 'u3', 'cx'])
        ugrx = ac._unroll_gate(grx, ['u1', 'u3', 'cx'])
        ugry = ac._unroll_gate(gry, ['u1', 'u3', 'cx'])
        ugrz = ac._unroll_gate(grz, ['u1', 'u3', 'cx'])
        ugrz.params = grz.params

        cgu1 = ugu1.control(num_ctrl)
        cgrx = ugrx.control(num_ctrl)
        cgry = ugry.control(num_ctrl)
        cgrz = ugrz.control(num_ctrl)

        for gate, cgate in zip([gu1, grx, gry, grz], [cgu1, cgrx, cgry, cgrz]):
            with self.subTest(i=gate.name):
                if gate.name == 'rz':
                    iden = Operator.from_label('I')
                    zgen = Operator.from_label('Z')
                    op_mat = (np.cos(0.5 * theta) * iden - 1j * np.sin(0.5 * theta) * zgen).data
                else:
                    op_mat = Operator(gate).data
                ref_mat = Operator(cgate).data
                cop_mat = _compute_control_matrix(op_mat, num_ctrl)
                self.assertTrue(matrix_equal(cop_mat, ref_mat,
                                             ignore_phase=True))
                cqc = QuantumCircuit(num_ctrl + num_target)
                cqc.append(cgate, cqc.qregs[0])
                dag = circuit_to_dag(cqc)
                unroller = Unroller(['u3', 'cx'])
                uqc = dag_to_circuit(unroller.run(dag))
                self.log.info('%s gate count: %d', cgate.name, uqc.size())
                self.log.info('\n%s', str(uqc))
                # these limits could be changed
                if gate.name == 'ry':
                    self.assertTrue(uqc.size() <= 32)
                elif gate.name == 'rz':
                    self.assertTrue(uqc.size() <= 40)
                else:
                    self.assertTrue(uqc.size() <= 20)
        qc = QuantumCircuit(qreg, name='composite')
        qc.append(grx.control(num_ctrl), qreg)
        qc.append(gry.control(num_ctrl), qreg)
        qc.append(gry, qreg[0:gry.num_qubits])
        qc.append(grz.control(num_ctrl), qreg)

        dag = circuit_to_dag(qc)
        unroller = Unroller(['u3', 'cx'])
        uqc = dag_to_circuit(unroller.run(dag))
        self.log.info('%s gate count: %d', uqc.name, uqc.size())
        self.assertTrue(uqc.size() <= 93)  # this limit could be changed

    @combine(num_controls=[1, 2, 4],
             base_gate_name=['x', 'y', 'z'],
             use_basis_gates=[True, False])
    def test_multi_controlled_rotation_gate_matrices(self, num_controls, base_gate_name,
                                                     use_basis_gates):
        """Test the multi controlled rotation gates without ancillas."""
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        allsubsets = list(itertools.chain(*[itertools.combinations(range(num_controls), ni) for
                                            ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = 0.871236 * pi
            qc = QuantumCircuit(q_target, q_controls)
            for idx in subset:
                control_int += 2**idx
                qc.x(q_controls[idx])

            # call mcrx/mcry/mcrz
            if base_gate_name == 'y':
                qc.mcry(theta, q_controls, q_target[0], None, mode='noancilla',
                        use_basis_gates=use_basis_gates)
            else:  # case 'x' or 'z' only support the noancilla mode and do not have this keyword
                getattr(qc, 'mcr' + base_gate_name)(theta, q_controls, q_target[0],
                                                    use_basis_gates=use_basis_gates)

            for idx in subset:
                qc.x(q_controls[idx])

            backend = BasicAer.get_backend('unitary_simulator')
            mat_mcu = execute(qc, backend).result().get_unitary(qc)

            dim = 2**(num_controls + 1)
            pos = dim - 2 * (control_int + 1)
            mat_groundtruth = np.eye(dim, dtype=complex)

            if base_gate_name == 'x':
                rot_mat = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                                    [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
                                   dtype=complex)
            elif base_gate_name == 'y':
                rot_mat = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                                    [np.sin(theta / 2), np.cos(theta / 2)]],
                                   dtype=complex)
            else:  # case 'z'
                rot_mat = np.array([[1, 0],
                                    [0, np.exp(1j * theta)]],
                                   dtype=complex)

            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            assert_allclose(mat_mcu, mat_groundtruth)

    @combine(num_controls=[1, 2, 4], use_basis_gates=[True, False])
    def test_multi_controlled_y_rotation_matrix_basic_mode(self, num_controls, use_basis_gates):
        """Test the multi controlled Y rotation using the mode 'basic'."""

        # get the number of required ancilla qubits
        if num_controls <= 2:
            num_ancillas = 0
        else:
            num_ancillas = num_controls - 2

        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        allsubsets = list(itertools.chain(*[itertools.combinations(range(num_controls), ni) for
                                            ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = 0.871236 * pi
            qc = QuantumCircuit(q_target, q_controls)
            if num_ancillas > 0:
                q_ancillas = QuantumRegister(num_ancillas)
                qc.add_register(q_ancillas)
            else:
                q_ancillas = None

            for idx in subset:
                control_int += 2**idx
                qc.x(q_controls[idx])

            qc.mcry(theta, q_controls, q_target[0], q_ancillas, mode='basic',
                    use_basis_gates=use_basis_gates)

            for idx in subset:
                qc.x(q_controls[idx])

            backend = BasicAer.get_backend('unitary_simulator')
            mat_mcu = execute(qc, backend).result().get_unitary(qc)

            dim = 2**(num_controls + 1)
            mat_mcu = mat_mcu[:dim, :dim]
            pos = dim - 2 * (control_int + 1)
            mat_groundtruth = np.eye(dim, dtype=complex)

            rot_mat = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                                [np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)

            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            assert_allclose(mat_mcu, mat_groundtruth)

    @data(1, 2, 3, 4)
    def test_inverse_x(self, num_ctrl_qubits):
        """Test inverting the controlled X gate."""
        cnx = XGate().control(num_ctrl_qubits)
        inv_cnx = cnx.inverse()
        result = Operator(cnx).compose(Operator(inv_cnx))
        np.testing.assert_array_almost_equal(result.data,
                                             np.identity(result.dim[0]))

    @data(1, 2, 3)
    def test_inverse_circuit(self, num_ctrl_qubits):
        """Test inverting a controlled gate based on a circuit definition."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rx(np.pi / 4, [0, 1, 2])
        gate = qc.to_gate()
        cgate = gate.control(num_ctrl_qubits)
        inv_cgate = cgate.inverse()
        result = Operator(cgate).compose(Operator(inv_cgate))
        np.testing.assert_array_almost_equal(result.data,
                                             np.identity(result.dim[0]))

    @data(1, 2, 3, 4, 5)
    def test_controlled_unitary(self, num_ctrl_qubits):
        """Test the matrix data of an Operator, which is based on a controlled gate."""
        num_target = 1
        q_target = QuantumRegister(num_target)
        qc1 = QuantumCircuit(q_target)
        # for h-rx(pi/2)
        theta, phi, lamb = 1.57079632679490, 0.0, 4.71238898038469
        qc1.u3(theta, phi, lamb, q_target[0])
        base_gate = qc1.to_gate()
        # get UnitaryGate version of circuit
        base_op = Operator(qc1)
        base_mat = base_op.data
        cgate = base_gate.control(num_ctrl_qubits)
        test_op = Operator(cgate)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(is_unitary_matrix(base_mat))
        self.assertTrue(matrix_equal(cop_mat, test_op.data, ignore_phase=True))

    @data(1, 2, 3, 4, 5)
    def test_controlled_random_unitary(self, num_ctrl_qubits):
        """Test the matrix data of an Operator based on a random UnitaryGate."""
        num_target = 2
        base_gate = UnitaryGate(scipy.stats.unitary_group.rvs(num_target))
        base_mat = base_gate.to_matrix()
        cgate = base_gate.control(num_ctrl_qubits)
        test_op = Operator(cgate)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op.data, ignore_phase=True))

    def test_base_gate_setting(self):
        """
        Test all gates in standard extensions which are of type ControlledGate and have a base gate
        setting.
        """
        params = [0.1 * i for i in range(10)]
        for gate_class in ControlledGate.__subclasses__():
            sig = signature(gate_class.__init__)
            free_params = len(sig.parameters) - 1  # subtract "self"
            base_gate = gate_class(*params[0:free_params])
            cgate = base_gate.control()
            self.assertEqual(base_gate.base_gate, cgate.base_gate)

    def test_all_inverses(self):
        """
        Test all gates in standard extensions except those that cannot be controlled or are being
        deprecated.
        """
        gate_classes = [cls for name, cls in allGates.__dict__.items()
                        if isinstance(cls, type)]
        for cls in gate_classes:
            # only verify basic gates right now, as already controlled ones
            # will generate differing definitions
            if issubclass(cls, ControlledGate) or cls == allGates.IdGate:
                continue
            try:
                sig = signature(cls)
                numargs = len([param for param in sig.parameters.values()
                               if param.kind == param.POSITIONAL_ONLY or
                               (param.kind == param.POSITIONAL_OR_KEYWORD and
                                param.default is param.empty)])
                args = [1] * numargs

                gate = cls(*args)
                self.assertEqual(gate.inverse().control(2),
                                 gate.control(2).inverse())
            except AttributeError:
                # skip gates that do not have a control attribute (e.g. barrier)
                pass

    @data(1, 2, 3)
    def test_controlled_standard_gates(self, num_ctrl_qubits):
        """Test the controlled versions of all standard gates."""
        gate_classes = [cls for name, cls in allGates.__dict__.items()
                        if isinstance(cls, type)]
        theta = pi / 2
        for cls in gate_classes:
            with self.subTest(i=cls):
                sig = signature(cls)
                numargs = len([param for param in sig.parameters.values()
                               if param.kind == param.POSITIONAL_ONLY or
                               (param.kind == param.POSITIONAL_OR_KEYWORD and
                                param.default is param.empty)])
                args = [theta] * numargs
                if cls in [MSGate, Barrier]:
                    args[0] = 2
                gate = cls(*args)
                try:
                    cgate = gate.control(num_ctrl_qubits)
                except (AttributeError, QiskitError):
                    # 'object has no attribute "control"'
                    # skipping Id and Barrier
                    continue
                if gate.name == 'rz':
                    iden = Operator.from_label('I')
                    zgen = Operator.from_label('Z')
                    base_mat = (np.cos(0.5 * theta) * iden - 1j * np.sin(0.5 * theta) * zgen).data
                else:
                    base_mat = Operator(gate).data
                target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
                self.assertTrue(matrix_equal(Operator(cgate).data, target_mat, ignore_phase=True))


def _compute_control_matrix(base_mat, num_ctrl_qubits):
    """
    Compute the controlled version of the input matrix with qiskit ordering.

    Args:
        base_mat (ndarray): unitary to be controlled
        num_ctrl_qubits (int): number of controls for new unitary

    Returns:
        ndarray: controlled version of base matrix.
    """
    num_target = int(np.log2(base_mat.shape[0]))
    ctrl_dim = 2**num_ctrl_qubits
    ctrl_grnd = np.repeat([[1], [0]], [1, ctrl_dim - 1])
    full_mat_dim = ctrl_dim * base_mat.shape[0]
    full_mat = np.zeros((full_mat_dim, full_mat_dim), dtype=base_mat.dtype)
    ctrl_proj = np.diag(np.roll(ctrl_grnd, ctrl_dim - 1))
    full_mat = (np.kron(np.eye(2**num_target),
                        np.eye(ctrl_dim) - ctrl_proj) +
                np.kron(base_mat, ctrl_proj))
    return full_mat


if __name__ == '__main__':
    unittest.main()
