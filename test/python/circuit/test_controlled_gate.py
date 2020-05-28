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
from test import combine
import numpy as np
from numpy import pi
from ddt import ddt, data, unpack

from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer, QiskitError
from qiskit.test import QiskitTestCase
from qiskit.circuit import ControlledGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.predicates import matrix_equal, is_unitary_matrix
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.states import Statevector
import qiskit.circuit.add_control as ac
from qiskit.transpiler.passes import Unroller
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.extensions.standard import (CXGate, XGate, YGate, ZGate, U1Gate,
                                        CYGate, CZGate, CU1Gate, SwapGate,
                                        CCXGate, HGate, RZGate, RXGate,
                                        RYGate, CRYGate, CRXGate, CSwapGate,
                                        U3Gate, CHGate, CRZGate, CU3Gate,
                                        MSGate, Barrier, RCCXGate, RC3XGate,
                                        MCU1Gate, MCXGate, MCXGrayCode, MCXRecursive,
                                        MCXVChain, C3XGate, C4XGate)
from qiskit.circuit._utils import _compute_control_matrix
import qiskit.extensions.standard as allGates

from .gate_utils import _get_free_params


@ddt
class TestControlledGate(QiskitTestCase):
    """Tests for controlled gates and the ControlledGate class."""

    def test_controlled_x(self):
        """Test creation of controlled x gate"""
        self.assertEqual(XGate().control(), CXGate())

    def test_controlled_y(self):
        """Test creation of controlled y gate"""
        self.assertEqual(YGate().control(), CYGate())

    def test_controlled_z(self):
        """Test creation of controlled z gate"""
        self.assertEqual(ZGate().control(), CZGate())

    def test_controlled_h(self):
        """Test the creation of a controlled H gate."""
        self.assertEqual(HGate().control(), CHGate())

    def test_controlled_u1(self):
        """Test the creation of a controlled U1 gate."""
        theta = 0.5
        self.assertEqual(U1Gate(theta).control(), CU1Gate(theta))

    def test_controlled_rz(self):
        """Test the creation of a controlled RZ gate."""
        theta = 0.5
        self.assertEqual(RZGate(theta).control(), CRZGate(theta))

    def test_controlled_ry(self):
        """Test the creation of a controlled RY gate."""
        theta = 0.5
        self.assertEqual(RYGate(theta).control(), CRYGate(theta))

    def test_controlled_rx(self):
        """Test the creation of a controlled RX gate."""
        theta = 0.5
        self.assertEqual(RXGate(theta).control(), CRXGate(theta))

    def test_controlled_u3(self):
        """Test the creation of a controlled U3 gate."""
        theta, phi, lamb = 0.1, 0.2, 0.3
        self.assertEqual(U3Gate(theta, phi, lamb).control(),
                         CU3Gate(theta, phi, lamb))

    def test_controlled_cx(self):
        """Test creation of controlled cx gate"""
        self.assertEqual(CXGate().control(), CCXGate())

    def test_controlled_swap(self):
        """Test creation of controlled swap gate"""
        self.assertEqual(SwapGate().control(), CSwapGate())

    def test_circuit_append(self):
        """Test appending a controlled gate to a quantum circuit."""
        circ = QuantumCircuit(5)
        inst = CXGate()
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
        control = QuantumRegister(num_ctrl, 'control')
        target = QuantumRegister(num_target, 'target')
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:] + target[:])
        simulator = BasicAer.get_backend('unitary_simulator')
        op_mat = execute(cgate, simulator).result().get_unitary(0)
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = execute(qc, simulator).result().get_unitary(0)
        self.assertTrue(matrix_equal(cop_mat, ref_mat, ignore_phase=True))

    def test_multi_control_u3(self):
        """Test the matrix representation of the controlled and controlled-controlled U3 gate."""
        import qiskit.extensions.standard.u3 as u3

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
        cu3gate = u3.CU3Gate(alpha, beta, gamma)

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

    def test_multi_control_u1(self):
        """Test the matrix representation of the controlled and controlled-controlled U1 gate."""
        import qiskit.extensions.standard.u1 as u1

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
        cu1gate = u1.CU1Gate(theta)
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
        """Test the matrix representation of the multi-controlled CU1 gate.

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mcu1.py
        """

        # registers for the circuit
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)

        # iterate over all possible combinations of control qubits
        for ctrl_state in range(2 ** num_controls):
            bitstr = bin(ctrl_state)[2:].zfill(num_controls)[::-1]
            lam = 0.3165354 * pi
            qc = QuantumCircuit(q_controls, q_target)
            for idx, bit in enumerate(bitstr):
                if bit == '0':
                    qc.x(q_controls[idx])

            qc.mcu1(lam, q_controls, q_target[0])

            # for idx in subset:
            for idx, bit in enumerate(bitstr):
                if bit == '0':
                    qc.x(q_controls[idx])

            backend = BasicAer.get_backend('unitary_simulator')
            simulated = execute(qc, backend).result().get_unitary(qc)

            base = U1Gate(lam).to_matrix()
            expected = _compute_control_matrix(base, num_controls, ctrl_state=ctrl_state)
            with self.subTest(msg='control state = {}'.format(ctrl_state)):
                self.assertTrue(matrix_equal(simulated, expected))

    @data(1, 2, 3, 4)
    def test_multi_control_toffoli_matrix_clean_ancillas(self, num_controls):
        """Test the multi-control Toffoli gate with clean ancillas.

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mct.py
        """
        # set up circuit
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        qc = QuantumCircuit(q_controls, q_target)

        if num_controls > 2:
            num_ancillas = num_controls - 2
            q_ancillas = QuantumRegister(num_controls)
            qc.add_register(q_ancillas)
        else:
            num_ancillas = 0
            q_ancillas = None

        # apply hadamard on control qubits and toffoli gate
        qc.mct(q_controls, q_target[0], q_ancillas, mode='basic')

        # execute the circuit and obtain statevector result
        backend = BasicAer.get_backend('unitary_simulator')
        simulated = execute(qc, backend).result().get_unitary(qc)

        # compare to expectation
        if num_ancillas > 0:
            simulated = simulated[:2 ** (num_controls + 1), :2 ** (num_controls + 1)]

        base = XGate().to_matrix()
        expected = _compute_control_matrix(base, num_controls)
        self.assertTrue(matrix_equal(simulated, expected))

    @data(1, 2, 3, 4, 5)
    def test_multi_control_toffoli_matrix_basic_dirty_ancillas(self, num_controls):
        """Test the multi-control Toffoli gate with dirty ancillas (basic-dirty-ancilla).

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mct.py
        """
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        qc = QuantumCircuit(q_controls, q_target)

        q_ancillas = None
        if num_controls <= 2:
            num_ancillas = 0
        else:
            num_ancillas = num_controls - 2
            q_ancillas = QuantumRegister(num_ancillas)
            qc.add_register(q_ancillas)

        qc.mct(q_controls, q_target[0], q_ancillas, mode='basic-dirty-ancilla')

        simulated = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary(qc)
        if num_ancillas > 0:
            simulated = simulated[:2 ** (num_controls + 1), :2 ** (num_controls + 1)]

        base = XGate().to_matrix()
        expected = _compute_control_matrix(base, num_controls)
        self.assertTrue(matrix_equal(simulated, expected, atol=1e-8))

    @data(1, 2, 3, 4, 5)
    def test_multi_control_toffoli_matrix_advanced_dirty_ancillas(self, num_controls):
        """Test the multi-control Toffoli gate with dirty ancillas (advanced).

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mct.py
        """
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        qc = QuantumCircuit(q_controls, q_target)

        q_ancillas = None
        if num_controls <= 4:
            num_ancillas = 0
        else:
            num_ancillas = 1
            q_ancillas = QuantumRegister(num_ancillas)
            qc.add_register(q_ancillas)

        qc.mct(q_controls, q_target[0], q_ancillas, mode='advanced')

        simulated = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary(qc)
        if num_ancillas > 0:
            simulated = simulated[:2 ** (num_controls + 1), :2 ** (num_controls + 1)]

        base = XGate().to_matrix()
        expected = _compute_control_matrix(base, num_controls)
        self.assertTrue(matrix_equal(simulated, expected, atol=1e-8))

    @data(1, 2, 3)
    def test_multi_control_toffoli_matrix_noancilla_dirty_ancillas(self, num_controls):
        """Test the multi-control Toffoli gate with dirty ancillas (noancilla).

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mct.py
        """
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)
        qc = QuantumCircuit(q_controls, q_target)

        qc.mct(q_controls, q_target[0], None, mode='noancilla')

        simulated = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary(qc)

        base = XGate().to_matrix()
        expected = _compute_control_matrix(base, num_controls)
        self.assertTrue(matrix_equal(simulated, expected, atol=1e-8))

    @combine(num_controls=[1, 2, 4],
             base_gate_name=['x', 'y', 'z'],
             use_basis_gates=[True, False])
    def test_multi_controlled_rotation_gate_matrices(self, num_controls, base_gate_name,
                                                     use_basis_gates):
        """Test the multi controlled rotation gates without ancillas.

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mcr.py
        """
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)

        # iterate over all possible combinations of control qubits
        for ctrl_state in range(2 ** num_controls):
            bitstr = bin(ctrl_state)[2:].zfill(num_controls)[::-1]
            theta = 0.871236 * pi
            qc = QuantumCircuit(q_controls, q_target)
            for idx, bit in enumerate(bitstr):
                if bit == '0':
                    qc.x(q_controls[idx])

            # call mcrx/mcry/mcrz
            if base_gate_name == 'y':
                qc.mcry(theta, q_controls, q_target[0], None, mode='noancilla',
                        use_basis_gates=use_basis_gates)
            else:  # case 'x' or 'z' only support the noancilla mode and do not have this keyword
                getattr(qc, 'mcr' + base_gate_name)(theta, q_controls, q_target[0],
                                                    use_basis_gates=use_basis_gates)

            for idx, bit in enumerate(bitstr):
                if bit == '0':
                    qc.x(q_controls[idx])

            backend = BasicAer.get_backend('unitary_simulator')
            simulated = execute(qc, backend).result().get_unitary(qc)

            if base_gate_name == 'x':
                rot_mat = RXGate(theta).to_matrix()
            elif base_gate_name == 'y':
                rot_mat = RYGate(theta).to_matrix()
            else:  # case 'z'
                rot_mat = U1Gate(theta).to_matrix()

            expected = _compute_control_matrix(rot_mat, num_controls, ctrl_state=ctrl_state)
            with self.subTest(msg='control state = {}'.format(ctrl_state)):
                self.assertTrue(matrix_equal(simulated, expected))

    @combine(num_controls=[1, 2, 4], use_basis_gates=[True, False])
    def test_multi_controlled_y_rotation_matrix_basic_mode(self, num_controls, use_basis_gates):
        """Test the multi controlled Y rotation using the mode 'basic'.

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mcr.py
        """

        # get the number of required ancilla qubits
        if num_controls <= 2:
            num_ancillas = 0
        else:
            num_ancillas = num_controls - 2

        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)

        for ctrl_state in range(2 ** num_controls):
            bitstr = bin(ctrl_state)[2:].zfill(num_controls)[::-1]
            theta = 0.871236 * pi
            if num_ancillas > 0:
                q_ancillas = QuantumRegister(num_ancillas)
                qc = QuantumCircuit(q_controls, q_target, q_ancillas)
            else:
                qc = QuantumCircuit(q_controls, q_target)
                q_ancillas = None

            for idx, bit in enumerate(bitstr):
                if bit == '0':
                    qc.x(q_controls[idx])

            qc.mcry(theta, q_controls, q_target[0], q_ancillas, mode='basic',
                    use_basis_gates=use_basis_gates)

            for idx, bit in enumerate(bitstr):
                if bit == '0':
                    qc.x(q_controls[idx])

            rot_mat = RYGate(theta).to_matrix()

            backend = BasicAer.get_backend('unitary_simulator')
            simulated = execute(qc, backend).result().get_unitary(qc)
            if num_ancillas > 0:
                simulated = simulated[:2 ** (num_controls + 1), :2 ** (num_controls + 1)]

            expected = _compute_control_matrix(rot_mat, num_controls, ctrl_state=ctrl_state)

            with self.subTest(msg='control state = {}'.format(ctrl_state)):
                self.assertTrue(matrix_equal(simulated, expected))

    @data(0, 1, 2)
    def test_mcx_gates_yield_explicit_gates(self, num_ctrl_qubits):
        """Test the creating a MCX gate yields the explicit definition if we know it."""
        cls = MCXGate(num_ctrl_qubits).__class__
        explicit = {0: XGate, 1: CXGate, 2: CCXGate}
        self.assertEqual(cls, explicit[num_ctrl_qubits])

    @data(0, 3, 4, 5, 8)
    def test_mcx_gates(self, num_ctrl_qubits):
        """Test the mcx gates."""
        backend = BasicAer.get_backend('statevector_simulator')
        reference = np.zeros(2 ** (num_ctrl_qubits + 1))
        reference[-1] = 1

        for gate in [MCXGrayCode(num_ctrl_qubits),
                     MCXRecursive(num_ctrl_qubits),
                     MCXVChain(num_ctrl_qubits, False),
                     MCXVChain(num_ctrl_qubits, True),
                     ]:
            with self.subTest(gate=gate):
                circuit = QuantumCircuit(gate.num_qubits)
                if num_ctrl_qubits > 0:
                    circuit.x(list(range(num_ctrl_qubits)))
                circuit.append(gate, list(range(gate.num_qubits)), [])
                statevector = execute(circuit, backend).result().get_statevector()

                # account for ancillas
                if hasattr(gate, 'num_ancilla_qubits') and gate.num_ancilla_qubits > 0:
                    corrected = np.zeros(2 ** (num_ctrl_qubits + 1), dtype=complex)
                    for i, statevector_amplitude in enumerate(statevector):
                        i = int(bin(i)[2:].zfill(circuit.num_qubits)[gate.num_ancilla_qubits:], 2)
                        corrected[i] += statevector_amplitude
                    statevector = corrected

                np.testing.assert_array_almost_equal(statevector.real, reference)

    @data(1, 2, 3, 4)
    def test_inverse_x(self, num_ctrl_qubits):
        """Test inverting the controlled X gate."""
        cnx = XGate().control(num_ctrl_qubits)
        inv_cnx = cnx.inverse()
        result = Operator(cnx).compose(Operator(inv_cnx))
        np.testing.assert_array_almost_equal(result.data, np.identity(result.dim[0]))

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
        np.testing.assert_array_almost_equal(result.data, np.identity(result.dim[0]))

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
        base_gate = random_unitary(2 ** num_target).to_instruction()
        base_mat = base_gate.to_matrix()
        cgate = base_gate.control(num_ctrl_qubits)
        test_op = Operator(cgate)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op.data, ignore_phase=True))

    @data(1, 2, 3)
    def test_open_controlled_unitary_matrix(self, num_ctrl_qubits):
        """test open controlled unitary matrix"""
        # verify truth table
        num_target_qubits = 2
        num_qubits = num_ctrl_qubits + num_target_qubits
        target_op = Operator(XGate())
        for i in range(num_target_qubits - 1):
            target_op = target_op.tensor(XGate())

        for i in range(2 ** num_qubits):
            input_bitstring = bin(i)[2:].zfill(num_qubits)
            input_target = input_bitstring[0:num_target_qubits]
            input_ctrl = input_bitstring[-num_ctrl_qubits:]
            phi = Statevector.from_label(input_bitstring)
            cop = Operator(_compute_control_matrix(target_op.data,
                                                   num_ctrl_qubits,
                                                   ctrl_state=input_ctrl))
            for j in range(2 ** num_qubits):
                output_bitstring = bin(j)[2:].zfill(num_qubits)
                output_target = output_bitstring[0:num_target_qubits]
                output_ctrl = output_bitstring[-num_ctrl_qubits:]
                psi = Statevector.from_label(output_bitstring)
                cxout = np.dot(phi.data, psi.evolve(cop).data)
                if input_ctrl == output_ctrl:
                    # flip the target bits
                    cond_output = ''.join([str(int(not int(a))) for a in input_target])
                else:
                    cond_output = input_target
                if cxout == 1:
                    self.assertTrue(
                        (output_ctrl == input_ctrl) and
                        (output_target == cond_output))
                else:
                    self.assertTrue(
                        ((output_ctrl == input_ctrl) and
                         (output_target != cond_output)) or
                        output_ctrl != input_ctrl)

    @data(*ControlledGate.__subclasses__())
    def test_base_gate_setting(self, gate_class):
        """Test all gates in standard extensions which are of type ControlledGate
        and have a base gate setting.
        """
        num_free_params = len(_get_free_params(gate_class.__init__, ignore=['self']))
        free_params = [0.1 * i for i in range(num_free_params)]
        if gate_class in [MCU1Gate]:
            free_params[1] = 3
        elif gate_class in [MCXGate]:
            free_params[0] = 3

        base_gate = gate_class(*free_params)
        cgate = base_gate.control()
        self.assertEqual(base_gate.base_gate, cgate.base_gate)

    @data(*[gate for name, gate in allGates.__dict__.items() if isinstance(gate, type)])
    def test_all_inverses(self, gate):
        """Test all gates in standard extensions except those that cannot be controlled
        or are being deprecated.
        """
        if not (issubclass(gate, ControlledGate) or issubclass(gate, allGates.IGate)):
            # only verify basic gates right now, as already controlled ones
            # will generate differing definitions
            try:
                numargs = len(_get_free_params(gate))
                args = [2] * numargs

                gate = gate(*args)
                self.assertEqual(gate.inverse().control(2), gate.control(2).inverse())
            except AttributeError:
                # skip gates that do not have a control attribute (e.g. barrier)
                pass

    @data(2, 3)
    def test_relative_phase_toffoli_gates(self, num_ctrl_qubits):
        """Test the relative phase Toffoli gates.

        This test compares the matrix representation of the relative phase gate classes
        (i.e. RCCXGate().to_matrix()), the matrix obtained from the unitary simulator,
        and the exact version of the gate as obtained through `_compute_control_matrix`.
        """
        # get target matrix (w/o relative phase)
        base_mat = XGate().to_matrix()
        target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)

        # build the matrix for the relative phase toffoli using the unitary simulator
        circuit = QuantumCircuit(num_ctrl_qubits + 1)
        if num_ctrl_qubits == 2:
            circuit.rccx(0, 1, 2)
        else:  # num_ctrl_qubits == 3:
            circuit.rcccx(0, 1, 2, 3)
        simulator = BasicAer.get_backend('unitary_simulator')
        simulated_mat = execute(circuit, simulator).result().get_unitary()

        # get the matrix representation from the class itself
        if num_ctrl_qubits == 2:
            repr_mat = RCCXGate().to_matrix()
        else:  # num_ctrl_qubits == 3:
            repr_mat = RC3XGate().to_matrix()

        # test up to phase
        # note, that all entries may have an individual phase! (as opposed to a global phase)
        self.assertTrue(matrix_equal(np.abs(simulated_mat), target_mat))

        # compare simulated matrix with the matrix representation provided by the class
        self.assertTrue(matrix_equal(simulated_mat, repr_mat))

    def test_open_controlled_gate(self):
        """
        Test controlled gates with control on '0'
        """
        base_gate = XGate()
        base_mat = base_gate.to_matrix()
        num_ctrl_qubits = 3

        ctrl_state = 5
        cgate = base_gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
        target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=ctrl_state)
        self.assertEqual(Operator(cgate), Operator(target_mat))

        ctrl_state = None
        cgate = base_gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
        target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=ctrl_state)
        self.assertEqual(Operator(cgate), Operator(target_mat))

        ctrl_state = 0
        cgate = base_gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
        target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=ctrl_state)
        self.assertEqual(Operator(cgate), Operator(target_mat))

        ctrl_state = 7
        cgate = base_gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
        target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=ctrl_state)
        self.assertEqual(Operator(cgate), Operator(target_mat))

        ctrl_state = '110'
        cgate = base_gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
        target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=ctrl_state)
        self.assertEqual(Operator(cgate), Operator(target_mat))

    def test_open_controlled_gate_raises(self):
        """
        Test controlled gates with open controls raises if ctrl_state isn't allowed.
        """
        base_gate = XGate()
        num_ctrl_qubits = 3
        with self.assertRaises(CircuitError):
            base_gate.control(num_ctrl_qubits, ctrl_state=-1)
        with self.assertRaises(CircuitError):
            base_gate.control(num_ctrl_qubits, ctrl_state=2 ** num_ctrl_qubits)
        with self.assertRaises(CircuitError):
            base_gate.control(num_ctrl_qubits, ctrl_state='201')


@ddt
class TestSingleControlledRotationGates(QiskitTestCase):
    """Test the controlled rotation gates controlled on one qubit."""
    import qiskit.extensions.standard.u1 as u1
    import qiskit.extensions.standard.rx as rx
    import qiskit.extensions.standard.ry as ry
    import qiskit.extensions.standard.rz as rz

    num_ctrl = 2
    num_target = 1

    theta = pi / 2
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

    @data((gu1, cgu1), (grx, cgrx), (gry, cgry), (grz, cgrz))
    @unpack
    def test_single_controlled_rotation_gates(self, gate, cgate):
        """Test the controlled rotation gates controlled on one qubit."""
        if gate.name == 'rz':
            iden = Operator.from_label('I')
            zgen = Operator.from_label('Z')
            op_mat = (np.cos(0.5 * self.theta) * iden - 1j * np.sin(0.5 * self.theta) * zgen).data
        else:
            op_mat = Operator(gate).data
        ref_mat = Operator(cgate).data
        cop_mat = _compute_control_matrix(op_mat, self.num_ctrl)
        self.assertTrue(matrix_equal(cop_mat, ref_mat, ignore_phase=True))
        cqc = QuantumCircuit(self.num_ctrl + self.num_target)
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

    def test_composite(self):
        """Test composite gate count."""
        qreg = QuantumRegister(self.num_ctrl + self.num_target)
        qc = QuantumCircuit(qreg, name='composite')
        qc.append(self.grx.control(self.num_ctrl), qreg)
        qc.append(self.gry.control(self.num_ctrl), qreg)
        qc.append(self.gry, qreg[0:self.gry.num_qubits])
        qc.append(self.grz.control(self.num_ctrl), qreg)

        dag = circuit_to_dag(qc)
        unroller = Unroller(['u3', 'cx'])
        uqc = dag_to_circuit(unroller.run(dag))
        self.log.info('%s gate count: %d', uqc.name, uqc.size())
        self.assertTrue(uqc.size() <= 93)  # this limit could be changed


@ddt
class TestControlledStandardGates(QiskitTestCase):
    """Tests for control standard gates."""

    @combine(num_ctrl_qubits=[1, 2, 3],
             gate_class=[cls for cls in allGates.__dict__.values() if isinstance(cls, type)])
    def test_controlled_standard_gates(self, num_ctrl_qubits, gate_class):
        """Test controlled versions of all standard gates."""
        theta = pi / 2
        ctrl_state_ones = 2 ** num_ctrl_qubits - 1
        ctrl_state_zeros = 0
        ctrl_state_mixed = ctrl_state_ones >> int(num_ctrl_qubits / 2)

        numargs = len(_get_free_params(gate_class))
        args = [theta] * numargs
        if gate_class in [MSGate, Barrier]:
            args[0] = 2
        elif gate_class in [MCU1Gate]:
            args[1] = 2
        elif issubclass(gate_class, MCXGate):
            args = [5]

        gate = gate_class(*args)
        for ctrl_state in {ctrl_state_ones, ctrl_state_zeros, ctrl_state_mixed}:
            with self.subTest(i='{0}, ctrl_state={1}'.format(gate_class.__name__,
                                                             ctrl_state)):
                if hasattr(gate, 'num_ancilla_qubits') and gate.num_ancilla_qubits > 0:
                    # skip matrices that include ancilla qubits
                    continue
                try:
                    cgate = gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
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
                target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits,
                                                     ctrl_state=ctrl_state)
                self.assertTrue(matrix_equal(Operator(cgate).data, target_mat, ignore_phase=True))


@ddt
class TestDeprecatedGates(QiskitTestCase):
    """Test controlled of deprecated gates."""

    import qiskit.extensions as ext

    import qiskit.extensions.standard.i as i
    import qiskit.extensions.standard.rx as rx
    import qiskit.extensions.standard.ry as ry
    import qiskit.extensions.standard.rz as rz
    import qiskit.extensions.standard.swap as swap
    import qiskit.extensions.standard.u1 as u1
    import qiskit.extensions.standard.u3 as u3
    import qiskit.extensions.standard.x as x
    import qiskit.extensions.standard.y as y
    import qiskit.extensions.standard.z as z

    import qiskit.extensions.quantum_initializer.diagonal as diagonal
    import qiskit.extensions.quantum_initializer.uc as uc
    import qiskit.extensions.quantum_initializer.uc_pauli_rot as uc_pauli_rot
    import qiskit.extensions.quantum_initializer.ucrx as ucrx
    import qiskit.extensions.quantum_initializer.ucry as ucry
    import qiskit.extensions.quantum_initializer.ucrz as ucrz

    @data((diagonal.DiagonalGate, diagonal.DiagGate, [[1, 1]]),
          (i.IGate, i.IdGate, []),
          (rx.CRXGate, rx.CrxGate, [0.1]),
          (ry.CRYGate, ry.CryGate, [0.1]),
          (rz.CRZGate, rz.CrzGate, [0.1]),
          (swap.CSwapGate, swap.FredkinGate, []),
          (u1.CU1Gate, u1.Cu1Gate, [0.1]),
          (u3.CU3Gate, u3.Cu3Gate, [0.1, 0.2, 0.3]),
          (uc.UCGate, uc.UCG, [[np.array([[1, 0], [0, 1]])]]),
          (uc_pauli_rot.UCPauliRotGate, uc_pauli_rot.UCRot, [[0.1], 'X']),
          (ucrx.UCRXGate, ucrx.UCX, [[0.1]]),
          (ucry.UCRYGate, ucry.UCY, [[0.1]]),
          (ucrz.UCRZGate, ucrz.UCZ, [[0.1]]),
          (x.CXGate, x.CnotGate, []),
          (x.CCXGate, x.ToffoliGate, []),
          (y.CYGate, y.CyGate, []),
          (z.CZGate, z.CzGate, []),
          (i.IGate, ext.IdGate, []),
          (rx.CRXGate, ext.CrxGate, [0.1]),
          (ry.CRYGate, ext.CryGate, [0.1]),
          (rz.CRZGate, ext.CrzGate, [0.1]),
          (swap.CSwapGate, ext.FredkinGate, []),
          (u1.CU1Gate, ext.Cu1Gate, [0.1]),
          (u3.CU3Gate, ext.Cu3Gate, [0.1, 0.2, 0.3]),
          (x.CXGate, ext.CnotGate, []),
          (x.CCXGate, ext.ToffoliGate, []),
          (y.CYGate, ext.CyGate, []),
          (z.CZGate, ext.CzGate, []))
    @unpack
    def test_deprecated_gates(self, new, old, params):
        """Test types of the deprecated gate classes."""
        # assert old gate class derives from new
        self.assertTrue(issubclass(old, new))

        # assert both are representatives of one another
        self.assertTrue(isinstance(new(*params), old))
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(isinstance(old(*params), new))


@ddt
class TestParameterCtrlState(QiskitTestCase):
    """Test gate equality with ctrl_state parameter."""

    @data((RXGate(0.5), CRXGate(0.5)),
          (RYGate(0.5), CRYGate(0.5)),
          (RZGate(0.5), CRZGate(0.5)),
          (XGate(), CXGate()),
          (YGate(), CYGate()),
          (ZGate(), CZGate()),
          (U1Gate(0.5), CU1Gate(0.5)),
          (SwapGate(), CSwapGate()),
          (HGate(), CHGate()),
          (U3Gate(0.1, 0.2, 0.3), CU3Gate(0.1, 0.2, 0.3)))
    @unpack
    def test_ctrl_state_one(self, gate, controlled_gate):
        """Test controlled gates with ctrl_state
        See https://github.com/Qiskit/qiskit-terra/pull/4025
        """
        self.assertEqual(gate.control(1, ctrl_state='1'), controlled_gate)
        # TODO: once https://github.com/Qiskit/qiskit-terra/issues/3304 is fixed
        # TODO: move this test to
        # self.assertEqual(Operator(gate.control(1, ctrl_state='1')),
        # Operator(controlled_gate.to_matrix()))


@ddt
class TestControlledGateLabel(QiskitTestCase):
    """Tests for controlled gate labels."""
    gates_and_args = [(XGate, []),
                      (YGate, []),
                      (ZGate, []),
                      (HGate, []),
                      (CXGate, []),
                      (CCXGate, []),
                      (C3XGate, []),
                      (C4XGate, []),
                      (MCXGate, [5]),
                      (U1Gate, [0.1]),
                      (CYGate, []),
                      (CZGate, []),
                      (CU1Gate, [0.1]),
                      (SwapGate, []),
                      (CCXGate, []),
                      (RZGate, [0.1]),
                      (RXGate, [0.1]),
                      (RYGate, [0.1]),
                      (CRYGate, [0.1]),
                      (CRXGate, [0.1]),
                      (CSwapGate, []),
                      (U3Gate, [0.1, 0.2, 0.3]),
                      (CHGate, []),
                      (CRZGate, [0.1]),
                      (CU3Gate, [0.1, 0.2, 0.3]),
                      (MSGate, [5, 0.1]),
                      (RCCXGate, []),
                      (RC3XGate, []),
                      (MCU1Gate, [0.1, 1]),
                      (MCXGate, [5])
                      ]

    @data(*gates_and_args)
    @unpack
    def test_control_label(self, gate, args):
        """Test gate(label=...).control(label=...)"""
        cgate = gate(*args, label='a gate').control(label='a controlled gate')
        self.assertEqual(cgate.label, 'a controlled gate')
        self.assertEqual(cgate.base_gate.label, 'a gate')

    @data(*gates_and_args)
    @unpack
    def test_control_label_1(self, gate, args):
        """Test gate(label=...).control(1, label=...)"""
        cgate = gate(*args, label='a gate').control(1, label='a controlled gate')
        self.assertEqual(cgate.label, 'a controlled gate')
        self.assertEqual(cgate.base_gate.label, 'a gate')


if __name__ == '__main__':
    unittest.main()
