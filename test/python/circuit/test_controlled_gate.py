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


"""Test Qiskit's controlled gate operation."""

import unittest
from test import combine
import numpy as np
from numpy import pi
from ddt import ddt, data, unpack

from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer, QiskitError
from qiskit.test import QiskitTestCase
from qiskit.circuit import ControlledGate, Parameter, Gate
from qiskit.circuit.singleton import SingletonControlledGate, _SingletonControlledGateOverrides
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.predicates import matrix_equal, is_unitary_matrix
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.states import Statevector
import qiskit.circuit.add_control as ac
from qiskit.transpiler.passes import Unroller
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import (
    CXGate,
    XGate,
    YGate,
    ZGate,
    U1Gate,
    CYGate,
    CZGate,
    CU1Gate,
    SwapGate,
    PhaseGate,
    CCXGate,
    HGate,
    RZGate,
    RXGate,
    CPhaseGate,
    RYGate,
    CRYGate,
    CRXGate,
    CSwapGate,
    UGate,
    U3Gate,
    CHGate,
    CRZGate,
    CU3Gate,
    CUGate,
    SXGate,
    CSXGate,
    MSGate,
    Barrier,
    RCCXGate,
    RC3XGate,
    MCU1Gate,
    MCXGate,
    MCXGrayCode,
    MCXRecursive,
    MCXVChain,
    C3XGate,
    C3SXGate,
    C4XGate,
    MCPhaseGate,
    GlobalPhaseGate,
    UnitaryGate,
)
from qiskit.circuit._utils import _compute_control_matrix
import qiskit.circuit.library.standard_gates as allGates
from qiskit.circuit.library.standard_gates.multi_control_rotation_gates import _mcsu2_real_diagonal

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

    def test_controlled_phase(self):
        """Test the creation of a controlled U1 gate."""
        theta = 0.5
        self.assertEqual(PhaseGate(theta).control(), CPhaseGate(theta))

    def test_double_controlled_phase(self):
        """Test the creation of a controlled phase gate."""
        theta = 0.5
        self.assertEqual(PhaseGate(theta).control(2), MCPhaseGate(theta, 2))

    def test_controlled_u1(self):
        """Test the creation of a controlled U1 gate."""
        theta = 0.5
        self.assertEqual(U1Gate(theta).control(), CU1Gate(theta))

        circ = QuantumCircuit(1)
        circ.append(U1Gate(theta), circ.qregs[0])
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["cx", "u", "p"])
        ctrl_circ_gate = dag_to_circuit(unroller.run(circuit_to_dag(circ))).control()
        ctrl_circ = QuantumCircuit(2)
        ctrl_circ.append(ctrl_circ_gate, ctrl_circ.qregs[0])
        ctrl_circ = ctrl_circ.decompose().decompose()
        self.assertEqual(ctrl_circ.size(), 1)

    def test_controlled_rz(self):
        """Test the creation of a controlled RZ gate."""
        theta = 0.5
        self.assertEqual(RZGate(theta).control(), CRZGate(theta))

    def test_control_parameters(self):
        """Test different ctrl_state formats for control function."""
        theta = 0.5

        self.assertEqual(
            CRYGate(theta).control(2, ctrl_state="01"), CRYGate(theta).control(2, ctrl_state=1)
        )
        self.assertEqual(
            CRYGate(theta).control(2, ctrl_state=None), CRYGate(theta).control(2, ctrl_state=3)
        )

        self.assertEqual(CCXGate().control(2, ctrl_state="01"), CCXGate().control(2, ctrl_state=1))
        self.assertEqual(CCXGate().control(2, ctrl_state=None), CCXGate().control(2, ctrl_state=3))

    def test_controlled_ry(self):
        """Test the creation of a controlled RY gate."""
        theta = 0.5
        self.assertEqual(RYGate(theta).control(), CRYGate(theta))

    def test_controlled_rx(self):
        """Test the creation of a controlled RX gate."""
        theta = 0.5
        self.assertEqual(RXGate(theta).control(), CRXGate(theta))

    def test_controlled_u(self):
        """Test the creation of a controlled U gate."""
        theta, phi, lamb = 0.1, 0.2, 0.3
        self.assertEqual(UGate(theta, phi, lamb).control(), CUGate(theta, phi, lamb, 0))

    def test_controlled_u3(self):
        """Test the creation of a controlled U3 gate."""
        theta, phi, lamb = 0.1, 0.2, 0.3
        self.assertEqual(U3Gate(theta, phi, lamb).control(), CU3Gate(theta, phi, lamb))

        circ = QuantumCircuit(1)
        circ.append(U3Gate(theta, phi, lamb), circ.qregs[0])
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["cx", "u", "p"])
        ctrl_circ_gate = dag_to_circuit(unroller.run(circuit_to_dag(circ))).control()
        ctrl_circ = QuantumCircuit(2)
        ctrl_circ.append(ctrl_circ_gate, ctrl_circ.qregs[0])
        ctrl_circ = ctrl_circ.decompose().decompose()

        self.assertEqual(ctrl_circ.size(), 1)

    def test_controlled_cx(self):
        """Test creation of controlled cx gate"""
        self.assertEqual(CXGate().control(), CCXGate())

    def test_controlled_swap(self):
        """Test creation of controlled swap gate"""
        self.assertEqual(SwapGate().control(), CSwapGate())

    def test_special_cases_equivalent_to_controlled_base_gate(self):
        """Test that ``ControlledGate`` subclasses for more efficient representations give
        equivalent matrices and definitions to the naive ``base_gate.control(n)``."""
        # Angles used here are not important, we just pick slightly strange values to ensure that
        # there are no coincidental equivalences.
        tests = [
            (CXGate(), 1),
            (CCXGate(), 2),
            (C3XGate(), 3),
            (C4XGate(), 4),
            (MCXGate(5), 5),
            (CYGate(), 1),
            (CZGate(), 1),
            (CPhaseGate(np.pi / 7), 1),
            (MCPhaseGate(np.pi / 7, 2), 2),
            (CSwapGate(), 1),
            (CSXGate(), 1),
            (C3SXGate(), 3),
            (CHGate(), 1),
            (CU1Gate(np.pi / 7), 1),
            (MCU1Gate(np.pi / 7, 2), 2),
            # `CUGate` takes an extra "global" phase parameter compared to `UGate`, and consequently
            # is only equal to `base_gate.control()` when this extra phase is 0.
            (CUGate(np.pi / 7, np.pi / 5, np.pi / 3, 0), 1),
            (CU3Gate(np.pi / 7, np.pi / 5, np.pi / 3), 1),
            (CRXGate(np.pi / 7), 1),
            (CRYGate(np.pi / 7), 1),
            (CRZGate(np.pi / 7), 1),
        ]
        for special_case_gate, n_controls in tests:
            with self.subTest(gate=special_case_gate.name):
                naive_operator = Operator(special_case_gate.base_gate.control(n_controls))
                # Ensure that both the array form (if the gate overrides `__array__`) and the
                # circuit-definition form are tested.
                self.assertTrue(Operator(special_case_gate).equiv(naive_operator))
                if not isinstance(special_case_gate, CXGate):
                    # CX is treated like a primitive within Terra, and doesn't have a definition.
                    self.assertTrue(Operator(special_case_gate.definition).equiv(naive_operator))

    def test_global_phase_control(self):
        """Test creation of a GlobalPhaseGate."""
        base = GlobalPhaseGate(np.pi / 7)
        expected_1q = PhaseGate(np.pi / 7)
        self.assertEqual(Operator(base.control()), Operator(expected_1q))

        expected_2q = PhaseGate(np.pi / 7).control()
        self.assertEqual(Operator(base.control(2)), Operator(expected_2q))

        expected_open = QuantumCircuit(1)
        expected_open.x(0)
        expected_open.p(np.pi / 7, 0)
        expected_open.x(0)
        self.assertEqual(Operator(base.control(ctrl_state=0)), Operator(expected_open))

    def test_circuit_append(self):
        """Test appending a controlled gate to a quantum circuit."""
        circ = QuantumCircuit(5)
        inst = CXGate()
        circ.append(inst.control(), qargs=[0, 2, 1])
        circ.append(inst.control(2), qargs=[0, 3, 1, 2])
        circ.append(inst.control().control(), qargs=[0, 3, 1, 2])  # should be same as above
        self.assertEqual(circ[1].operation, circ[2].operation)
        self.assertEqual(circ.depth(), 3)
        self.assertEqual(circ[0].operation.num_ctrl_qubits, 2)
        self.assertEqual(circ[1].operation.num_ctrl_qubits, 3)
        self.assertEqual(circ[2].operation.num_ctrl_qubits, 3)
        self.assertEqual(circ[0].operation.num_qubits, 3)
        self.assertEqual(circ[1].operation.num_qubits, 4)
        self.assertEqual(circ[2].operation.num_qubits, 4)
        for instr in circ:
            self.assertTrue(isinstance(instr.operation, ControlledGate))

    def test_swap_definition_specification(self):
        """Test the instantiation of a controlled swap gate with explicit definition."""
        swap = SwapGate()
        cswap = ControlledGate(
            "cswap", 3, [], num_ctrl_qubits=1, definition=swap.definition, base_gate=swap
        )
        self.assertEqual(swap.definition, cswap.definition)

    def test_multi_controlled_composite_gate(self):
        """Test a multi controlled composite gate."""
        num_ctrl = 3
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name="cgate")
        cgate.h(sub_q[0])
        cgate.crz(pi / 2, sub_q[0], sub_q[1])
        cgate.swap(sub_q[0], sub_q[1])
        cgate.u(0.1, 0.2, 0.3, sub_q[1])
        cgate.t(sub_q[0])
        num_target = cgate.width()
        gate = cgate.to_gate()
        cont_gate = gate.control(num_ctrl_qubits=num_ctrl)
        control = QuantumRegister(num_ctrl)
        target = QuantumRegister(num_target)
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:] + target[:])
        op_mat = Operator(cgate).data
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = Operator(qc).data
        self.assertTrue(matrix_equal(cop_mat, ref_mat))

    def test_single_controlled_composite_gate(self):
        """Test a singly controlled composite gate."""
        num_ctrl = 1
        # create composite gate
        sub_q = QuantumRegister(2)
        cgate = QuantumCircuit(sub_q, name="cgate")
        cgate.h(sub_q[0])
        cgate.cx(sub_q[0], sub_q[1])
        num_target = cgate.width()
        gate = cgate.to_gate()
        cont_gate = gate.control(num_ctrl_qubits=num_ctrl)
        control = QuantumRegister(num_ctrl, "control")
        target = QuantumRegister(num_target, "target")
        qc = QuantumCircuit(control, target)
        qc.append(cont_gate, control[:] + target[:])
        op_mat = Operator(cgate).data
        cop_mat = _compute_control_matrix(op_mat, num_ctrl)
        ref_mat = Operator(qc).data
        self.assertTrue(matrix_equal(cop_mat, ref_mat))

    def test_control_open_controlled_gate(self):
        """Test control(2) vs control.control where inner gate has open controls."""
        gate1pre = ZGate().control(1, ctrl_state=0)
        gate1 = gate1pre.control(1, ctrl_state=1)
        gate2 = ZGate().control(2, ctrl_state=1)
        expected = Operator(_compute_control_matrix(ZGate().to_matrix(), 2, ctrl_state=1))
        self.assertEqual(expected, Operator(gate1))
        self.assertEqual(expected, Operator(gate2))

    def test_multi_control_z(self):
        """Test a multi controlled Z gate."""
        qc = QuantumCircuit(1)
        qc.z(0)
        ctr_gate = qc.to_gate().control(2)

        ctr_circ = QuantumCircuit(3)
        ctr_circ.append(ctr_gate, range(3))

        ref_circ = QuantumCircuit(3)
        ref_circ.h(2)
        ref_circ.ccx(0, 1, 2)
        ref_circ.h(2)

        self.assertEqual(ctr_circ.decompose(), ref_circ)

    def test_multi_control_u3(self):
        """Test the matrix representation of the controlled and controlled-controlled U3 gate."""
        from qiskit.circuit.library.standard_gates import u3

        num_ctrl = 3
        # U3 gate params
        alpha, beta, gamma = 0.2, 0.3, 0.4
        u3gate = u3.U3Gate(alpha, beta, gamma)
        cu3gate = u3.CU3Gate(alpha, beta, gamma)

        # cnu3 gate
        cnu3 = u3gate.control(num_ctrl)
        width = cnu3.num_qubits
        qr = QuantumRegister(width)
        qcnu3 = QuantumCircuit(qr)
        qcnu3.append(cnu3, qr, [])

        # U3 gate
        qu3 = QuantumCircuit(1)
        qu3.append(u3gate, [0])

        # CU3 gate
        qcu3 = QuantumCircuit(2)
        qcu3.append(cu3gate, [0, 1])

        # c-cu3 gate
        width = 3
        qr = QuantumRegister(width)
        qc_cu3 = QuantumCircuit(qr)

        c_cu3 = cu3gate.control(1)
        qc_cu3.append(c_cu3, qr, [])

        # Circuit unitaries
        mat_cnu3 = Operator(qcnu3).data
        mat_u3 = Operator(qu3).data
        mat_cu3 = Operator(qcu3).data
        mat_c_cu3 = Operator(qc_cu3).data

        # Target Controlled-U3 unitary
        target_cnu3 = _compute_control_matrix(mat_u3, num_ctrl)
        target_cu3 = np.kron(mat_u3, np.diag([0, 1])) + np.kron(np.eye(2), np.diag([1, 0]))
        target_c_cu3 = np.kron(mat_cu3, np.diag([0, 1])) + np.kron(np.eye(4), np.diag([1, 0]))

        tests = [
            ("check unitary of u3.control against tensored unitary of u3", target_cu3, mat_cu3),
            (
                "check unitary of cu3.control against tensored unitary of cu3",
                target_c_cu3,
                mat_c_cu3,
            ),
            ("check unitary of cnu3 against tensored unitary of u3", target_cnu3, mat_cnu3),
        ]
        for itest in tests:
            info, target, decomp = itest[0], itest[1], itest[2]
            with self.subTest(i=info):
                self.assertTrue(matrix_equal(target, decomp, atol=1e-8, rtol=1e-5))

    def test_multi_control_u1(self):
        """Test the matrix representation of the controlled and controlled-controlled U1 gate."""
        from qiskit.circuit.library.standard_gates import u1

        num_ctrl = 3
        # U1 gate params
        theta = 0.2
        u1gate = u1.U1Gate(theta)
        cu1gate = u1.CU1Gate(theta)

        # cnu1 gate
        cnu1 = u1gate.control(num_ctrl)
        width = cnu1.num_qubits
        qr = QuantumRegister(width)
        qcnu1 = QuantumCircuit(qr)
        qcnu1.append(cnu1, qr, [])

        # U1 gate
        qu1 = QuantumCircuit(1)
        qu1.append(u1gate, [0])

        # CU1 gate
        qcu1 = QuantumCircuit(2)
        qcu1.append(cu1gate, [0, 1])

        # c-cu1 gate
        width = 3
        qr = QuantumRegister(width)
        qc_cu1 = QuantumCircuit(qr)
        c_cu1 = cu1gate.control(1)
        qc_cu1.append(c_cu1, qr, [])

        job = execute(
            [qcnu1, qu1, qcu1, qc_cu1],
            BasicAer.get_backend("unitary_simulator"),
            basis_gates=["u1", "u2", "u3", "id", "cx"],
        )
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

        tests = [
            ("check unitary of u1.control against tensored unitary of u1", target_cu1, mat_cu1),
            (
                "check unitary of cu1.control against tensored unitary of cu1",
                target_c_cu1,
                mat_c_cu1,
            ),
            ("check unitary of cnu1 against tensored unitary of u1", target_cnu1, mat_cnu1),
        ]
        for itest in tests:
            info, target, decomp = itest[0], itest[1], itest[2]
            with self.subTest(i=info):
                self.log.info(info)
                self.assertTrue(matrix_equal(target, decomp))

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
        for ctrl_state in range(2**num_controls):
            bitstr = bin(ctrl_state)[2:].zfill(num_controls)[::-1]
            lam = 0.3165354 * pi
            qc = QuantumCircuit(q_controls, q_target)
            for idx, bit in enumerate(bitstr):
                if bit == "0":
                    qc.x(q_controls[idx])

            qc.mcp(lam, q_controls, q_target[0])

            # for idx in subset:
            for idx, bit in enumerate(bitstr):
                if bit == "0":
                    qc.x(q_controls[idx])

            backend = BasicAer.get_backend("unitary_simulator")
            simulated = execute(qc, backend).result().get_unitary(qc)

            base = PhaseGate(lam).to_matrix()
            expected = _compute_control_matrix(base, num_controls, ctrl_state=ctrl_state)
            with self.subTest(msg=f"control state = {ctrl_state}"):
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
        qc.mcx(q_controls, q_target[0], q_ancillas, mode="basic")

        # execute the circuit and obtain statevector result
        backend = BasicAer.get_backend("unitary_simulator")
        simulated = execute(qc, backend).result().get_unitary(qc)

        # compare to expectation
        if num_ancillas > 0:
            simulated = simulated[: 2 ** (num_controls + 1), : 2 ** (num_controls + 1)]

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

        qc.mcx(q_controls, q_target[0], q_ancillas, mode="basic-dirty-ancilla")

        simulated = execute(qc, BasicAer.get_backend("unitary_simulator")).result().get_unitary(qc)
        if num_ancillas > 0:
            simulated = simulated[: 2 ** (num_controls + 1), : 2 ** (num_controls + 1)]

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

        qc.mcx(q_controls, q_target[0], q_ancillas, mode="advanced")

        simulated = execute(qc, BasicAer.get_backend("unitary_simulator")).result().get_unitary(qc)
        if num_ancillas > 0:
            simulated = simulated[: 2 ** (num_controls + 1), : 2 ** (num_controls + 1)]

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

        qc.mcx(q_controls, q_target[0], None, mode="noancilla")

        simulated = execute(qc, BasicAer.get_backend("unitary_simulator")).result().get_unitary(qc)

        base = XGate().to_matrix()
        expected = _compute_control_matrix(base, num_controls)
        self.assertTrue(matrix_equal(simulated, expected, atol=1e-8))

    def test_mcsu2_real_diagonal(self):
        """Test mcsu2_real_diagonal"""
        num_ctrls = 6
        theta = 0.3
        ry_matrix = RYGate(theta).to_matrix()
        qc = _mcsu2_real_diagonal(ry_matrix, num_ctrls)

        mcry_matrix = _compute_control_matrix(ry_matrix, 6)
        self.assertTrue(np.allclose(mcry_matrix, Operator(qc).to_matrix()))

    @combine(num_controls=[1, 2, 4], base_gate_name=["x", "y", "z"], use_basis_gates=[True, False])
    def test_multi_controlled_rotation_gate_matrices(
        self, num_controls, base_gate_name, use_basis_gates
    ):
        """Test the multi controlled rotation gates without ancillas.

        Based on the test moved here from Aqua:
        https://github.com/Qiskit/qiskit-aqua/blob/769ca8f/test/aqua/test_mcr.py
        """
        q_controls = QuantumRegister(num_controls)
        q_target = QuantumRegister(1)

        # iterate over all possible combinations of control qubits
        for ctrl_state in range(2**num_controls):
            bitstr = bin(ctrl_state)[2:].zfill(num_controls)[::-1]
            theta = 0.871236 * pi
            qc = QuantumCircuit(q_controls, q_target)
            for idx, bit in enumerate(bitstr):
                if bit == "0":
                    qc.x(q_controls[idx])

            # call mcrx/mcry/mcrz
            if base_gate_name == "y":
                qc.mcry(
                    theta,
                    q_controls,
                    q_target[0],
                    None,
                    mode="noancilla",
                    use_basis_gates=use_basis_gates,
                )
            else:  # case 'x' or 'z' only support the noancilla mode and do not have this keyword
                getattr(qc, "mcr" + base_gate_name)(
                    theta, q_controls, q_target[0], use_basis_gates=use_basis_gates
                )

            for idx, bit in enumerate(bitstr):
                if bit == "0":
                    qc.x(q_controls[idx])

            if use_basis_gates:
                with self.subTest(msg="check only basis gates used"):
                    gates_used = set(qc.count_ops().keys())
                    self.assertTrue(gates_used.issubset({"x", "u", "p", "cx"}))

            backend = BasicAer.get_backend("unitary_simulator")
            simulated = execute(qc, backend).result().get_unitary(qc)

            if base_gate_name == "x":
                rot_mat = RXGate(theta).to_matrix()
            elif base_gate_name == "y":
                rot_mat = RYGate(theta).to_matrix()
            else:  # case 'z'
                rot_mat = RZGate(theta).to_matrix()

            expected = _compute_control_matrix(rot_mat, num_controls, ctrl_state=ctrl_state)
            with self.subTest(msg=f"control state = {ctrl_state}"):
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

        for ctrl_state in range(2**num_controls):
            bitstr = bin(ctrl_state)[2:].zfill(num_controls)[::-1]
            theta = 0.871236 * pi
            if num_ancillas > 0:
                q_ancillas = QuantumRegister(num_ancillas)
                qc = QuantumCircuit(q_controls, q_target, q_ancillas)
            else:
                qc = QuantumCircuit(q_controls, q_target)
                q_ancillas = None

            for idx, bit in enumerate(bitstr):
                if bit == "0":
                    qc.x(q_controls[idx])

            qc.mcry(
                theta,
                q_controls,
                q_target[0],
                q_ancillas,
                mode="basic",
                use_basis_gates=use_basis_gates,
            )

            for idx, bit in enumerate(bitstr):
                if bit == "0":
                    qc.x(q_controls[idx])

            rot_mat = RYGate(theta).to_matrix()

            backend = BasicAer.get_backend("unitary_simulator")
            simulated = execute(qc, backend).result().get_unitary(qc)
            if num_ancillas > 0:
                simulated = simulated[: 2 ** (num_controls + 1), : 2 ** (num_controls + 1)]

            expected = _compute_control_matrix(rot_mat, num_controls, ctrl_state=ctrl_state)

            with self.subTest(msg=f"control state = {ctrl_state}"):
                self.assertTrue(matrix_equal(simulated, expected))

    def test_mcry_defaults_to_vchain(self):
        """Test mcry defaults to the v-chain mode if sufficient work qubits are provided."""
        circuit = QuantumCircuit(5)
        control_qubits = circuit.qubits[:3]
        target_qubit = circuit.qubits[3]
        additional_qubits = circuit.qubits[4:]
        circuit.mcry(0.2, control_qubits, target_qubit, additional_qubits)

        # If the v-chain mode is selected, all qubits are used. If the noancilla mode would be
        # selected, the bottom qubit would remain unused.
        dag = circuit_to_dag(circuit)
        self.assertEqual(len(list(dag.idle_wires())), 0)

    @data(1, 2)
    def test_mcx_gates_yield_explicit_gates(self, num_ctrl_qubits):
        """Test the creating a MCX gate yields the explicit definition if we know it."""
        cls = MCXGate(num_ctrl_qubits).__class__
        explicit = {1: CXGate, 2: CCXGate}
        self.assertEqual(cls, explicit[num_ctrl_qubits])

    @data(1, 2, 3, 4)
    def test_mcxgraycode_gates_yield_explicit_gates(self, num_ctrl_qubits):
        """Test creating an mcx gate calls MCXGrayCode and yeilds explicit definition."""
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.mcx(list(range(num_ctrl_qubits)), [num_ctrl_qubits])
        explicit = {1: CXGate, 2: CCXGate, 3: C3XGate, 4: C4XGate}
        self.assertEqual(type(qc[0].operation), explicit[num_ctrl_qubits])

    @data(3, 4, 5, 8)
    def test_mcx_gates(self, num_ctrl_qubits):
        """Test the mcx gates."""
        backend = BasicAer.get_backend("statevector_simulator")
        reference = np.zeros(2 ** (num_ctrl_qubits + 1))
        reference[-1] = 1

        for gate in [
            MCXGrayCode(num_ctrl_qubits),
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
                if hasattr(gate, "num_ancilla_qubits") and gate.num_ancilla_qubits > 0:
                    corrected = np.zeros(2 ** (num_ctrl_qubits + 1), dtype=complex)
                    for i, statevector_amplitude in enumerate(statevector):
                        i = int(bin(i)[2:].zfill(circuit.num_qubits)[gate.num_ancilla_qubits :], 2)
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
    def test_inverse_gate(self, num_ctrl_qubits):
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

    @data(1, 2, 3)
    def test_inverse_circuit(self, num_ctrl_qubits):
        """Test inverting a controlled gate based on a circuit definition."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rx(np.pi / 4, [0, 1, 2])
        cqc = qc.control(num_ctrl_qubits)
        cqc_inv = cqc.inverse()
        result = Operator(cqc).compose(Operator(cqc_inv))
        np.testing.assert_array_almost_equal(result.data, np.identity(result.dim[0]))

    @data(1, 2, 3, 4, 5)
    def test_controlled_unitary(self, num_ctrl_qubits):
        """Test the matrix data of an Operator, which is based on a controlled gate."""
        num_target = 1
        q_target = QuantumRegister(num_target)
        qc1 = QuantumCircuit(q_target)
        # for h-rx(pi/2)
        theta, phi, lamb = 1.57079632679490, 0.0, 4.71238898038469
        qc1.u(theta, phi, lamb, q_target[0])
        base_gate = qc1.to_gate()
        # get UnitaryGate version of circuit
        base_op = Operator(qc1)
        base_mat = base_op.data
        cgate = base_gate.control(num_ctrl_qubits)
        test_op = Operator(cgate)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(is_unitary_matrix(base_mat))
        self.assertTrue(matrix_equal(cop_mat, test_op.data))

    @data(1, 2, 3, 4, 5)
    def test_controlled_random_unitary(self, num_ctrl_qubits):
        """Test the matrix data of an Operator based on a random UnitaryGate."""
        num_target = 2
        base_gate = random_unitary(2**num_target).to_instruction()
        base_mat = base_gate.to_matrix()
        cgate = base_gate.control(num_ctrl_qubits)
        test_op = Operator(cgate)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op.data))

    @combine(num_ctrl_qubits=[1, 2, 3], ctrl_state=[0, None])
    def test_open_controlled_unitary_z(self, num_ctrl_qubits, ctrl_state):
        """Test that UnitaryGate with control returns params."""
        umat = np.array([[1, 0], [0, -1]])
        ugate = UnitaryGate(umat)
        cugate = ugate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
        ref_mat = _compute_control_matrix(umat, num_ctrl_qubits, ctrl_state=ctrl_state)
        self.assertEqual(Operator(cugate), Operator(ref_mat))

    def test_controlled_controlled_rz(self):
        """Test that UnitaryGate with control returns params."""
        qc = QuantumCircuit(1)
        qc.rz(0.2, 0)
        controlled = QuantumCircuit(2)
        controlled.compose(qc.control(), inplace=True)
        self.assertEqual(Operator(controlled), Operator(CRZGate(0.2)))
        self.assertEqual(Operator(controlled), Operator(RZGate(0.2).control()))

    def test_controlled_controlled_unitary(self):
        """Test that global phase in iso decomposition of unitary is handled."""
        umat = np.array([[1, 0], [0, -1]])
        ugate = UnitaryGate(umat)
        cugate = ugate.control()
        ccugate = cugate.control()
        ccugate2 = ugate.control(2)
        ref_mat = _compute_control_matrix(umat, 2)
        self.assertTrue(Operator(ccugate2).equiv(Operator(ref_mat)))
        self.assertTrue(Operator(ccugate).equiv(Operator(ccugate2)))

    @data(1, 2, 3)
    def test_open_controlled_unitary_matrix(self, num_ctrl_qubits):
        """test open controlled unitary matrix"""
        # verify truth table
        num_target_qubits = 2
        num_qubits = num_ctrl_qubits + num_target_qubits
        target_op = Operator(XGate())
        for i in range(num_target_qubits - 1):
            target_op = target_op.tensor(XGate())

        for i in range(2**num_qubits):
            input_bitstring = bin(i)[2:].zfill(num_qubits)
            input_target = input_bitstring[0:num_target_qubits]
            input_ctrl = input_bitstring[-num_ctrl_qubits:]
            phi = Statevector.from_label(input_bitstring)
            cop = Operator(
                _compute_control_matrix(target_op.data, num_ctrl_qubits, ctrl_state=input_ctrl)
            )
            for j in range(2**num_qubits):
                output_bitstring = bin(j)[2:].zfill(num_qubits)
                output_target = output_bitstring[0:num_target_qubits]
                output_ctrl = output_bitstring[-num_ctrl_qubits:]
                psi = Statevector.from_label(output_bitstring)
                cxout = np.dot(phi.data, psi.evolve(cop).data)
                if input_ctrl == output_ctrl:
                    # flip the target bits
                    cond_output = "".join([str(int(not int(a))) for a in input_target])
                else:
                    cond_output = input_target
                if cxout == 1:
                    self.assertTrue((output_ctrl == input_ctrl) and (output_target == cond_output))
                else:
                    self.assertTrue(
                        ((output_ctrl == input_ctrl) and (output_target != cond_output))
                        or output_ctrl != input_ctrl
                    )

    def test_open_control_cx_unrolling(self):
        """test unrolling of open control gates when gate is in basis"""
        qc = QuantumCircuit(2)
        qc.cx(0, 1, ctrl_state=0)
        dag = circuit_to_dag(qc)
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["u3", "cx"])
        uqc = dag_to_circuit(unroller.run(dag))

        ref_circuit = QuantumCircuit(2)
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [0])
        ref_circuit.cx(0, 1)
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [0])
        self.assertEqual(uqc, ref_circuit)

    def test_open_control_cy_unrolling(self):
        """test unrolling of open control gates when gate is in basis"""
        qc = QuantumCircuit(2)
        qc.cy(0, 1, ctrl_state=0)
        dag = circuit_to_dag(qc)
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["u3", "cy"])
        uqc = dag_to_circuit(unroller.run(dag))

        ref_circuit = QuantumCircuit(2)
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [0])
        ref_circuit.cy(0, 1)
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [0])
        self.assertEqual(uqc, ref_circuit)

    def test_open_control_ccx_unrolling(self):
        """test unrolling of open control gates when gate is in basis"""
        qreg = QuantumRegister(3)
        qc = QuantumCircuit(qreg)
        ccx = CCXGate(ctrl_state=0)
        qc.append(ccx, [0, 1, 2])
        dag = circuit_to_dag(qc)
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["x", "ccx"])
        unrolled_dag = unroller.run(dag)

        #       ┌───┐     ┌───┐
        # q0_0: ┤ X ├──■──┤ X ├
        #       ├───┤  │  ├───┤
        # q0_1: ┤ X ├──■──┤ X ├
        #       └───┘┌─┴─┐└───┘
        # q0_2: ─────┤ X ├─────
        #            └───┘
        ref_circuit = QuantumCircuit(qreg)
        ref_circuit.x(qreg[0])
        ref_circuit.x(qreg[1])
        ref_circuit.ccx(qreg[0], qreg[1], qreg[2])
        ref_circuit.x(qreg[0])
        ref_circuit.x(qreg[1])
        ref_dag = circuit_to_dag(ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)

    def test_ccx_ctrl_state_consistency(self):
        """Test the consistency of parameters ctrl_state in CCX
        See issue: https://github.com/Qiskit/qiskit-terra/issues/6465
        """
        qreg = QuantumRegister(3)
        qc = QuantumCircuit(qreg)
        qc.ccx(qreg[0], qreg[1], qreg[2], ctrl_state=0)

        ref_circuit = QuantumCircuit(qreg)
        ccx = CCXGate(ctrl_state=0)
        ref_circuit.append(ccx, [qreg[0], qreg[1], qreg[2]])
        self.assertEqual(qc, ref_circuit)

    def test_open_control_composite_unrolling(self):
        """test unrolling of open control gates when gate is in basis"""
        # create composite gate
        qreg = QuantumRegister(2)
        qcomp = QuantumCircuit(qreg, name="bell")
        qcomp.h(qreg[0])
        qcomp.cx(qreg[0], qreg[1])
        bell = qcomp.to_gate()
        # create controlled composite gate
        cqreg = QuantumRegister(3)
        qc = QuantumCircuit(cqreg)
        qc.append(bell.control(ctrl_state=0), qc.qregs[0][:])
        dag = circuit_to_dag(qc)
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["x", "u1", "cbell"])
        unrolled_dag = unroller.run(dag)
        # create reference circuit
        ref_circuit = QuantumCircuit(cqreg)
        ref_circuit.x(cqreg[0])
        ref_circuit.append(bell.control(), [cqreg[0], cqreg[1], cqreg[2]])
        ref_circuit.x(cqreg[0])
        ref_dag = circuit_to_dag(ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)

    @data(*ControlledGate.__subclasses__())
    def test_standard_base_gate_setting(self, gate_class):
        """Test all gates in standard extensions which are of type ControlledGate
        and have a base gate setting.
        """
        if gate_class in {SingletonControlledGate, _SingletonControlledGateOverrides}:
            self.skipTest("SingletonControlledGate isn't directly instantiated.")
        num_free_params = len(_get_free_params(gate_class.__init__, ignore=["self"]))
        free_params = [0.1 * i for i in range(num_free_params)]
        if gate_class in [MCU1Gate, MCPhaseGate]:
            free_params[1] = 3
        elif gate_class in [MCXGate]:
            free_params[0] = 3

        base_gate = gate_class(*free_params)
        cgate = base_gate.control()

        # the base gate of CU is U (3 params), the base gate of CCU is CU (4 params)
        if gate_class == CUGate:
            self.assertListEqual(cgate.base_gate.params[:3], base_gate.base_gate.params[:3])
        else:
            self.assertEqual(base_gate.base_gate, cgate.base_gate)

    @combine(
        gate=[cls for cls in allGates.__dict__.values() if isinstance(cls, type)],
        num_ctrl_qubits=[1, 2],
        ctrl_state=[None, 0, 1],
    )
    def test_all_inverses(self, gate, num_ctrl_qubits, ctrl_state):
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

                self.assertEqual(
                    gate.inverse().control(num_ctrl_qubits, ctrl_state=ctrl_state),
                    gate.control(num_ctrl_qubits, ctrl_state=ctrl_state).inverse(),
                )

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
        simulator = BasicAer.get_backend("unitary_simulator")
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

        ctrl_state = "110"
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
            base_gate.control(num_ctrl_qubits, ctrl_state=2**num_ctrl_qubits)
        with self.assertRaises(CircuitError):
            base_gate.control(num_ctrl_qubits, ctrl_state="201")

    def test_base_gate_params_reference(self):
        """
        Test all gates in standard extensions which are of type ControlledGate and have a base gate
        setting have params which reference the one in their base gate.
        """
        num_ctrl_qubits = 1
        for gate_class in ControlledGate.__subclasses__():
            with self.subTest(i=repr(gate_class)):
                if gate_class in {SingletonControlledGate, _SingletonControlledGateOverrides}:
                    self.skipTest("Singleton class isn't intended to be created directly.")
                num_free_params = len(_get_free_params(gate_class.__init__, ignore=["self"]))
                free_params = [0.1 * (i + 1) for i in range(num_free_params)]
                if gate_class in [MCU1Gate, MCPhaseGate]:
                    free_params[1] = 3
                elif gate_class in [MCXGate]:
                    free_params[0] = 3

                base_gate = gate_class(*free_params)
                if base_gate.params:
                    cgate = base_gate.control(num_ctrl_qubits)
                    self.assertIs(cgate.base_gate.params, cgate.params)

    def test_assign_parameters(self):
        """Test assigning parameters to quantum circuit with controlled gate."""
        qc = QuantumCircuit(2, name="assign")
        ptest = Parameter("p")
        gate = CRYGate(ptest)
        qc.append(gate, [0, 1])

        subs1, subs2 = {ptest: Parameter("a")}, {ptest: Parameter("b")}
        bound1 = qc.assign_parameters(subs1, inplace=False)
        bound2 = qc.assign_parameters(subs2, inplace=False)

        self.assertEqual(qc.parameters, {ptest})
        self.assertEqual(bound1.parameters, {subs1[ptest]})
        self.assertEqual(bound2.parameters, {subs2[ptest]})

    def test_assign_cugate(self):
        """Test assignment of CUGate, which breaks the `ControlledGate` requirements by not being
        equivalent to a direct control of its base gate."""

        parameters = [Parameter("t"), Parameter("p"), Parameter("l"), Parameter("g")]
        values = [0.1, 0.2, 0.3, 0.4]

        qc = QuantumCircuit(2)
        qc.cu(*parameters, 0, 1)
        assigned = qc.assign_parameters(dict(zip(parameters, values)), inplace=False)

        expected = QuantumCircuit(2)
        expected.cu(*values, 0, 1)

        self.assertEqual(assigned.data[0].operation.base_gate, expected.data[0].operation.base_gate)
        self.assertEqual(assigned, expected)

    def test_modify_cugate_params_slice(self):
        """Test that CUGate.params can be modified by a standard slice (without changing the number
        of elements) and changes propagate to the base gate.  This is only needed for as long as
        CUGate's `base_gate` is `UGate`, which has the "wrong" number of parameters."""
        cu = CUGate(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(cu.params, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(cu.base_gate.params, [0.1, 0.2, 0.3])

        cu.params[0:4] = [0.5, 0.4, 0.3, 0.2]
        self.assertEqual(cu.params, [0.5, 0.4, 0.3, 0.2])
        self.assertEqual(cu.base_gate.params, [0.5, 0.4, 0.3])

        cu.params[:] = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(cu.params, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(cu.base_gate.params, [0.1, 0.2, 0.3])

        cu.params[:3] = [0.5, 0.4, 0.3]
        self.assertEqual(cu.params, [0.5, 0.4, 0.3, 0.4])
        self.assertEqual(cu.base_gate.params, [0.5, 0.4, 0.3])

        # indices (3, 2, 1, 0), note that the assignment is in reverse.
        cu.params[-1::-1] = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(cu.params, [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(cu.base_gate.params, [0.4, 0.3, 0.2])

    def test_assign_nested_controlled_cu(self):
        """Test assignment of an arbitrary controlled parametrised gate that appears through the
        `Gate.control()` method on an already-controlled gate."""
        theta = Parameter("t")
        qc_c = QuantumCircuit(2)
        qc_c.crx(theta, 1, 0)
        custom_gate = qc_c.to_gate().control()
        qc = QuantumCircuit(3)
        qc.append(custom_gate, [0, 1, 2])
        assigned = qc.assign_parameters({theta: 0.5})

        # We're testing here that everything's been propagated through to the base gates; the `reps`
        # is just some high number to make sure we unwrap any controlled and custom gates.
        self.assertEqual(set(assigned.decompose(reps=3).parameters), set())

    @data(-1, 0, 1.4, "1", 4, 10)
    def test_improper_num_ctrl_qubits(self, num_ctrl_qubits):
        """
        Test improperly specified num_ctrl_qubits.
        """
        num_qubits = 4
        with self.assertRaises(CircuitError):
            ControlledGate(
                name="cgate", num_qubits=num_qubits, params=[], num_ctrl_qubits=num_ctrl_qubits
            )

    def test_improper_num_ctrl_qubits_base_gate(self):
        """Test that the allowed number of control qubits takes the base gate into account."""
        with self.assertRaises(CircuitError):
            ControlledGate(
                name="cx?", num_qubits=2, params=[], num_ctrl_qubits=2, base_gate=XGate()
            )
        self.assertIsInstance(
            ControlledGate(
                name="cx?", num_qubits=2, params=[], num_ctrl_qubits=1, base_gate=XGate()
            ),
            ControlledGate,
        )
        self.assertIsInstance(
            ControlledGate(
                name="p",
                num_qubits=1,
                params=[np.pi],
                num_ctrl_qubits=1,
                base_gate=Gate("gphase", 0, [np.pi]),
            ),
            ControlledGate,
        )

    def test_open_controlled_equality(self):
        """
        Test open controlled gates are equal if their base gates and control states are equal.
        """

        self.assertEqual(XGate().control(1), XGate().control(1))

        self.assertNotEqual(XGate().control(1), YGate().control(1))

        self.assertNotEqual(XGate().control(1), XGate().control(2))

        self.assertEqual(XGate().control(1, ctrl_state="0"), XGate().control(1, ctrl_state="0"))

        self.assertNotEqual(XGate().control(1, ctrl_state="0"), XGate().control(1, ctrl_state="1"))

    def test_cx_global_phase(self):
        """
        Test controlling CX with global phase
        """
        theta = pi / 2
        circ = QuantumCircuit(2, global_phase=theta)
        circ.cx(0, 1)
        cx = circ.to_gate()
        self.assertNotEqual(Operator(CXGate()), Operator(cx))

        ccx = cx.control(1)
        base_mat = Operator(cx).data
        target = _compute_control_matrix(base_mat, 1)
        self.assertEqual(Operator(ccx), Operator(target))

        expected = QuantumCircuit(*ccx.definition.qregs)
        expected.ccx(0, 1, 2)
        expected.p(theta, 0)
        self.assertEqual(ccx.definition, expected)

    @data(1, 2)
    def test_controlled_global_phase(self, num_ctrl_qubits):
        """
        Test controlled global phase on base gate.
        """
        theta = pi / 4
        circ = QuantumCircuit(2, global_phase=theta)
        base_gate = circ.to_gate()
        base_mat = Operator(base_gate).data
        target = _compute_control_matrix(base_mat, num_ctrl_qubits)
        cgate = base_gate.control(num_ctrl_qubits)
        ccirc = circ.control(num_ctrl_qubits)
        self.assertEqual(Operator(cgate), Operator(target))
        self.assertEqual(Operator(ccirc), Operator(target))

    @data(1, 2)
    def test_rz_composite_global_phase(self, num_ctrl_qubits):
        """
        Test controlling CX with global phase
        """
        theta = pi / 4
        circ = QuantumCircuit(2, global_phase=theta)
        circ.rz(0.1, 0)
        circ.rz(0.2, 1)
        ccirc = circ.control(num_ctrl_qubits)
        base_gate = circ.to_gate()
        cgate = base_gate.control(num_ctrl_qubits)
        base_mat = Operator(base_gate).data
        target = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertEqual(Operator(cgate), Operator(target))
        self.assertEqual(Operator(ccirc), Operator(target))

    @data(1, 2)
    def test_nested_global_phase(self, num_ctrl_qubits):
        """
        Test controlling a gate with nested global phase.
        """
        theta = pi / 4
        circ = QuantumCircuit(1, global_phase=theta)
        circ.z(0)
        v = circ.to_gate()

        qc = QuantumCircuit(1)
        qc.append(v, [0])
        ctrl_qc = qc.control(num_ctrl_qubits)

        base_mat = Operator(qc).data
        target = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertEqual(Operator(ctrl_qc), Operator(target))

    @data(1, 2)
    def test_control_zero_operand_gate(self, num_ctrl_qubits):
        """Test that a zero-operand gate (such as a make-shift global-phase gate) can be
        controlled."""
        gate = QuantumCircuit(global_phase=np.pi).to_gate()
        controlled = gate.control(num_ctrl_qubits)
        self.assertIsInstance(controlled, ControlledGate)
        self.assertEqual(controlled.num_ctrl_qubits, num_ctrl_qubits)
        self.assertEqual(controlled.num_qubits, num_ctrl_qubits)
        target = np.eye(2**num_ctrl_qubits, dtype=np.complex128)
        target.flat[-1] = -1
        self.assertEqual(Operator(controlled), Operator(target))


@ddt
class TestOpenControlledToMatrix(QiskitTestCase):
    """Test controlled_gates implementing to_matrix work with ctrl_state"""

    @combine(gate_class=ControlledGate.__subclasses__(), ctrl_state=[0, None])
    def test_open_controlled_to_matrix(self, gate_class, ctrl_state):
        """Test open controlled to_matrix."""
        if gate_class in {SingletonControlledGate, _SingletonControlledGateOverrides}:
            self.skipTest("SingletonGateClass isn't intended for direct initalization")
        num_free_params = len(_get_free_params(gate_class.__init__, ignore=["self"]))
        free_params = [0.1 * i for i in range(1, num_free_params + 1)]
        if gate_class in [MCU1Gate, MCPhaseGate]:
            free_params[1] = 3
        elif gate_class in [MCXGate]:
            free_params[0] = 3
        cgate = gate_class(*free_params)
        cgate.ctrl_state = ctrl_state

        base_mat = Operator(cgate.base_gate).data
        if gate_class == CUGate:  # account for global phase
            base_mat = np.array(base_mat) * np.exp(1j * cgate.params[3])

        target = _compute_control_matrix(base_mat, cgate.num_ctrl_qubits, ctrl_state=ctrl_state)
        try:
            actual = cgate.to_matrix()
        except CircuitError as cerr:
            self.skipTest(str(cerr))
        self.assertTrue(np.allclose(actual, target))


@ddt
class TestSingleControlledRotationGates(QiskitTestCase):
    """Test the controlled rotation gates controlled on one qubit."""

    from qiskit.circuit.library.standard_gates import u1, rx, ry, rz

    num_ctrl = 2
    num_target = 1

    theta = pi / 2
    gu1 = u1.U1Gate(theta)
    grx = rx.RXGate(theta)
    gry = ry.RYGate(theta)
    grz = rz.RZGate(theta)

    ugu1 = ac._unroll_gate(gu1, ["p", "u", "cx"])
    ugrx = ac._unroll_gate(grx, ["p", "u", "cx"])
    ugry = ac._unroll_gate(gry, ["p", "u", "cx"])
    ugrz = ac._unroll_gate(grz, ["p", "u", "cx"])
    ugrz.params = grz.params

    cgu1 = ugu1.control(num_ctrl)
    cgrx = ugrx.control(num_ctrl)
    cgry = ugry.control(num_ctrl)
    cgrz = ugrz.control(num_ctrl)

    @data((gu1, cgu1), (grx, cgrx), (gry, cgry), (grz, cgrz))
    @unpack
    def test_single_controlled_rotation_gates(self, gate, cgate):
        """Test the controlled rotation gates controlled on one qubit."""
        if gate.name == "rz":
            iden = Operator.from_label("I")
            zgen = Operator.from_label("Z")
            op_mat = (np.cos(0.5 * self.theta) * iden - 1j * np.sin(0.5 * self.theta) * zgen).data
        else:
            op_mat = Operator(gate).data
        ref_mat = Operator(cgate).data
        cop_mat = _compute_control_matrix(op_mat, self.num_ctrl)
        self.assertTrue(matrix_equal(cop_mat, ref_mat))
        cqc = QuantumCircuit(self.num_ctrl + self.num_target)
        cqc.append(cgate, cqc.qregs[0])
        dag = circuit_to_dag(cqc)
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["u", "cx"])
        uqc = dag_to_circuit(unroller.run(dag))
        self.log.info("%s gate count: %d", cgate.name, uqc.size())
        self.log.info("\n%s", str(uqc))
        # these limits could be changed
        if gate.name == "ry":
            self.assertLessEqual(uqc.size(), 32, f"\n{uqc}")
        elif gate.name == "rz":
            self.assertLessEqual(uqc.size(), 43, f"\n{uqc}")
        else:
            self.assertLessEqual(uqc.size(), 20, f"\n{uqc}")

    def test_composite(self):
        """Test composite gate count."""
        qreg = QuantumRegister(self.num_ctrl + self.num_target)
        qc = QuantumCircuit(qreg, name="composite")
        qc.append(self.grx.control(self.num_ctrl), qreg)
        qc.append(self.gry.control(self.num_ctrl), qreg)
        qc.append(self.gry, qreg[0 : self.gry.num_qubits])
        qc.append(self.grz.control(self.num_ctrl), qreg)

        dag = circuit_to_dag(qc)
        with self.assertWarns(DeprecationWarning):
            unroller = Unroller(["u", "cx"])
        uqc = dag_to_circuit(unroller.run(dag))
        self.log.info("%s gate count: %d", uqc.name, uqc.size())
        self.assertLessEqual(uqc.size(), 96, f"\n{uqc}")  # this limit could be changed


@ddt
class TestControlledStandardGates(QiskitTestCase):
    """Tests for control standard gates."""

    @combine(
        num_ctrl_qubits=[1, 2, 3],
        gate_class=[cls for cls in allGates.__dict__.values() if isinstance(cls, type)],
    )
    def test_controlled_standard_gates(self, num_ctrl_qubits, gate_class):
        """Test controlled versions of all standard gates."""
        theta = pi / 2
        ctrl_state_ones = 2**num_ctrl_qubits - 1
        ctrl_state_zeros = 0
        ctrl_state_mixed = ctrl_state_ones >> int(num_ctrl_qubits / 2)

        numargs = len(_get_free_params(gate_class))
        args = [theta] * numargs
        if gate_class in [MSGate, Barrier]:
            args[0] = 2
        elif gate_class in [MCU1Gate, MCPhaseGate]:
            args[1] = 2
        elif issubclass(gate_class, MCXGate):
            args = [5]
        gate = gate_class(*args)

        for ctrl_state in (ctrl_state_ones, ctrl_state_zeros, ctrl_state_mixed):
            with self.subTest(i=f"{gate_class.__name__}, ctrl_state={ctrl_state}"):
                if hasattr(gate, "num_ancilla_qubits") and gate.num_ancilla_qubits > 0:
                    # skip matrices that include ancilla qubits
                    continue
                try:
                    cgate = gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)
                except (AttributeError, QiskitError):
                    # 'object has no attribute "control"'
                    # skipping Id and Barrier
                    continue
                base_mat = Operator(gate).data

                target_mat = _compute_control_matrix(
                    base_mat, num_ctrl_qubits, ctrl_state=ctrl_state
                )
                self.assertEqual(Operator(cgate), Operator(target_mat))


@ddt
class TestParameterCtrlState(QiskitTestCase):
    """Test gate equality with ctrl_state parameter."""

    @data(
        (RXGate(0.5), CRXGate(0.5)),
        (RYGate(0.5), CRYGate(0.5)),
        (RZGate(0.5), CRZGate(0.5)),
        (XGate(), CXGate()),
        (YGate(), CYGate()),
        (ZGate(), CZGate()),
        (U1Gate(0.5), CU1Gate(0.5)),
        (PhaseGate(0.5), CPhaseGate(0.5)),
        (SwapGate(), CSwapGate()),
        (HGate(), CHGate()),
        (U3Gate(0.1, 0.2, 0.3), CU3Gate(0.1, 0.2, 0.3)),
        (UGate(0.1, 0.2, 0.3), CUGate(0.1, 0.2, 0.3, 0)),
    )
    @unpack
    def test_ctrl_state_one(self, gate, controlled_gate):
        """Test controlled gates with ctrl_state
        See https://github.com/Qiskit/qiskit-terra/pull/4025
        """
        self.assertEqual(
            Operator(gate.control(1, ctrl_state="1")), Operator(controlled_gate.to_matrix())
        )


@ddt
class TestControlledGateLabel(QiskitTestCase):
    """Tests for controlled gate labels."""

    gates_and_args = [
        (XGate, []),
        (YGate, []),
        (ZGate, []),
        (HGate, []),
        (CXGate, []),
        (CCXGate, []),
        (C3XGate, []),
        (C3SXGate, []),
        (C4XGate, []),
        (MCXGate, [5]),
        (PhaseGate, [0.1]),
        (U1Gate, [0.1]),
        (CYGate, []),
        (CZGate, []),
        (CPhaseGate, [0.1]),
        (CU1Gate, [0.1]),
        (SwapGate, []),
        (SXGate, []),
        (CSXGate, []),
        (CCXGate, []),
        (RZGate, [0.1]),
        (RXGate, [0.1]),
        (RYGate, [0.1]),
        (CRYGate, [0.1]),
        (CRXGate, [0.1]),
        (CSwapGate, []),
        (UGate, [0.1, 0.2, 0.3]),
        (U3Gate, [0.1, 0.2, 0.3]),
        (CHGate, []),
        (CRZGate, [0.1]),
        (CUGate, [0.1, 0.2, 0.3, 0.4]),
        (CU3Gate, [0.1, 0.2, 0.3]),
        (MSGate, [5, 0.1]),
        (RCCXGate, []),
        (RC3XGate, []),
        (MCU1Gate, [0.1, 1]),
        (MCXGate, [5]),
    ]

    @data(*gates_and_args)
    @unpack
    def test_control_label(self, gate, args):
        """Test gate(label=...).control(label=...)"""
        cgate = gate(*args, label="a gate").control(label="a controlled gate")
        self.assertEqual(cgate.label, "a controlled gate")
        self.assertEqual(cgate.base_gate.label, "a gate")

    @data(*gates_and_args)
    @unpack
    def test_control_label_1(self, gate, args):
        """Test gate(label=...).control(1, label=...)"""
        cgate = gate(*args, label="a gate").control(1, label="a controlled gate")
        self.assertEqual(cgate.label, "a controlled gate")
        self.assertEqual(cgate.base_gate.label, "a gate")


if __name__ == "__main__":
    unittest.main()
