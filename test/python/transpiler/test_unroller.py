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


"""Test the Unroller pass"""

from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.simulator import Snapshot
from qiskit.transpiler.passes import Unroller
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.circuit import Parameter, Qubit, Clbit
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, CU1Gate, CU3Gate
from qiskit.transpiler.target import Target


class TestUnroller(QiskitTestCase):
    """Tests the Unroller pass."""

    def test_basic_unroll(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(["u2"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u2")

    def test_basic_unroll_target(self):
        """Test decompose a single H into U2 from target."""
        qc = QuantumCircuit(1)
        qc.h(0)
        target = Target(num_qubits=1)
        phi = Parameter("phi")
        lam = Parameter("lam")
        target.add_instruction(U2Gate(phi, lam))
        dag = circuit_to_dag(qc)
        pass_ = Unroller(target=target)
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u2")

    def test_unroll_toffoli(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(["h", "t", "tdg", "cx"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_unroll_1q_chain_conditional(self):
        """Test unroll chain of 1-qubit gates interrupted by conditional."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.tdg(qr)
        circuit.z(qr)
        circuit.t(qr)
        circuit.ry(0.5, qr)
        circuit.rz(0.3, qr)
        circuit.rx(0.1, qr)
        circuit.measure(qr, cr)
        circuit.x(qr).c_if(cr, 1)
        circuit.y(qr).c_if(cr, 1)
        circuit.z(qr).c_if(cr, 1)
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(["u1", "u2", "u3"])
        unrolled_dag = pass_.run(dag)

        # Pick up -1 * 0.3 / 2 global phase for one RZ -> U1.
        ref_circuit = QuantumCircuit(qr, cr, global_phase=-0.3 / 2)
        ref_circuit.append(U2Gate(0, pi), [qr[0]])
        ref_circuit.append(U1Gate(-pi / 4), [qr[0]])
        ref_circuit.append(U1Gate(pi), [qr[0]])
        ref_circuit.append(U1Gate(pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0.5, 0, 0), [qr[0]])
        ref_circuit.append(U1Gate(0.3), [qr[0]])
        ref_circuit.append(U3Gate(0.1, -pi / 2, pi / 2), [qr[0]])
        ref_circuit.measure(qr[0], cr[0])
        ref_circuit.append(U3Gate(pi, 0, pi), [qr[0]]).c_if(cr, 1)
        ref_circuit.append(U3Gate(pi, pi / 2, pi / 2), [qr[0]]).c_if(cr, 1)
        ref_circuit.append(U1Gate(pi), [qr[0]]).c_if(cr, 1)
        ref_dag = circuit_to_dag(ref_circuit)

        self.assertEqual(unrolled_dag, ref_dag)

    def test_unroll_no_basis(self):
        """Test when a given gate has no decompositions."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(basis=[])

        with self.assertRaises(QiskitError):
            pass_.run(dag)

    def test_simple_unroll_parameterized_without_expressions(self):
        """Verify unrolling parameterized gates without expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.rz(theta, qr[0])
        dag = circuit_to_dag(qc)

        unrolled_dag = Unroller(["u1", "u3", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-theta / 2)
        expected.append(U1Gate(theta), [qr[0]])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_simple_unroll_parameterized_with_expressions(self):
        """Verify unrolling parameterized gates with expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        qc.rz(sum_, qr[0])
        dag = circuit_to_dag(qc)

        unrolled_dag = Unroller(["u1", "u3", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-sum_ / 2)
        expected.append(U1Gate(sum_), [qr[0]])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_definition_unroll_parameterized(self):
        """Verify that unrolling complex gates with parameters does not raise."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.append(CU1Gate(theta), [qr[1], qr[0]])
        qc.append(CU1Gate(theta * theta), [qr[0], qr[1]])
        dag = circuit_to_dag(qc)

        out_dag = Unroller(["u1", "cx"]).run(dag)

        self.assertEqual(out_dag.count_ops(), {"u1": 6, "cx": 4})

    def test_unrolling_parameterized_composite_gates(self):
        """Verify unrolling circuits with parameterized composite gates."""
        qr1 = QuantumRegister(2)
        subqc = QuantumCircuit(qr1)

        theta = Parameter("theta")

        subqc.rz(theta, qr1[0])
        subqc.cx(qr1[0], qr1[1])
        subqc.rz(theta, qr1[1])

        # Expanding across register with shared parameter
        qr2 = QuantumRegister(4)
        qc = QuantumCircuit(qr2)

        qc.append(subqc.to_instruction(), [qr2[0], qr2[1]])
        qc.append(subqc.to_instruction(), [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        out_dag = Unroller(["u1", "u3", "cx"]).run(dag)

        # Pick up -1 * theta / 2 global phase four twice (once for each RZ -> P
        # in each of the two sub_instr instructions).
        expected = QuantumCircuit(qr2, global_phase=-1 * 4 * theta / 2.0)
        expected.append(U1Gate(theta), [qr2[0]])
        expected.cx(qr2[0], qr2[1])
        expected.append(U1Gate(theta), [qr2[1]])

        expected.append(U1Gate(theta), [qr2[2]])
        expected.cx(qr2[2], qr2[3])
        expected.append(U1Gate(theta), [qr2[3]])
        self.assertEqual(circuit_to_dag(expected), out_dag)

        # Expanding across register with shared parameter
        qc = QuantumCircuit(qr2)

        phi = Parameter("phi")
        gamma = Parameter("gamma")

        qc.append(subqc.to_instruction({theta: phi}), [qr2[0], qr2[1]])
        qc.append(subqc.to_instruction({theta: gamma}), [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        out_dag = Unroller(["u1", "u3", "cx"]).run(dag)

        expected = QuantumCircuit(qr2, global_phase=-1 * (2 * phi + 2 * gamma) / 2.0)
        expected.append(U1Gate(phi), [qr2[0]])
        expected.cx(qr2[0], qr2[1])
        expected.append(U1Gate(phi), [qr2[1]])

        expected.append(U1Gate(gamma), [qr2[2]])
        expected.cx(qr2[2], qr2[3])
        expected.append(U1Gate(gamma), [qr2[3]])

        self.assertEqual(circuit_to_dag(expected), out_dag)

    def test_unrolling_preserves_qregs_order(self):
        """Test unrolling a gate preseveres it's definition registers order"""
        qr = QuantumRegister(2, "qr1")
        qc = QuantumCircuit(qr)
        qc.cx(1, 0)
        gate = qc.to_gate()

        qr2 = QuantumRegister(2, "qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.append(gate, qr2)

        dag = circuit_to_dag(qc2)
        out_dag = Unroller(["cx"]).run(dag)

        expected = QuantumCircuit(qr2)
        expected.cx(1, 0)

        self.assertEqual(circuit_to_dag(expected), out_dag)

    def test_unrolling_nested_gates_preserves_qregs_order(self):
        """Test unrolling a nested gate preseveres it's definition registers order."""
        qr = QuantumRegister(2, "qr1")
        qc = QuantumCircuit(qr)
        qc.cx(1, 0)
        gate_level_1 = qc.to_gate()

        qr2 = QuantumRegister(2, "qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.append(gate_level_1, [1, 0])
        qc2.cp(pi, 1, 0)
        gate_level_2 = qc2.to_gate()

        qr3 = QuantumRegister(2, "qr3")
        qc3 = QuantumCircuit(qr3)
        qc3.append(gate_level_2, [1, 0])
        qc3.cu(pi, pi, pi, 0, 1, 0)
        gate_level_3 = qc3.to_gate()

        qr4 = QuantumRegister(2, "qr4")
        qc4 = QuantumCircuit(qr4)
        qc4.append(gate_level_3, [0, 1])

        dag = circuit_to_dag(qc4)
        out_dag = Unroller(["cx", "cp", "cu"]).run(dag)

        expected = QuantumCircuit(qr4)
        expected.cx(1, 0)
        expected.cp(pi, 0, 1)
        expected.cu(pi, pi, pi, 0, 1, 0)

        self.assertEqual(circuit_to_dag(expected), out_dag)

    def test_unrolling_global_phase_1q(self):
        """Test unrolling a circuit with global phase in a composite gate."""
        circ = QuantumCircuit(1, global_phase=pi / 2)
        circ.x(0)
        circ.h(0)
        v = circ.to_gate()

        qc = QuantumCircuit(1)
        qc.append(v, [0])

        dag = circuit_to_dag(qc)
        out_dag = Unroller(["cx", "x", "h"]).run(dag)
        qcd = dag_to_circuit(out_dag)

        self.assertEqual(Operator(qc), Operator(qcd))

    def test_unrolling_global_phase_nested_gates(self):
        """Test unrolling a nested gate preseveres global phase."""
        qc = QuantumCircuit(1, global_phase=pi)
        qc.x(0)
        gate = qc.to_gate()

        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        gate = qc.to_gate()

        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out_dag = Unroller(["x", "u"]).run(dag)
        qcd = dag_to_circuit(out_dag)

        self.assertEqual(Operator(qc), Operator(qcd))

    def test_if_simple(self):
        """Test a simple if statement unrolls correctly."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        qc = QuantumCircuit(qubits, clbits)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((clbits[0], 0)):
            qc.x(0)
        qc.h(0)
        qc.measure(0, 1)
        with qc.if_test((clbits[1], 0)):
            qc.h(1)
            qc.cx(1, 0)
        dag = circuit_to_dag(qc)
        unrolled_dag = Unroller(["u", "cx"]).run(dag)

        expected = QuantumCircuit(qubits, clbits)
        expected.u(pi / 2, 0, pi, 0)
        expected.measure(0, 0)
        with expected.if_test((clbits[0], 0)):
            expected.u(pi, 0, pi, 0)
        expected.u(pi / 2, 0, pi, 0)
        expected.measure(0, 1)
        with expected.if_test((clbits[1], 0)):
            expected.u(pi / 2, 0, pi, 1)
            expected.cx(1, 0)
        expected_dag = circuit_to_dag(expected)
        self.assertEqual(unrolled_dag, expected_dag)

    def test_if_else_simple(self):
        """Test a simple if-else statement unrolls correctly."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        qc = QuantumCircuit(qubits, clbits)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((clbits[0], 0)) as else_:
            qc.x(0)
        with else_:
            qc.z(0)
        qc.h(0)
        qc.measure(0, 1)
        with qc.if_test((clbits[1], 0)) as else_:
            qc.h(1)
            qc.cx(1, 0)
        with else_:
            qc.h(0)
            qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        unrolled_dag = Unroller(["u", "cx"]).run(dag)

        expected = QuantumCircuit(qubits, clbits)
        expected.u(pi / 2, 0, pi, 0)
        expected.measure(0, 0)
        with expected.if_test((clbits[0], 0)) as else_:
            expected.u(pi, 0, pi, 0)
        with else_:
            expected.u(0, 0, pi, 0)
        expected.u(pi / 2, 0, pi, 0)
        expected.measure(0, 1)
        with expected.if_test((clbits[1], 0)) as else_:
            expected.u(pi / 2, 0, pi, 1)
            expected.cx(1, 0)
        with else_:
            expected.u(pi / 2, 0, pi, 0)
            expected.cx(0, 1)
        expected_dag = circuit_to_dag(expected)
        self.assertEqual(unrolled_dag, expected_dag)

    def test_nested_control_flow(self):
        """Test unrolling nested control flow blocks."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        cr3 = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr1, cr2, cr3)
        with qc.for_loop(range(3)):
            with qc.while_loop((cr1, 0)):
                qc.x(0)
            with qc.while_loop((cr2, 0)):
                qc.y(0)
            with qc.while_loop((cr3, 0)):
                qc.z(0)
        dag = circuit_to_dag(qc)
        unrolled_dag = Unroller(["u", "cx"]).run(dag)

        expected = QuantumCircuit(qr, cr1, cr2, cr3)
        with expected.for_loop(range(3)):
            with expected.while_loop((cr1, 0)):
                expected.u(pi, 0, pi, 0)
            with expected.while_loop((cr2, 0)):
                expected.u(pi, pi / 2, pi / 2, 0)
            with expected.while_loop((cr3, 0)):
                expected.u(0, 0, pi, 0)
        expected_dag = circuit_to_dag(expected)
        self.assertEqual(unrolled_dag, expected_dag)

    def test_parameterized_angle(self):
        """Test unrolling with parameterized angle"""
        qc = QuantumCircuit(1)
        index = Parameter("index")
        with qc.for_loop((0, 0.5 * pi), index) as param:
            qc.rx(param, 0)
        dag = circuit_to_dag(qc)
        unrolled_dag = Unroller(["u", "cx"]).run(dag)

        expected = QuantumCircuit(1)
        with expected.for_loop((0, 0.5 * pi), index) as param:
            expected.u(param, -pi / 2, pi / 2, 0)
        expected_dag = circuit_to_dag(expected)
        self.assertEqual(unrolled_dag, expected_dag)


class TestUnrollAllInstructions(QiskitTestCase):
    """Test unrolling a circuit containing all standard instructions."""

    def setUp(self):
        super().setUp()
        qr = self.qr = QuantumRegister(3, "qr")
        cr = self.cr = ClassicalRegister(3, "cr")
        self.circuit = QuantumCircuit(qr, cr)
        self.ref_circuit = QuantumCircuit(qr, cr)
        self.pass_ = Unroller(basis=["u3", "cx", "id"])

    def compare_dags(self):
        """compare dags in class tests"""
        dag = circuit_to_dag(self.circuit)
        unrolled_dag = self.pass_.run(dag)
        ref_dag = circuit_to_dag(self.ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)

    def test_unroll_crx(self):
        """test unroll crx"""
        # qr_1: ─────■─────     qr_1: ─────────────────■─────────────────────■─────────────────────
        #       ┌────┴────┐  =        ┌─────────────┐┌─┴─┐┌───────────────┐┌─┴─┐┌─────────────────┐
        # qr_2: ┤ Rx(0.5) ├     qr_2: ┤ U3(0,0,π/2) ├┤ X ├┤ U3(-0.25,0,0) ├┤ X ├┤ U3(0.25,-π/2,0) ├
        #       └─────────┘           └─────────────┘└───┘└───────────────┘└───┘└─────────────────┘
        self.circuit.crx(0.5, 1, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 2), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(-0.25, 0, 0), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0.25, -pi / 2, 0), [2])
        self.compare_dags()

    def test_unroll_cry(self):
        """test unroll cry"""
        # qr_1: ─────■─────     qr_1: ──────────────────■─────────────────────■──
        #       ┌────┴────┐  =        ┌──────────────┐┌─┴─┐┌───────────────┐┌─┴─┐
        # qr_2: ┤ Ry(0.5) ├     qr_2: ┤ U3(0.25,0,0) ├┤ X ├┤ U3(-0.25,0,0) ├┤ X ├
        #       └─────────┘           └──────────────┘└───┘└───────────────┘└───┘
        self.circuit.cry(0.5, 1, 2)
        self.ref_circuit.append(U3Gate(0.25, 0, 0), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(-0.25, 0, 0), [2])
        self.ref_circuit.cx(1, 2)
        self.compare_dags()

    def test_unroll_ccx(self):
        """test unroll ccx"""

        # qr_0: ──■──     qr_0: ──────────────────────────────────────■──────────────────────»
        #         │                                                   │                      »
        # qr_1: ──■──  =  qr_1: ─────────────────■────────────────────┼───────────────────■──»
        #       ┌─┴─┐           ┌─────────────┐┌─┴─┐┌──────────────┐┌─┴─┐┌─────────────┐┌─┴─┐»
        # qr_2: ┤ X ├     qr_2: ┤ U3(π/2,0,π) ├┤ X ├┤ U3(0,0,-π/4) ├┤ X ├┤ U3(0,0,π/4) ├┤ X ├»
        #       └───┘           └─────────────┘└───┘└──────────────┘└───┘└─────────────┘└───┘»
        # «                                          ┌─────────────┐
        # «qr_0: ──────────────────■─────────■───────┤ U3(0,0,π/4) ├───■──
        # «      ┌─────────────┐   │       ┌─┴─┐     ├─────────────┴┐┌─┴─┐
        # «qr_1: ┤ U3(0,0,π/4) ├───┼───────┤ X ├─────┤ U3(0,0,-π/4) ├┤ X ├
        # «      ├─────────────┴┐┌─┴─┐┌────┴───┴────┐├─────────────┬┘└───┘
        # «qr_2: ┤ U3(0,0,-π/4) ├┤ X ├┤ U3(0,0,π/4) ├┤ U3(π/2,0,π) ├──────
        # «      └──────────────┘└───┘└─────────────┘└─────────────┘
        self.circuit.ccx(0, 1, 2)
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [2])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [1])
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [2])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.cx(0, 1)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [0])
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [1])
        self.ref_circuit.cx(0, 1)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [2])
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [2])
        self.compare_dags()

    def test_unroll_ch(self):
        """test unroll ch"""

        # qr_0: ──■──     qr_0: ───────────────────────────────────────────────■──────────────────»
        #       ┌─┴─┐  =        ┌─────────────┐┌─────────────┐┌─────────────┐┌─┴─┐┌──────────────┐»
        # qr_2: ┤ H ├     qr_2: ┤ U3(0,0,π/2) ├┤ U3(π/2,0,π) ├┤ U3(0,0,π/4) ├┤ X ├┤ U3(0,0,-π/4) ├»
        #       └───┘           └─────────────┘└─────────────┘└─────────────┘└───┘└──────────────┘»
        # «
        # «qr_0: ───────────────────────────────
        # «      ┌─────────────┐┌──────────────┐
        # «qr_2: ┤ U3(π/2,0,π) ├┤ U3(0,0,-π/2) ├
        # «      └─────────────┘└──────────────┘
        self.circuit.ch(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 2), [2])
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [2])
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [2])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [2])
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [2])
        self.ref_circuit.append(U3Gate(0, 0, -pi / 2), [2])
        self.compare_dags()

    def test_unroll_crz(self):
        """test unroll crz"""

        # qr_1: ─────■─────     qr_1: ──────────────────■─────────────────────■──
        #       ┌────┴────┐  =        ┌──────────────┐┌─┴─┐┌───────────────┐┌─┴─┐
        # qr_2: ┤ Rz(0.5) ├     qr_2: ┤ U3(0,0,0.25) ├┤ X ├┤ U3(0,0,-0.25) ├┤ X ├
        #       └─────────┘           └──────────────┘└───┘└───────────────┘└───┘
        self.circuit.crz(0.5, 1, 2)
        self.ref_circuit.append(U3Gate(0, 0, 0.25), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0, 0, -0.25), [2])
        self.ref_circuit.cx(1, 2)

    def test_unroll_cswap(self):
        """test unroll cswap"""
        #                     ┌───┐                                                             »
        # qr_0: ─X─     qr_0: ┤ X ├─────────────────■────────────────────────────────────────■──»
        #        │            └─┬─┘                 │                                        │  »
        # qr_1: ─■─  =  qr_1: ──┼───────────────────┼────────────────────■───────────────────┼──»
        #        │              │  ┌─────────────┐┌─┴─┐┌──────────────┐┌─┴─┐┌─────────────┐┌─┴─┐»
        # qr_2: ─X─     qr_2: ──■──┤ U3(π/2,0,π) ├┤ X ├┤ U3(0,0,-π/4) ├┤ X ├┤ U3(0,0,π/4) ├┤ X ├»
        #                          └─────────────┘└───┘└──────────────┘└───┘└─────────────┘└───┘»
        # «      ┌─────────────┐           ┌───┐     ┌──────────────┐┌───┐┌───┐
        # «qr_0: ┤ U3(0,0,π/4) ├───────────┤ X ├─────┤ U3(0,0,-π/4) ├┤ X ├┤ X ├
        # «      └─────────────┘           └─┬─┘     ├─────────────┬┘└─┬─┘└─┬─┘
        # «qr_1: ──────────────────■─────────■───────┤ U3(0,0,π/4) ├───■────┼──
        # «      ┌──────────────┐┌─┴─┐┌─────────────┐├─────────────┤        │
        # «qr_2: ┤ U3(0,0,-π/4) ├┤ X ├┤ U3(0,0,π/4) ├┤ U3(π/2,0,π) ├────────■──
        # «      └──────────────┘└───┘└─────────────┘└─────────────┘
        self.circuit.cswap(1, 0, 2)
        self.ref_circuit.cx(2, 0)
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [2])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [2])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [0])
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.cx(1, 0)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [0])
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [1])
        self.ref_circuit.cx(1, 0)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [2])
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [2])
        self.ref_circuit.cx(2, 0)
        self.compare_dags()

    def test_unroll_cu1(self):
        """test unroll cu1"""
        #                            ┌──────────────┐
        # qr_0: ─■────────     qr_0: ┤ U3(0,0,0.05) ├──■─────────────────────■──────────────────
        #        │U1(0.1)   =        └──────────────┘┌─┴─┐┌───────────────┐┌─┴─┐┌──────────────┐
        # qr_2: ─■────────     qr_2: ────────────────┤ X ├┤ U3(0,0,-0.05) ├┤ X ├┤ U3(0,0,0.05) ├
        #                                            └───┘└───────────────┘└───┘└──────────────┘
        self.circuit.append(CU1Gate(0.1), [0, 2])
        self.ref_circuit.append(U3Gate(0, 0, 0.05), [0])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, -0.05), [2])
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.append(U3Gate(0, 0, 0.05), [2])
        self.compare_dags()

    def test_unroll_cu3(self):
        """test unroll cu3"""
        #                                ┌──────────────┐
        # q_1: ────────■────────   q_1: ─┤ U3(0,0,0.05) ├──■────────────────────────■───────────────────
        #      ┌───────┴───────┐ =      ┌┴──────────────┤┌─┴─┐┌──────────────────┐┌─┴─┐┌───────────────┐
        # q_2: ┤ U3(0.2,0.1,0) ├   q_2: ┤ U3(0,0,-0.05) ├┤ X ├┤ U3(-0.1,0,-0.05) ├┤ X ├┤ U3(0.1,0.1,0) ├
        #      └───────────────┘        └───────────────┘└───┘└──────────────────┘└───┘└───────────────┘
        self.circuit.append(CU3Gate(0.2, 0.1, 0.0), [1, 2])
        self.ref_circuit.append(U3Gate(0, 0, 0.05), [1])
        self.ref_circuit.append(U3Gate(0, 0, -0.05), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(-0.1, 0, -0.05), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0.1, 0.1, 0), [2])
        self.compare_dags()

    def test_unroll_cx(self):
        """test unroll cx"""
        self.circuit.cx(1, 0)
        self.ref_circuit.cx(1, 0)
        self.compare_dags()

    def test_unroll_cy(self):
        """test unroll cy"""
        # qr_1: ──■──     qr_1: ──────────────────■─────────────────
        #       ┌─┴─┐  =        ┌──────────────┐┌─┴─┐┌─────────────┐
        # qr_2: ┤ Y ├     qr_2: ┤ U3(0,0,-π/2) ├┤ X ├┤ U3(0,0,π/2) ├
        #       └───┘           └──────────────┘└───┘└─────────────┘
        self.circuit.cy(1, 2)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 2), [2])
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 2), [2])
        self.compare_dags()

    def test_unroll_cz(self):
        """test unroll cz"""
        #                     ┌─────────────┐┌───┐┌─────────────┐
        # qr_0: ─■─     qr_0: ┤ U3(π/2,0,π) ├┤ X ├┤ U3(π/2,0,π) ├
        #        │   =        └─────────────┘└─┬─┘└─────────────┘
        # qr_2: ─■─     qr_2: ─────────────────■─────────────────
        self.circuit.cz(2, 0)
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [0])
        self.ref_circuit.cx(2, 0)
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [0])
        self.compare_dags()

    def test_unroll_h(self):
        """test unroll h"""
        self.circuit.h(1)
        self.ref_circuit.append(U3Gate(pi / 2, 0, pi), [1])
        self.compare_dags()

    def test_unroll_i(self):
        """test unroll i"""
        self.circuit.i(0)
        self.ref_circuit.i(0)
        self.compare_dags()

    def test_unroll_rx(self):
        """test unroll rx"""
        self.circuit.rx(0.1, 0)
        self.ref_circuit.append(U3Gate(0.1, -pi / 2, pi / 2), [0])
        self.compare_dags()

    def test_unroll_ry(self):
        """test unroll ry"""
        self.circuit.ry(0.2, 1)
        self.ref_circuit.append(U3Gate(0.2, 0, 0), [1])
        self.compare_dags()

    def test_unroll_rz(self):
        """test unroll rz"""
        self.circuit.rz(0.3, 2)
        self.ref_circuit.global_phase = -1 * 0.3 / 2
        self.ref_circuit.append(U3Gate(0, 0, 0.3), [2])
        self.compare_dags()

    def test_unroll_rzz(self):
        """test unroll rzz"""
        #                      global phase: 5.9832
        #                            ┌───┐┌─────────────┐┌───┐
        # qr_0: ─■────────     qr_0: ┤ X ├┤ U3(0,0,0.6) ├┤ X ├
        #        │ZZ(0.6)   =        └─┬─┘└─────────────┘└─┬─┘
        # qr_1: ─■────────     qr_1: ──■───────────────────■──
        self.circuit.rzz(0.6, 1, 0)
        self.ref_circuit.global_phase = -1 * 0.6 / 2
        self.ref_circuit.cx(1, 0)
        self.ref_circuit.append(U3Gate(0, 0, 0.6), [0])
        self.ref_circuit.cx(1, 0)
        self.compare_dags()

    def test_unroll_s(self):
        """test unroll s"""
        self.circuit.s(0)
        self.ref_circuit.append(U3Gate(0, 0, pi / 2), [0])
        self.compare_dags()

    def test_unroll_sdg(self):
        """test unroll sdg"""
        self.circuit.sdg(1)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 2), [1])
        self.compare_dags()

    def test_unroll_swap(self):
        """test unroll swap"""
        #                          ┌───┐
        # qr_1: ─X─     qr_1: ──■──┤ X ├──■──
        #        │   =        ┌─┴─┐└─┬─┘┌─┴─┐
        # qr_2: ─X─     qr_2: ┤ X ├──■──┤ X ├
        #                     └───┘     └───┘
        self.circuit.swap(1, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.cx(2, 1)
        self.ref_circuit.cx(1, 2)
        self.compare_dags()

    def test_unroll_t(self):
        """test unroll t"""
        self.circuit.t(2)
        self.ref_circuit.append(U3Gate(0, 0, pi / 4), [2])
        self.compare_dags()

    def test_unroll_tdg(self):
        """test unroll tdg"""
        self.circuit.tdg(0)
        self.ref_circuit.append(U3Gate(0, 0, -pi / 4), [0])
        self.compare_dags()

    def test_unroll_u1(self):
        """test unroll u1"""
        self.circuit.append(U1Gate(0.1), [1])
        self.ref_circuit.append(U3Gate(0, 0, 0.1), [1])
        self.compare_dags()

    def test_unroll_u2(self):
        """test unroll u2"""
        self.circuit.append(U2Gate(0.2, -0.1), [0])
        self.ref_circuit.append(U3Gate(pi / 2, 0.2, -0.1), [0])
        self.compare_dags()

    def test_unroll_u3(self):
        """test unroll u3"""
        self.circuit.append(U3Gate(0.3, 0.0, -0.1), [2])
        self.ref_circuit.append(U3Gate(0.3, 0.0, -0.1), [2])
        self.compare_dags()

    def test_unroll_x(self):
        """test unroll x"""
        self.circuit.x(2)
        self.ref_circuit.append(U3Gate(pi, 0, pi), [2])
        self.compare_dags()

    def test_unroll_y(self):
        """test unroll y"""
        self.circuit.y(1)
        self.ref_circuit.append(U3Gate(pi, pi / 2, pi / 2), [1])
        self.compare_dags()

    def test_unroll_z(self):
        """test unroll z"""
        self.circuit.z(0)
        self.ref_circuit.append(U3Gate(0, 0, pi), [0])
        self.compare_dags()

    def test_unroll_snapshot(self):
        """test unroll snapshot"""
        num_qubits = self.circuit.num_qubits
        instr = Snapshot("0", num_qubits=num_qubits)
        self.circuit.append(instr, range(num_qubits))
        self.ref_circuit.append(instr, range(num_qubits))
        self.compare_dags()

    def test_unroll_measure(self):
        """test unroll measure"""
        self.circuit.measure(self.qr, self.cr)
        self.ref_circuit.measure(self.qr, self.cr)
        self.compare_dags()
