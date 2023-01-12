# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FlushDiagonal transpiler class"""

import numpy as np
import scipy.stats

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.library.standard_gates import (
    XGate,
    YGate,
    RYGate,
    RZGate,
    CXGate,
    CCXGate,
    HGate,
    ZGate,
    CHGate,
    CZGate,
)
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.extensions.unitary import UnitaryGate
from qiskit.quantum_info.operators import Operator
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.optimization.commute_diagonal import (
    CommuteDiagonal,
    _collect_circuit_between_nodes,
    Diagonality,
)
from qiskit import transpile


class TestCommuteDiagonal(QiskitTestCase):
    """Test class for CommuteDiagonal transpiler optimization class"""

    def setUp(self):
        super().setUp()
        self._pass = CommuteDiagonal()

    def test_gather_1q_between(self):
        """
        Test getting 1q gates between 2q groups. For the test circuit

             ┌──────────┐┌───┐┌─────────┐     ┌──────────┐
        q_0: ┤0         ├┤ X ├┤ Ry(0.2) ├─────┤1         ├
             │  Unitary │├───┤├─────────┤┌───┐│  Unitary │
        q_1: ┤1         ├┤ Y ├┤ Rz(0.1) ├┤ Y ├┤0         ├
             └──────────┘└───┘└─────────┘└───┘└──────────┘
        """

        qr = [Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat = scipy.stats.unitary_group.rvs(4, random_state=5627)
        start_node = dag.apply_operation_back(UnitaryGate(mat), qargs=qr)
        dag.apply_operation_back(XGate(), qargs=[qr[0]])
        dag.apply_operation_back(YGate(), qargs=[qr[1]])
        dag.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        dag.apply_operation_back(YGate(), qargs=[qr[1]])
        stop_node = dag.apply_operation_back(UnitaryGate(mat), qargs=qr[::-1])
        inter_nodes = _collect_circuit_between_nodes(
            dag, [qr[0], qr[1]], (start_node, start_node), (stop_node, stop_node)
        )
        inter_dag = DAGCircuit()
        inter_dag.add_qubits(qr)
        for node in inter_nodes:
            inter_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        expected = dag.copy_empty_like()
        expected.apply_operation_back(XGate(), qargs=[qr[0]])
        expected.apply_operation_back(YGate(), qargs=[qr[1]])
        expected.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        expected.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        expected.apply_operation_back(YGate(), qargs=[qr[1]])

        self.assertEqual(inter_dag, expected)

    def test_gather_1q_between_non_2q(self):
        """
        Test getting 1q gates between 2q groups where one of the
        2q groups ends or begins in 1q gates.

             ┌──────────┐┌───┐┌─────────┐     ┌──────────┐
        q_0: ┤0         ├┤ X ├┤ Ry(0.2) ├─────┤1         ├
             │  Unitary │├───┤├─────────┤┌───┐│  Unitary │
        q_1: ┤1         ├┤ Y ├┤ Rz(0.1) ├┤ Y ├┤0         ├
             └──────────┘└───┘└─────────┘└───┘└──────────┘

        """

        qr = [Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat = scipy.stats.unitary_group.rvs(4, random_state=594)
        dag.apply_operation_back(UnitaryGate(mat), qargs=qr)
        start_node0 = dag.apply_operation_back(XGate(), qargs=[qr[0]])
        start_node1 = dag.apply_operation_back(YGate(), qargs=[qr[1]])
        dag.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        stop_node1 = dag.apply_operation_back(YGate(), qargs=[qr[1]])
        stop_node0 = dag.apply_operation_back(UnitaryGate(mat), qargs=qr[::-1])
        inter_nodes = _collect_circuit_between_nodes(
            dag, [qr[0], qr[1]], (start_node0, start_node1), (stop_node0, stop_node1)
        )
        inter_dag = DAGCircuit()
        inter_dag.add_qubits(qr)
        for node in inter_nodes:
            inter_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        expected = dag.copy_empty_like()
        expected.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        expected.apply_operation_back(RZGate(0.1), qargs=[qr[1]])

        self.assertEqual(inter_dag, expected)

    def test_gather_2q_between(self):
        """
        Test getting 1q and 2q gates between 2q groups. For the test circuit

           ┌──────────┐┌───┐┌─────────┐                ┌──────────┐
        0: ┤0         ├┤ X ├┤ Ry(0.2) ├────────────────┤1         ├
           │  Unitary │├───┤└─────────┘┌─────────┐┌───┐│  Unitary │
        1: ┤1         ├┤ Y ├─────■─────┤ Rz(0.1) ├┤ Y ├┤0         ├
           └──────────┘└───┘   ┌─┴─┐   └─────────┘└───┘└──────────┘
        2: ────────────────────┤ X ├───────────────────────────────
                               └───┘
        """

        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat = scipy.stats.unitary_group.rvs(4, random_state=974)
        start_node = dag.apply_operation_back(UnitaryGate(mat), qargs=qr[0:2])
        dag.apply_operation_back(XGate(), qargs=[qr[0]])
        dag.apply_operation_back(YGate(), qargs=[qr[1]])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        dag.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        dag.apply_operation_back(YGate(), qargs=[qr[1]])
        stop_node = dag.apply_operation_back(UnitaryGate(mat), qargs=qr[1::-1])
        inter_nodes = _collect_circuit_between_nodes(
            dag, [qr[0], qr[1]], (start_node, start_node), (stop_node, stop_node)
        )
        inter_dag = DAGCircuit()
        inter_dag.add_qubits(qr)
        for node in inter_nodes:
            inter_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        expected = dag.copy_empty_like()
        expected.apply_operation_back(XGate(), qargs=[qr[0]])
        expected.apply_operation_back(YGate(), qargs=[qr[1]])
        expected.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        expected.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        expected.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        expected.apply_operation_back(YGate(), qargs=[qr[1]])
        self.assertEqual(inter_dag, expected)

    def test_gather_2q_between_interacting(self):
        """
        Test getting 1q and 2q gates between 2q groups when 2q gates
        interact within 2q of interest.

           ┌──────────┐┌───┐     ┌─────────┐     ┌──────────┐
        0: ┤0         ├┤ X ├──■──┤ Ry(0.2) ├─────┤1         ├
           │  Unitary │├───┤┌─┴─┐├─────────┤┌───┐│  Unitary │
        1: ┤1         ├┤ Y ├┤ X ├┤ Rz(0.1) ├┤ Y ├┤0         ├
           └──────────┘└───┘└───┘└─────────┘└───┘└──────────┘

        """

        qr = [Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat = scipy.stats.unitary_group.rvs(4, random_state=974)
        start_node = dag.apply_operation_back(UnitaryGate(mat), qargs=qr[0:2])
        dag.apply_operation_back(XGate(), qargs=[qr[0]])
        dag.apply_operation_back(YGate(), qargs=[qr[1]])
        dag.apply_operation_back(CXGate(), qargs=[qr[0], qr[1]])
        dag.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        dag.apply_operation_back(YGate(), qargs=[qr[1]])
        stop_node = dag.apply_operation_back(UnitaryGate(mat), qargs=qr[1::-1])
        inter_nodes = _collect_circuit_between_nodes(
            dag, [qr[0], qr[1]], (start_node, start_node), (stop_node, stop_node)
        )
        inter_dag = DAGCircuit()
        inter_dag.add_qubits(qr)
        for node in inter_nodes:
            inter_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        expected = dag.copy_empty_like()
        expected.apply_operation_back(XGate(), qargs=[qr[0]])
        expected.apply_operation_back(YGate(), qargs=[qr[1]])
        expected.apply_operation_back(CXGate(), qargs=[qr[0], qr[1]])
        expected.apply_operation_back(RYGate(0.2), qargs=[qr[0]])
        expected.apply_operation_back(RZGate(0.1), qargs=[qr[1]])
        expected.apply_operation_back(YGate(), qargs=[qr[1]])
        self.assertEqual(inter_dag, expected)

    def test_evaluate_diagonal_commutation_2q_true(self):
        """
        Test evaluating diagonal commutation on qubits 0 and 1 in the circuit,

              ┌───┐
        0: ───┤ Z ├───
           ┌──┴───┴──┐
        1: ┤ Rz(0.2) ├
           └─────────┘
        """
        qr = [Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        dag.apply_operation_back(ZGate(), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.2), qargs=[qr[1]])
        result, _ = self._pass.evaluate_diagonal_commutation(dag, [qr[0], qr[1]])
        self.assertTrue(result == Diagonality.TRUE)

    def test_evaluate_diagonal_commutation_2q_false(self):
        """
        Test evaluating diagonal commutation on qubits 0 and 1 in the circuit,

              ┌───┐
        0: ───┤ H ├───
           ┌──┴───┴──┐
        1: ┤ Rz(0.2) ├
           └─────────┘
        """
        qr = [Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.2), qargs=[qr[1]])
        result, _ = self._pass.evaluate_diagonal_commutation(dag, [qr[0], qr[1]])
        self.assertTrue(result == Diagonality.FALSE)

    def test_evaluate_diagonal_commutation_non_diagonal_ops(self):
        """
        Test evaluating diagonal commutation on qubits 0 and 1 in the circuit when
        individual gates are not diagonal
              ┌───┐   ┌───┐┌───┐
        0: ───┤ H ├───┤ X ├┤ H ├
           ┌──┴───┴──┐└───┘└───┘
        1: ┤ Rz(0.2) ├──────────
           └─────────┘
        """
        qr = [Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(XGate(), qargs=[qr[0]])
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(RZGate(0.2), qargs=[qr[1]])
        result, _ = self._pass.evaluate_diagonal_commutation(dag, [qr[0], qr[1]])
        self.assertTrue(result == Diagonality.TRUE)

    def test_evaluate_diagonal_commutation_3q_true(self):
        """
        Test evaluating diagonal commutation on qubits 0 and 1 in the circuit,

           ┌───┐┌───┐   ┌───┐
        0: ┤ H ├┤ X ├───┤ H ├───
           ├───┤└───┘┌──┴───┴──┐
        1: ┤ Z ├──■──┤ Rz(0.2) ├
           └───┘┌─┴─┐└─────────┘
        2: ─────┤ X ├───────────
                └───┘
        """
        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(XGate(), qargs=[qr[0]])
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(ZGate(), qargs=[qr[1]])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        dag.apply_operation_back(RZGate(0.2), qargs=[qr[1]])
        result, _ = self._pass.evaluate_diagonal_commutation(dag, [qr[0], qr[1]])
        self.assertTrue(result == Diagonality.TRUE)

    def test_evaluate_diagonal_commutation_3q_false(self):
        """
        Test evaluating diagonal commutation on qubits 0 and 2 in the circuit,

           ┌───┐┌───┐   ┌───┐
        0: ┤ H ├┤ X ├───┤ H ├───
           ├───┤└───┘┌──┴───┴──┐
        1: ┤ Z ├──■──┤ Rz(0.2) ├
           └───┘┌─┴─┐└─────────┘
        2: ─────┤ X ├───────────
                └───┘
        """
        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(XGate(), qargs=[qr[0]])
        dag.apply_operation_back(HGate(), qargs=[qr[0]])
        dag.apply_operation_back(ZGate(), qargs=[qr[1]])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        dag.apply_operation_back(RZGate(0.2), qargs=[qr[1]])
        result, _ = self._pass.evaluate_diagonal_commutation(
            dag, [qr[0], qr[2]], do_equiv_check=False
        )
        self.assertTrue(result == Diagonality.FALSE)

    def test_identify_simply_separated_2q_runs(self):
        """
        identify circuit sections where 2q runs on common qubits are separated by
        any other 2q runs to disjoint qubits.
                ┌───┐┌───┐        ┌─────────┐
        0: ──■──┤ H ├┤ Z ├──■───■─┤ Ry(0.1) ├
           ┌─┴─┐└─┬─┘└───┘  │   │ └─────────┘
        1: ┤ X ├──■─────────■───■────────────
           └───┘          ┌─┴─┐
        2: ───────────────┤ X ├──────────────
                          └───┘
        """
        pass_ = CommuteDiagonal()
        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        # 2q block #1
        dag.apply_operation_back(CXGate(), qargs=qr[0:2])
        dag.apply_operation_back(CHGate(), qargs=[qr[1], qr[0]])
        dag.apply_operation_back(ZGate(), qargs=[qr[0]])
        # seperator node
        dag.apply_operation_back(CCXGate(), qargs=[qr[0], qr[1], qr[2]])
        # 2q block #2
        dag.apply_operation_back(CZGate(), qargs=[qr[0], qr[1]])
        dag.apply_operation_back(RYGate(0.1), qargs=[qr[0]])

        blocks = dag.collect_2q_runs()
        candidate_blocks = pass_.get_candidate_2q_blocks(blocks, dag)

        expected_dag0 = DAGCircuit()
        expected_dag0.add_qubits(qr)
        expected_dag0.apply_operation_back(CXGate(), qargs=qr[0:2])
        expected_dag0.apply_operation_back(CHGate(), qargs=[qr[1], qr[0]])
        expected_dag0.apply_operation_back(ZGate(), qargs=[qr[0]])

        expected_dag1 = DAGCircuit()
        expected_dag1.add_qubits(qr)
        expected_dag1.apply_operation_back(CZGate(), qargs=[qr[0], qr[1]])
        expected_dag1.apply_operation_back(RYGate(0.1), qargs=[qr[0]])

        expected_blocks = {(qr[0], qr[1]): [expected_dag0.op_nodes(), expected_dag1.op_nodes()]}
        bit_map = {bit: ind for ind, bit in enumerate(qr)}
        for block0, block1 in zip(candidate_blocks[qr[0], qr[1]], expected_blocks[qr[0], qr[1]]):
            for node0, node1 in zip(block0, block1):
                self.assertTrue(
                    DAGNode.semantic_eq(
                        node0,
                        node1,
                        bit_indices1=bit_map,
                        bit_indices2=bit_map,
                    )
                )

    def test_two_group_optimize(self):
        """
        Test pass correctly can convert two 3 CNOT unituries to one 3 CNOT
        and one 2 CNOT unitary.

           ┌───────────┐     ┌───────────┐
        0: ┤0          ├─────┤0          ├
           │  Unitary0 │     │  Unitary1 │
        1: ┤1          ├──■──┤1          ├
           └───────────┘┌─┴─┐└───────────┘
        2: ─────────────┤ X ├─────────────
                        └───┘

        """
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        pass_ = CommuteDiagonal()
        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat0 = scipy.stats.unitary_group.rvs(4, random_state=61)
        mat1 = scipy.stats.unitary_group.rvs(4, random_state=94)
        dag.apply_operation_back(UnitaryGate(mat0), qargs=qr[0:2])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        dag.apply_operation_back(UnitaryGate(mat1), qargs=qr[0:2])
        circ = dag_to_circuit(dag)
        cdag = pass_.run(dag)
        ccirc = dag_to_circuit(cdag)
        circ_opt = transpile(circ, basis_gates=["cx", "u"], optimization_level=1)
        ccirc_opt = transpile(ccirc, basis_gates=["cx", "u"], optimization_level=1)
        self.assertEqual(circ_opt.count_ops()["cx"], 7)
        self.assertEqual(ccirc_opt.count_ops()["cx"], 6)
        self.assertEqual(Operator(ccirc_opt), Operator(circ))

    def test_five_group_optimize(self):
        """
        Test commutation chains together properly for a sequence of five 2q unitaries.
           ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
        0: ┤0          ├─────┤0          ├─────┤0          ├─────┤0          ├─────┤0          ├─────
           │  Unitary0 │     │  Unitary1 │     │  Unitary2 │     │  Unitary3 │     │  Unitary4 │
        1: ┤1          ├──■──┤1          ├──■──┤1          ├──■──┤1          ├──■──┤1          ├──■──
           └───────────┘┌─┴─┐└───────────┘┌─┴─┐└───────────┘┌─┴─┐└───────────┘┌─┴─┐└───────────┘┌─┴─┐
        2: ─────────────┤ X ├─────────────┤ X ├─────────────┤ X ├─────────────┤ X ├─────────────┤ X ├
                        └───┘             └───┘             └───┘             └───┘             └───┘
        """
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        pass_ = CommuteDiagonal()
        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat0 = scipy.stats.unitary_group.rvs(4, random_state=61)
        mat1 = scipy.stats.unitary_group.rvs(4, random_state=94)
        mat2 = scipy.stats.unitary_group.rvs(4, random_state=98)
        mat3 = scipy.stats.unitary_group.rvs(4, random_state=945)
        mat4 = scipy.stats.unitary_group.rvs(4, random_state=45)
        ug0 = UnitaryGate(mat0)
        ug0.name = "unitary0"
        dag.apply_operation_back(ug0, qargs=qr[0:2])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        ug1 = UnitaryGate(mat1)
        ug1.name = "unitary1"
        dag.apply_operation_back(ug1, qargs=qr[0:2])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        ug2 = UnitaryGate(mat2)
        ug2.name = "unitary2"
        dag.apply_operation_back(ug2, qargs=qr[0:2])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        ug3 = UnitaryGate(mat3)
        ug3.name = "unitary3"
        dag.apply_operation_back(ug3, qargs=qr[0:2])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        ug4 = UnitaryGate(mat4)
        ug4.name = "unitary4"
        dag.apply_operation_back(ug4, qargs=qr[0:2])
        dag.apply_operation_back(CXGate(), qargs=[qr[1], qr[2]])
        circ = dag_to_circuit(dag)
        circ_opt = transpile(circ, basis_gates=["cx", "u"], optimization_level=1)
        ccirc = pass_(circ_opt)
        ccirc_opt = transpile(ccirc, basis_gates=["cx", "u"], optimization_level=1)
        self.assertEqual(circ_opt.count_ops()["u"], 36)
        self.assertEqual(circ_opt.count_ops()["cx"], 20)
        self.assertEqual(ccirc_opt.count_ops()["u"], 33)
        self.assertEqual(ccirc_opt.count_ops()["cx"], 16)
        self.assertEqual(Operator(ccirc_opt), Operator(circ))

    def test_collective_commute(self):
        """
        Test commutation works when gates between 2q gates collectivly commute with
        the diagonal but don't individually commute.
           ┌──────────┐               ┌──────────┐
        0: ┤0         ├───────────────┤0         ├
           │  Unitary │┌───┐┌───┐┌───┐│  Unitary │
        1: ┤1         ├┤ H ├┤ X ├┤ H ├┤1         ├
           └──────────┘└─┬─┘└─┬─┘└─┬─┘└──────────┘
        2: ──────────────■────■────■──────────────

        """
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        pass_ = CommuteDiagonal()
        qr = [Qubit(), Qubit(), Qubit()]
        dag = DAGCircuit()
        dag.add_qubits(qr)
        mat0 = scipy.stats.unitary_group.rvs(4, random_state=61)
        mat1 = scipy.stats.unitary_group.rvs(4, random_state=94)
        dag.apply_operation_back(UnitaryGate(mat0), qargs=qr[0:2])
        dag.apply_operation_back(CHGate(), qargs=[qr[2], qr[1]])
        dag.apply_operation_back(CXGate(), qargs=[qr[2], qr[1]])
        dag.apply_operation_back(CHGate(), qargs=[qr[2], qr[1]])
        dag.apply_operation_back(UnitaryGate(mat1), qargs=qr[0:2])
        circ = dag_to_circuit(dag)
        cdag = pass_.run(dag)
        ccirc = dag_to_circuit(cdag)
        circ_opt = transpile(circ, basis_gates=["cx", "u"], optimization_level=1)
        ccirc_opt = transpile(ccirc, basis_gates=["cx", "u"], optimization_level=1)
        self.assertTrue(circ_opt.count_ops()["cx"], 9)
        self.assertTrue(ccirc_opt.count_ops()["cx"], 8)
        self.assertEqual(Operator(ccirc_opt), Operator(circ))

    def test_equiv_commute(self):
        """Test commutation works when gates between 2q gates collectivly
        commute with the diagonal but don't individually commute. In
        this case the 2q unitaries are pre-expanded to see that the
        grouping doesn't get confused.

           ┌──────────┐               ┌──────────┐
        0: ┤0         ├───────────────┤0         ├
           │  Unitary │┌───┐┌───┐┌───┐│  Unitary │
        1: ┤1         ├┤ H ├┤ X ├┤ H ├┤1         ├
           └──────────┘└─┬─┘└─┬─┘└─┬─┘└──────────┘
        2: ──────────────■────■────■──────────────

        We know the middle section of qubits on [1, 2] commutes with
        diagonal on [1].  However, when this circuit is expanded to
        {u, cx} and the pass collects 2q runs to identify commutation,
        the locals of the middle section get grouped with the first
        unitary since `collect_2q_runs` is greedy. The operation that
        remains on [1, 2] is no longer diagonal. Fortuately, in this
        case, the TwoQubitWeylDecomposition class can identify this as
        locally equivalent to a controlled gate so it can still be
        optimized.

        """
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        pass_ = CommuteDiagonal()
        qr = [Qubit(), Qubit(), Qubit()]
        circ = QuantumCircuit(qr)
        circ.cx(0, 1)
        circ.ry(0.2, 0)
        circ.rz(0.2, 1)
        circ.cx(0, 1)
        circ.ry(0.3, 0)
        circ.rz(0.3, 1)
        circ.cx(0, 1)
        circ.ch(2, 1)
        circ.cx(2, 1)
        circ.ch(2, 1)
        circ.cx(0, 1)
        circ.ry(0.2, 0)
        circ.rz(0.2, 1)
        circ.cx(0, 1)
        circ.ry(0.3, 0)
        circ.rz(0.3, 1)
        circ.cx(0, 1)
        circ_expand = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        ccirc = pass_(circ_expand)
        ccirc_opt = transpile(ccirc, basis_gates=["cx", "u"], optimization_level=0)
        self.assertEqual(circ_expand.count_ops()["cx"], 9)
        self.assertEqual(ccirc_opt.count_ops()["cx"], 6)
        self.assertEqual(Operator(ccirc_opt), Operator(circ))

    def test_equiv_commute_random(self):
        """Although this appears similar to the test above of a similar
        name, this cought a bug in the grouping of qubits due to the
        difference in single qubit pattern after basis translation.

           ┌──────────┐               ┌──────────┐
        0: ┤0         ├───────────────┤0         ├
           │  Unitary │┌───┐┌───┐┌───┐│  Unitary │
        1: ┤1         ├┤ H ├┤ X ├┤ H ├┤1         ├
           └──────────┘└─┬─┘└─┬─┘└─┬─┘└──────────┘
        2: ──────────────■────■────■──────────────

        """
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        pass_ = CommuteDiagonal()
        qr = [Qubit(), Qubit(), Qubit()]
        circ = QuantumCircuit(qr)
        mat0 = scipy.stats.unitary_group.rvs(4, random_state=254)
        mat1 = scipy.stats.unitary_group.rvs(4, random_state=981)
        circ.unitary(mat0, [0, 1])
        circ.ch(2, 1)
        circ.cx(2, 1)
        circ.ch(2, 1)
        circ.unitary(mat1, [0, 1])
        circ_expand = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        ccirc = pass_(circ_expand)
        ccirc_opt = transpile(ccirc, basis_gates=["cx", "u"], optimization_level=0)
        self.assertEqual(circ_expand.count_ops()["cx"], 9)
        self.assertEqual(ccirc_opt.count_ops()["cx"], 6)
        self.assertEqual(Operator(ccirc_opt), Operator(circ))
