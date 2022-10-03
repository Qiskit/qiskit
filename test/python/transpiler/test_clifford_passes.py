# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test transpiler passes and conversion methods that deal with Cliffords."""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes import OptimizeCliffords
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators import Clifford
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator, random_clifford
from qiskit.compiler.transpiler import transpile


class TestCliffordPasses(QiskitTestCase):
    """Tests to verify correctness of transpiler passes and
    conversion methods that deal with Cliffords."""

    def create_cliff1(self):
        """Creates a simple Clifford."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def create_cliff2(self):
        """Creates another simple Clifford."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def create_cliff3(self):
        """Creates a third Clifford which is the composition of the previous two."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def test_circuit_with_cliffords(self):
        """Test that Cliffords get stored natively on a QuantumCircuit,
        and that QuantumCircuit's decompose() replaces Clifford with gates."""

        # Create a circuit with 2 cliffords and four other gates
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(2, 0)
        qc.append(cliff1, [3, 0, 2])
        qc.swap(1, 3)
        qc.append(cliff2, [1, 2, 3])
        qc.h(3)

        # Check that there are indeed two Clifford objects in the circuit,
        # and that these are not gates.
        cliffords = [inst for inst, _, _ in qc.data if isinstance(inst, Clifford)]
        gates = [inst for inst, _, _ in qc.data if isinstance(inst, Gate)]
        self.assertEqual(len(cliffords), 2)
        self.assertEqual(len(gates), 4)

        # Check that calling QuantumCircuit's decompose(), no Clifford objects remain
        qc2 = qc.decompose()
        cliffords2 = [inst for inst, _, _ in qc2.data if isinstance(inst, Clifford)]
        self.assertEqual(len(cliffords2), 0)

    def test_can_construct_operator(self):
        """Test that we can construct an Operator from a circuit that
        contains a Clifford gate."""

        cliff = self.create_cliff1()
        qc = QuantumCircuit(4)
        qc.append(cliff, [3, 1, 2])

        # Create an operator from the decomposition of qc into gates
        op1 = Operator(qc.decompose())

        # Create an operator from qc directly
        op2 = Operator(qc)

        # Check that the two operators are equal
        self.assertTrue(op1.equiv(op2))

    def test_can_combine_cliffords(self):
        """Test that we can combine a pair of Cliffords over the same qubits
        using OptimizeCliffords transpiler pass."""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        cliff3 = self.create_cliff3()

        # Create a circuit with two consective cliffords
        qc1 = QuantumCircuit(4)
        qc1.append(cliff1, [3, 1, 2])
        qc1.append(cliff2, [3, 1, 2])
        self.assertEqual(qc1.count_ops()["clifford"], 2)

        # Run OptimizeCliffords pass, and check that only one Clifford remains
        qc1opt = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc1opt.count_ops()["clifford"], 1)

        # Create the expected circuit
        qc2 = QuantumCircuit(4)
        qc2.append(cliff3, [3, 1, 2])

        # Check that all possible operators are equal
        self.assertTrue(Operator(qc1).equiv(Operator(qc1.decompose())))
        self.assertTrue(Operator(qc1opt).equiv(Operator(qc1opt.decompose())))
        self.assertTrue(Operator(qc1).equiv(Operator(qc1opt)))
        self.assertTrue(Operator(qc2).equiv(Operator(qc2.decompose())))
        self.assertTrue(Operator(qc1opt).equiv(Operator(qc2)))

    def test_cannot_combine(self):
        """Test that currently we cannot combine a pair of Cliffords.
        The result will be changed after pass is updated"""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        qc1 = QuantumCircuit(4)
        qc1.append(cliff1, [3, 1, 2])
        qc1.append(cliff2, [3, 2, 1])
        qc1 = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc1.count_ops()["clifford"], 2)

    def test_circuit_to_dag_conversion_and_back(self):
        """Test that converting a circuit containing Clifford to a DAG
        and back preserves the Clifford.
        """
        # Create a Clifford
        cliff_circ = QuantumCircuit(3)
        cliff_circ.cx(0, 1)
        cliff_circ.h(0)
        cliff_circ.s(1)
        cliff_circ.swap(1, 2)
        cliff = Clifford(cliff_circ)

        # Add this Clifford to a Quantum Circuit, and check that it remains a Clifford
        circ0 = QuantumCircuit(4)
        circ0.append(cliff, [0, 1, 2])
        circ0_cliffords = [inst for inst, _, _ in circ0.data if isinstance(inst, Clifford)]
        circ0_gates = [inst for inst, _, _ in circ0.data if isinstance(inst, Gate)]
        self.assertEqual(len(circ0_cliffords), 1)
        self.assertEqual(len(circ0_gates), 0)

        # Check that converting circuit to DAG preserves Clifford.
        dag0 = circuit_to_dag(circ0)
        dag0_cliffords = [
            node
            for node in dag0.topological_nodes()
            if isinstance(node, DAGOpNode) and isinstance(node.op, Clifford)
        ]
        self.assertEqual(len(dag0_cliffords), 1)

        # Check that converted DAG to a circuit also preserves Clifford.
        circ1 = dag_to_circuit(dag0)
        circ1_cliffords = [inst for inst, _, _ in circ1.data if isinstance(inst, Clifford)]
        circ1_gates = [inst for inst, _, _ in circ1.data if isinstance(inst, Gate)]
        self.assertEqual(len(circ1_cliffords), 1)
        self.assertEqual(len(circ1_gates), 0)

        # However, test that running an unrolling pass on the DAG replaces Clifford
        # by gates.
        dag1 = HighLevelSynthesis().run(dag0)
        dag1_cliffords = [
            node
            for node in dag1.topological_nodes()
            if isinstance(node, DAGOpNode) and isinstance(node.op, Clifford)
        ]
        self.assertEqual(len(dag1_cliffords), 0)

    def test_optimize_cliffords(self):
        """Test OptimizeCliffords pass."""

        rng = np.random.default_rng(1234)
        for _ in range(20):
            # Create several random Cliffords
            cliffs = [random_clifford(3, rng) for _ in range(5)]

            # The first circuit contains these cliffords
            qc1 = QuantumCircuit(5)
            for cliff in cliffs:
                qc1.append(cliff, [4, 0, 2])
            self.assertEqual(qc1.count_ops()["clifford"], 5)

            # The second circuit is obtained by running the OptimizeCliffords pass.
            qc2 = PassManager(OptimizeCliffords()).run(qc1)
            self.assertEqual(qc2.count_ops()["clifford"], 1)

            # The third circuit contains the decompositions of Cliffods.
            qc3 = QuantumCircuit(5)
            for cliff in cliffs:
                qc3.append(cliff.to_circuit(), [4, 0, 2])
            self.assertNotIn("clifford", qc3.count_ops())

            # Check that qc1, qc2 and qc3 and their decompositions are all equivalent.
            self.assertTrue(Operator(qc1).equiv(Operator(qc1.decompose())))
            self.assertTrue(Operator(qc2).equiv(Operator(qc2.decompose())))
            self.assertTrue(Operator(qc3).equiv(Operator(qc3.decompose())))
            self.assertTrue(Operator(qc1).equiv(Operator(qc2)))
            self.assertTrue(Operator(qc1).equiv(Operator(qc3)))

    def test_if_else(self):
        """Test pass recurses into simple if-else."""
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        combined = cliff1.compose(cliff2)

        inner_test = QuantumCircuit(cliff1.num_qubits)
        inner_test.append(cliff1, inner_test.qubits)
        inner_test.append(cliff2, inner_test.qubits)

        inner_expected = QuantumCircuit(combined.num_qubits)
        inner_expected.append(combined, inner_expected.qubits)

        test = QuantumCircuit(cliff1.num_qubits, 1)
        test.measure(0, 0)
        test.if_else(
            (test.clbits[0], True), inner_test.copy(), inner_test.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(combined.num_qubits, 1)
        expected.measure(0, 0)
        expected.if_else(
            (expected.clbits[0], True),
            inner_expected,
            inner_expected,
            expected.qubits,
            expected.clbits,
        )

        self.assertEqual(OptimizeCliffords()(test), expected)

    def test_nested_control_flow(self):
        """Test pass recurses into nested control flow."""
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        combined = cliff1.compose(cliff2)

        inner_test = QuantumCircuit(cliff1.num_qubits)
        inner_test.append(cliff1, inner_test.qubits)
        inner_test.append(cliff2, inner_test.qubits)

        while_test = QuantumCircuit(cliff1.num_qubits, 1)
        while_test.for_loop((0,), None, inner_test.copy(), while_test.qubits, [])

        inner_expected = QuantumCircuit(combined.num_qubits)
        inner_expected.append(combined, inner_expected.qubits)

        while_expected = QuantumCircuit(combined.num_qubits, 1)
        while_expected.for_loop((0,), None, inner_expected, while_expected.qubits, [])

        test = QuantumCircuit(cliff1.num_qubits, 1)
        test.measure(0, 0)
        test.while_loop((test.clbits[0], True), while_test, test.qubits, test.clbits)

        expected = QuantumCircuit(combined.num_qubits, 1)
        expected.measure(0, 0)
        expected.while_loop(
            (expected.clbits[0], True), while_expected, expected.qubits, expected.clbits
        )

        self.assertEqual(OptimizeCliffords()(test), expected)

    def test_topological_ordering(self):
        """Test that Clifford optimization pass optimizes Cliffords across a gate
        on a different qubit."""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff1()

        qc1 = QuantumCircuit(5)
        qc1.append(cliff1, [0, 1, 2])
        qc1.h(4)
        qc1.append(cliff2, [0, 1, 2])

        # The second circuit is obtained by running the OptimizeCliffords pass.
        qc2 = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc2.count_ops()["clifford"], 1)

    def test_transpile_level_0(self):
        """Make sure that transpile with optimization_level=0 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=0)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_transpile_level_1(self):
        """Make sure that transpile with optimization_level=1 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=1)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_transpile_level_2(self):
        """Make sure that transpile with optimization_level=2 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=2)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_transpile_level_3(self):
        """Make sure that transpile with optimization_level=2 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=3)
        self.assertNotIn("clifford", qc2.count_ops())


if __name__ == "__main__":
    unittest.main()
