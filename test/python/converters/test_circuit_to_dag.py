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

"""Tests for the converters."""

import unittest

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit, Clbit
from qiskit.circuit.library import HGate, Measure
from qiskit.circuit.classical import expr
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCircuitToDag(QiskitTestCase):
    """Test Circuit to DAG."""

    def test_circuit_and_dag(self):
        """Check convert to dag and back"""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit_in = QuantumCircuit(qr, cr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[1])
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.x(qr[0]).c_if(cr, 0x3)
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.measure(qr[2], cr[2])
        dag = circuit_to_dag(circuit_in)
        circuit_out = dag_to_circuit(dag)
        self.assertEqual(circuit_out, circuit_in)

    def test_calibrations(self):
        """Test that calibrations are properly copied over."""
        circuit_in = QuantumCircuit(1)
        circuit_in.add_calibration("h", [0], None)
        self.assertEqual(len(circuit_in.calibrations), 1)

        dag = circuit_to_dag(circuit_in)
        self.assertEqual(len(dag.calibrations), 1)

        circuit_out = dag_to_circuit(dag)
        self.assertEqual(len(circuit_out.calibrations), 1)

    def test_wires_from_expr_nodes_condition(self):
        """Test that the classical wires implied by an `Expr` node in a control-flow op's
        `condition` are correctly transferred."""
        # The control-flow builder interface always includes any classical wires in the blocks of
        # the operation, so we test by using manually constructed blocks that don't do that.  It's
        # not required by the `QuantumCircuit` model (just like `c_if` instructions don't expand
        # their `cargs`).
        inner = QuantumCircuit(1)
        inner.x(0)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        clbit = Clbit()
        outer = QuantumCircuit(QuantumRegister(1), cr1, cr2, [clbit])
        # Note that 'cr2' is not in the condition.
        outer.if_test(expr.logic_and(expr.equal(cr1, 3), expr.logic_not(clbit)), inner, [0], [])
        outer.while_loop(expr.logic_or(expr.less(2, cr1), clbit), inner, [0], [])

        dag = circuit_to_dag(outer)
        expected_wires = set(outer.qubits) | set(cr1) | {clbit}
        for node in dag.topological_op_nodes():
            test_wires = {wire for _source, _dest, wire in dag.edges(node)}
            self.assertIsInstance(node.op.condition, expr.Expr)
            self.assertEqual(test_wires, expected_wires)

        roundtripped = dag_to_circuit(dag)
        for original, test in zip(outer, roundtripped):
            self.assertEqual(original.operation.condition, test.operation.condition)

    def test_wires_from_expr_nodes_target(self):
        """Test that the classical wires implied by an `Expr` node in a control-flow op's
        `target` are correctly transferred."""
        case_1 = QuantumCircuit(1)
        case_1.x(0)
        case_2 = QuantumCircuit(1)
        case_2.y(0)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        outer = QuantumCircuit(QuantumRegister(1), cr1, cr2)
        # Note that 'cr2' is not in the condition.
        outer.switch(expr.bit_and(cr1, 2), [(1, case_1), (2, case_2)], [0], [])

        dag = circuit_to_dag(outer)
        expected_wires = set(outer.qubits) | set(cr1)
        for node in dag.topological_op_nodes():
            test_wires = {wire for _source, _dest, wire in dag.edges(node)}
            self.assertIsInstance(node.op.target, expr.Expr)
            self.assertEqual(test_wires, expected_wires)

        roundtripped = dag_to_circuit(dag)
        for original, test in zip(outer, roundtripped):
            self.assertEqual(original.operation.target, test.operation.target)

    def test_wire_order(self):
        """Test that the `qubit_order` and `clbit_order` parameters are respected."""
        permutation = [2, 3, 1, 4, 0, 5]  # Arbitrary.
        qr = QuantumRegister(len(permutation), "qr")
        cr = ClassicalRegister(len(permutation), "cr")

        qubits_permuted = [qr[i] for i in permutation]
        clbits_permuted = [cr[i] for i in permutation]

        qc = QuantumCircuit(qr, cr)
        for q, c in zip(qr, cr):
            qc.h(q)
            qc.measure(q, c)

        dag = circuit_to_dag(qc, qubit_order=qubits_permuted, clbit_order=clbits_permuted)

        expected = DAGCircuit()
        expected.add_qubits(qubits_permuted)
        expected.add_clbits(clbits_permuted)
        expected.add_qreg(qr)
        expected.add_creg(cr)
        for q, c in zip(qr, cr):
            expected.apply_operation_back(HGate(), [q], [])
            expected.apply_operation_back(Measure(), [q], [c])

        self.assertEqual(dag, expected)
        self.assertEqual(list(dag.qubits), qubits_permuted)
        self.assertEqual(list(dag.clbits), clbits_permuted)

    def test_wire_order_failures(self):
        """Test that the `qubit_order` and `clbit_order` parameters raise on bad inputs."""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(qr, cr)

        with self.assertRaisesRegex(ValueError, "does not contain exactly the same"):
            circuit_to_dag(qc, qubit_order=qc.qubits[:-1])
        with self.assertRaisesRegex(ValueError, "does not contain exactly the same"):
            circuit_to_dag(qc, qubit_order=qr[[0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "does not contain exactly the same"):
            circuit_to_dag(qc, clbit_order=qc.clbits[:-1])
        with self.assertRaisesRegex(ValueError, "does not contain exactly the same"):
            circuit_to_dag(qc, clbit_order=cr[[0, 1, 1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
