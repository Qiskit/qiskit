# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test ElideSwap pass"""

import unittest

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.compiler.transpiler import transpile
from qiskit.circuit.library.generalized_gates import PermutationGate
from qiskit.transpiler.passes.optimization.elide_permutations import ElidePermutations
from qiskit.test import QiskitTestCase
from qiskit.circuit.controlflow import IfElseOp
from qiskit.quantum_info import Operator
from qiskit.transpiler.coupling import CouplingMap


class TestElidePermutations(QiskitTestCase):
    """Test swap elision logical optimization pass."""

    def setUp(self):
        super().setUp()
        self.swap_pass = ElidePermutations()

    def test_no_swap(self):
        """Test no swap means no transform."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        res = self.swap_pass(qc)
        self.assertEqual(res, qc)

    def test_swap_in_middle(self):
        """Test swap in middle of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(0, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_at_beginning(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.swap(0, 1)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(1)
        expected.cx(0, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_at_end(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.swap(0, 1)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_before_measure(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.swap(0, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_if_else_block(self):
        """Test swap elision only happens outside control flow."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        with qc.if_test((0, 0)):
            qc.swap(0, 1)
        qc.cx(0, 1)
        res = self.swap_pass(qc)
        self.assertEqual(res, qc)

    def test_swap_if_else_block_with_outside_swap(self):
        """Test swap elision only happens outside control flow."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.swap(2, 0)
        body = QuantumCircuit(2)
        body.swap(0, 1)
        if_else_op = IfElseOp((qc.clbits[0], 0), body)

        qc.append(if_else_op, [0, 1])
        qc.cx(0, 1)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.append(IfElseOp((expected.clbits[0], 0), body), [2, 1])
        expected.cx(2, 1)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_condition(self):
        """Test swap elision doesn't touch conditioned swap."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.swap(0, 1).c_if(qc.clbits[0], 0)
        qc.cx(0, 1)
        res = self.swap_pass(qc)
        self.assertEqual(res, qc)

    def test_unitary_equivalence_pass_manager(self):
        """Test full transpile pipeline with pass preserves permutation for unitary equivalence."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.h(1)

        # with no layout
        res = transpile(qc, optimization_level=3, seed_transpiler=42)
        self.assertTrue(Operator.from_circuit(res).equiv(qc))
        # With layout
        res = transpile(qc, coupling_map=CouplingMap.from_line(3), optimization_level=3)
        self.assertTrue(Operator.from_circuit(res).equiv(qc))

    def test_permutation_in_middle(self):
        """Test swap in middle of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(2, 0)
        expected.barrier(0, 1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_at_beginning(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(2)
        expected.cx(2, 0)
        expected.barrier(0, 1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_at_end(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_and_permutation(self):
        """Test a combination of swap and permutation gates."""
        qc = QuantumCircuit(3, 3)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        qc.swap(0, 2)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_before_measure(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(2, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)


if __name__ == "__main__":
    unittest.main()
