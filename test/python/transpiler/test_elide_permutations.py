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

"""Test ElidePermutations pass"""

import unittest

from test import QiskitTestCase

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.compiler.transpiler import transpile
from qiskit.circuit.library.generalized_gates import PermutationGate
from qiskit.transpiler.passes.optimization.elide_permutations import ElidePermutations
from qiskit.circuit.controlflow import IfElseOp
from qiskit.quantum_info import Operator
from qiskit.transpiler.coupling import CouplingMap


class TestElidePermutations(QiskitTestCase):
    """Test swap elision logical optimization pass."""

    def setUp(self):
        super().setUp()
        self.swap_pass = ElidePermutations()

    def check_equivalence(self, qc, tqc):
        """
        This is a temporary function that checks that the original circuit
        qc is equivalent to the transpiled circuit tqc, while taking
        initial and final layouts into account.
        """

        # Unfortunately, we cannot do much when the original and the transpiled circuits
        # have different numbers of qubits, since Operator-based comparison does not
        # work in this case.
        if qc.num_qubits != tqc.num_qubits:
            return

        # This should work with the correct fix to Operator.from_circuit().
        # self.assertTrue(Operator.from_circuit(transpiled_qc).equiv(Operator(qc)))

        # For now do the slow but correct-by-construction checking
        from qiskit.transpiler import Layout

        def _layout_to_perm_pattern(layout, qubits):
            # Map layout object to permutation
            perm = [0] * len(qubits)
            for i, q in enumerate(qubits):
                pos = layout._v2p[q]
                perm[pos] = i
            return perm

        t_qubits = tqc.qubits

        if (t_initial_layout := tqc.layout.initial_layout) is None:
            t_initial_layout = Layout(dict(enumerate(t_qubits)))
        t_initial_perm = _layout_to_perm_pattern(t_initial_layout, t_qubits)

        if (t_final_layout := tqc.layout.final_layout) is None:
            t_final_layout = Layout(dict(enumerate(t_qubits)))
        t_final_perm = _layout_to_perm_pattern(t_final_layout, t_qubits)

        # Add layouts to qc
        qc_with_layouts = QuantumCircuit(qc.num_qubits)
        qc_with_layouts.append(PermutationGate(t_initial_perm).inverse(), t_qubits)
        qc_with_layouts.append(qc, t_qubits)
        qc_with_layouts.append(PermutationGate(t_initial_perm), t_qubits)
        qc_with_layouts.append(PermutationGate(t_final_perm), t_qubits)

        self.assertTrue(Operator(qc_with_layouts).equiv(Operator(tqc)))

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
        """Test swap at the end of bell is elided."""
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
        """Test swap before measure is elided."""
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

        expected = QuantumCircuit(3)
        expected.h(0)
        expected.cx(2, 1)
        expected.h(2)

        # First assert the pass works as expected
        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

        # Without coupling map
        res = transpile(qc, optimization_level=3, seed_transpiler=42)
        self.check_equivalence(qc, res)

        # With coupling map
        res = transpile(
            qc, coupling_map=CouplingMap.from_line(3), optimization_level=3, seed_transpiler=1234
        )
        self.check_equivalence(qc, res)

    def test_unitary_equivalence_with_elide_and_routing(self):
        """Test full transpile pipeline with pass preserves permutation for unitary equivalence including
        a larger example and a basis translation."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.h(1)

        expected = QuantumCircuit(5)
        expected.h(0)
        expected.cx(2, 1)
        expected.cx(1, 2)
        expected.cx(1, 0)
        expected.cx(1, 3)
        expected.cx(1, 4)
        expected.h(2)

        # First assert the pass works as expected
        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

        # without coupling map
        res = transpile(qc, optimization_level=3, seed_transpiler=42)
        # self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))
        self.check_equivalence(qc, res)

        # With coupling map
        res = transpile(
            qc,
            coupling_map=CouplingMap.from_line(5),
            basis_gates=["u", "cz"],
            optimization_level=3,
            seed_transpiler=1234,
        )
        # self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))
        self.check_equivalence(qc, res)

        # Without coupling map but with initial layout
        res = transpile(
            qc,
            initial_layout=[4, 2, 1, 3, 0],
            basis_gates=["u", "cz"],
            optimization_level=3,
            seed_transpiler=1234,
        )
        self.check_equivalence(qc, res)

        # With coupling map and with initial layout
        res = transpile(
            qc,
            coupling_map=CouplingMap.from_line(5),
            initial_layout=[4, 2, 1, 3, 0],
            basis_gates=["u", "cz"],
            optimization_level=3,
            seed_transpiler=1234,
        )
        self.check_equivalence(qc, res)

        # With coupling map over more qubits
        res = transpile(
            qc,
            optimization_level=3,
            seed_transpiler=42,
            coupling_map=CouplingMap.from_line(8),
        )
        self.check_equivalence(qc, res)

        # With coupling map over more qubits and initial layout
        res = transpile(
            qc,
            initial_layout=[4, 2, 7, 3, 6],
            optimization_level=3,
            seed_transpiler=42,
            coupling_map=CouplingMap.from_line(8),
        )
        self.check_equivalence(qc, res)

    def test_permutation_in_middle(self):
        """Test permutation in middle of bell is elided."""
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
        expected.cx(1, 0)
        expected.barrier(0, 1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_at_beginning(self):
        """Test permutation in beginning of bell is elided."""
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
        expected.cx(1, 0)
        expected.barrier(0, 1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_at_end(self):
        """Test permutation at end of bell is elided."""
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
        """Test permutation before measure is elided."""
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
