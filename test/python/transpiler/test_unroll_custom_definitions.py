# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the BasisTranslator pass"""

from qiskit.transpiler.passes.basis import UnrollCustomDefinitions

from qiskit.test import QiskitTestCase
from qiskit.circuit import EquivalenceLibrary, Gate
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.exceptions import QiskitError


class TestGate(Gate):
    """Mock one qubit zero param gate."""
    def __init__(self):
        super().__init__('tg', 1, [])


class TestCompositeGate(Gate):
    """Mock one qubit zero param gate."""
    def __init__(self):
        super().__init__('tcg', 1, [])


class TestUnrollCustomDefinitions(QiskitTestCase):
    """Test the UnrollCustomDefinitions pass."""

    def test_dont_unroll_a_gate_in_eq_lib(self):
        """Verify we don't unroll a gate found in equivalence_library."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = UnrollCustomDefinitions(eq_lib, ['u3', 'cx']).run(dag)

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_dont_unroll_a_gate_in_basis_gates(self):
        """Verify we don't unroll a gate in basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = UnrollCustomDefinitions(eq_lib, ['u3', 'cx', 'tg']).run(dag)

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_raise_for_opaque_not_in_eq_lib(self):
        """Verify we raise for an opaque gate not in basis_gates or eq_lib."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        with self.assertRaisesRegex(QiskitError, 'Cannot unroll'):
            UnrollCustomDefinitions(eq_lib, ['u3', 'cx']).run(dag)

    def test_unroll_gate_until_reach_basis_gates(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestCompositeGate()
        q = QuantumRegister(1, 'q')
        gate.definition = QuantumCircuit(q)
        gate.definition.data = [(TestGate(), [q[0]], [])]

        qc = QuantumCircuit(q)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = UnrollCustomDefinitions(eq_lib, ['u3', 'cx', 'tg']).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_twice_until_we_get_to_eqlib(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(base_gate, equiv)

        gate = TestCompositeGate()

        q = QuantumRegister(1, 'q')
        gate.definition = QuantumCircuit(q)
        gate.definition.data = [(TestGate(), [q[0]], [])]

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = UnrollCustomDefinitions(eq_lib, ['u3', 'cx']).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)
