# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's EquivalenceLibrary class."""

import numpy as np

from qiskit.test import QiskitTestCase

from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.circuit.exceptions import CircuitError

from qiskit.circuit import EquivalenceLibrary


class OneQubitZeroParamGate(Gate):
    """Mock one qubit zero param gate."""
    def __init__(self):
        super().__init__('1q0p', 1, [])


class OneQubitOneParamGate(Gate):
    """Mock one qubit one  param gate."""
    def __init__(self, theta):
        super().__init__('1q1p', 1, [theta])


class OneQubitTwoParamGate(Gate):
    """Mock one qubit two param gate."""
    def __init__(self, phi, lam):
        super().__init__('1q2p', 1, [phi, lam])


class TestEquivalenceLibraryWithoutBase(QiskitTestCase):
    """Test cases for basic EquivalenceLibrary."""

    def test_create_empty_library(self):
        """An empty library should return an empty entry."""
        eq_lib = EquivalenceLibrary()
        self.assertIsInstance(eq_lib, EquivalenceLibrary)

        gate = OneQubitZeroParamGate()

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 0)

    def test_add_single_entry(self):
        """Verify an equivalence added to the library can be retrieved."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(gate, equiv)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertIsNot(entry[0], equiv)
        self.assertEqual(entry[0], equiv)

    def test_add_double_entry(self):
        """Verify separately added equivalences can be retrieved."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit(1)
        first_equiv.h(0)

        eq_lib.add_equivalence(gate, first_equiv)

        second_equiv = QuantumCircuit(1)
        second_equiv.u2(0, np.pi, 0)

        eq_lib.add_equivalence(gate, second_equiv)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 2)
        self.assertTrue(
            (entry[0] == first_equiv and entry[1] == second_equiv)
            or (entry[1] == first_equiv and entry[0] == second_equiv))

    def test_set_entry(self):
        """Verify setting an entry overrides any previously added."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit(1)
        first_equiv.h(0)

        eq_lib.add_equivalence(gate, first_equiv)

        second_equiv = QuantumCircuit(1)
        second_equiv.u2(0, np.pi, 0)

        eq_lib.set_entry(gate, [second_equiv])

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], second_equiv)

    def test_raise_if_gate_entry_shape_mismatch(self):
        """Verify we raise if adding a circuit and gate with different shapes."""
        # This could be relaxed in the future to e.g. support ancilla management.

        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(2)
        equiv.h(0)

        with self.assertRaises(CircuitError):
            eq_lib.add_equivalence(gate, equiv)


class TestEquivalenceLibraryWithBase(QiskitTestCase):
    """Test cases for EquivalenceLibrary with base library."""

    def test_create_empty_library_with_base(self):
        """Verify retrieving from an empty library returns an empty entry."""
        base = EquivalenceLibrary()

        eq_lib = EquivalenceLibrary(base=base)
        self.assertIsInstance(eq_lib, EquivalenceLibrary)

        gate = OneQubitZeroParamGate()

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 0)

    def test_get_through_empty_library_to_base(self):
        """Verify we find an entry defined only in the base library."""
        base = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)
        base.add_equivalence(gate, equiv)

        eq_lib = EquivalenceLibrary(base=base)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertIsNot(entry[0], equiv)
        self.assertEqual(entry[0], equiv)

    def test_add_equivalence(self):
        """Verify we find all equivalences if a gate is added to top and base."""
        base = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit(1)
        first_equiv.h(0)
        base.add_equivalence(gate, first_equiv)

        eq_lib = EquivalenceLibrary(base=base)

        second_equiv = QuantumCircuit(1)
        second_equiv.u2(0, np.pi, 0)

        eq_lib.add_equivalence(gate, second_equiv)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 2)
        self.assertTrue(
            (entry[0] == first_equiv and entry[1] == second_equiv)
            or (entry[1] == first_equiv and entry[0] == second_equiv))

    def test_set_entry(self):
        """Verify we find only equivalences from top when explicitly set."""
        base = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit(1)
        first_equiv.h(0)
        base.add_equivalence(gate, first_equiv)

        eq_lib = EquivalenceLibrary(base=base)

        second_equiv = QuantumCircuit(1)
        second_equiv.u2(0, np.pi, 0)

        eq_lib.set_entry(gate, [second_equiv])

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], second_equiv)


class TestEquivalenceLibraryWithParameters(QiskitTestCase):
    """Test cases for EquivalenceLibrary with gate parameters."""

    def test_raise_if_gate_equiv_parameter_mismatch(self):
        """Verify we raise if adding a circuit and gate with different sets of parameters."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter('theta')
        phi = Parameter('phi')

        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1)
        equiv.u1(phi, 0)

        with self.assertRaises(CircuitError):
            eq_lib.add_equivalence(gate, equiv)

        with self.assertRaises(CircuitError):
            eq_lib.set_entry(gate, [equiv])

    def test_parameter_in_parameter_out(self):
        """Verify query parameters will be included in returned entry."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter('theta')

        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1)
        equiv.u1(theta, 0)

        eq_lib.add_equivalence(gate, equiv)

        phi = Parameter('phi')
        gate_phi = OneQubitOneParamGate(phi)

        entry = eq_lib.get_entry(gate_phi)

        expected = QuantumCircuit(1)
        expected.u1(phi, 0)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], expected)

    def test_partial_parameter_in_parameter_out(self):
        """Verify numeric query parameters will be included in returned entry."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter('theta')
        phi = Parameter('phi')

        gate = OneQubitTwoParamGate(theta, phi)
        equiv = QuantumCircuit(1)
        equiv.u2(theta, phi, 0)

        eq_lib.add_equivalence(gate, equiv)

        lam = Parameter('lam')
        gate_partial = OneQubitTwoParamGate(lam, 1.59)

        entry = eq_lib.get_entry(gate_partial)

        expected = QuantumCircuit(1)
        expected.u2(lam, 1.59, 0)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], expected)

    def test_adding_gate_under_different_parameters(self):
        """Verify a gate can be added under different sets of parameters."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter('theta')

        gate_theta = OneQubitOneParamGate(theta)
        equiv_theta = QuantumCircuit(1)
        equiv_theta.u1(theta, 0)

        eq_lib.add_equivalence(gate_theta, equiv_theta)

        phi = Parameter('phi')
        gate_phi = OneQubitOneParamGate(phi)
        equiv_phi = QuantumCircuit(1)
        equiv_phi.rz(phi, 0)

        eq_lib.add_equivalence(gate_phi, equiv_phi)

        lam = Parameter('lam')
        gate_query = OneQubitOneParamGate(lam)

        entry = eq_lib.get_entry(gate_query)

        first_expected = QuantumCircuit(1)
        first_expected.u1(lam, 0)

        second_expected = QuantumCircuit(1)
        second_expected.rz(lam, 0)

        self.assertEqual(len(entry), 2)
        self.assertTrue(
            (entry[0] == first_expected and entry[1] == second_expected)
            or (entry[1] == first_expected and entry[0] == second_expected))

    def test_adding_gate_and_partially_specified_gate(self):
        """Verify entries will different numbers of parameters will be returned."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter('theta')
        phi = Parameter('phi')

        # e.g. RGate(theta, phi)
        gate_full = OneQubitTwoParamGate(theta, phi)
        equiv_full = QuantumCircuit(1)
        equiv_full.u2(theta, phi, 0)

        eq_lib.add_equivalence(gate_full, equiv_full)

        gate_partial = OneQubitTwoParamGate(theta, 0)
        equiv_partial = QuantumCircuit(1)
        equiv_partial.rx(theta, 0)

        eq_lib.add_equivalence(gate_partial, equiv_partial)

        lam = Parameter('lam')
        gate_query = OneQubitTwoParamGate(lam, 0)

        entry = eq_lib.get_entry(gate_query)

        first_expected = QuantumCircuit(1)
        first_expected.rx(lam, 0)

        second_expected = QuantumCircuit(1)
        second_expected.u2(lam, 0, 0)

        self.assertEqual(len(entry), 2)
        self.assertTrue(
            (entry[0] == first_expected and entry[1] == second_expected)
            or (entry[1] == first_expected and entry[0] == second_expected))
