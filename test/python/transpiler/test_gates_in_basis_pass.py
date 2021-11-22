# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test GatesInBasis pass."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import GatesInBasis
from qiskit.test import QiskitTestCase


class TestGatesInBasisPass(QiskitTestCase):
    """Tests for GatesInBasis pass."""

    def test_all_gates_in_basis(self):
        """Test circuit with all gates in basis."""
        basis_gates = ["cx", "h"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_not_in_basis(self):
        """Test circuit with not all gates in basis."""
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_empty_circuit(self):
        """Test circuit with no gates."""
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_empty_basis(self):
        """Test circuit with gates and empty basis."""
        basis_gates = []
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_after_translation(self):
        """Test circuit with gates in basis after conditional translation."""
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])
        pm = PassManager()
        pm.append(analysis_pass)
        pm.append(
            BasisTranslator(SessionEquivalenceLibrary, basis_gates),
            condition=lambda property_set: not property_set["all_gates_in_basis"],
        )
        pm.append(analysis_pass)
        pm.run(circuit)
        self.assertTrue(pm.property_set["all_gates_in_basis"])
