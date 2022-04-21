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
from qiskit.circuit.library import HGate, CXGate, UGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import GatesInBasis
from qiskit.transpiler.target import Target
from qiskit.test import QiskitTestCase
from qiskit.test.mock.fake_backend_v2 import FakeBackend5QV2


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

    def test_all_gates_in_basis_with_target(self):
        """Test circuit with all gates in basis with target."""
        target = FakeBackend5QV2().target
        basis_gates = ["cx", "u"]  # not used
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.u(0, 0, 0, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_not_in_basis_with_target(self):
        """Test circuit with not all gates in basis with target."""
        target = FakeBackend5QV2().target
        basis_gates = ["cx", "h"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_not_on_all_qubits_with_target(self):
        """Test circuit with gate in global basis but not local basis."""
        target = FakeBackend5QV2().target
        basis_gates = ["ecr", "cx", "h"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ecr(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_empty_circuit_with_target(self):
        """Test circuit with no gates with target."""
        target = FakeBackend5QV2().target
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_in_empty_target(self):
        """Test circuit with gates and empty basis with target."""
        target = Target()
        basis_gates = []
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_ideal_sim_target(self):
        """Test with target that has ideal gates."""
        target = Target(num_qubits=2)
        target.add_instruction(HGate())
        target.add_instruction(CXGate())
        target.add_instruction(Measure())
        property_set = {}
        analysis_pass = GatesInBasis(target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_not_in_ideal_sim_target(self):
        """Test with target that has ideal gates."""
        target = Target()
        target.add_instruction(HGate())
        target.add_instruction(UGate(0, 0, 0))
        target.add_instruction(Measure())
        property_set = {}
        analysis_pass = GatesInBasis(target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_after_translation_with_target(self):
        """Test circuit with gates in basis after conditional translation."""
        target = FakeBackend5QV2().target
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])
        pm = PassManager()
        pm.append(analysis_pass)
        pm.append(
            BasisTranslator(SessionEquivalenceLibrary, basis_gates, target=target),
            condition=lambda property_set: not property_set["all_gates_in_basis"],
        )
        pm.append(analysis_pass)
        pm.run(circuit)
        self.assertTrue(pm.property_set["all_gates_in_basis"])
