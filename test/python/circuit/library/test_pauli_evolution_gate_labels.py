# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0.
# You may obtain a copy of this license in the LICENSE.txt file in the root directory of this source tree.
#
# -----------------------------------------------------------------------------------
# This file tests that PauliEvolutionGate correctly generates labels from SparseObservable inputs.

"""Tests for labels in :class:`~qiskit.circuit.library.PauliEvolutionGate`."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparseObservable
from qiskit.test import QiskitTestCase


class TestPauliEvolutionGateLabels(QiskitTestCase):
    """Test cases for verifying the labels of PauliEvolutionGate."""

    def test_single_term_label(self):
        """Test label for a single Pauli term."""
        evo = PauliEvolutionGate(SparseObservable.from_list([("XXII", 1)]), time=1)
        self.assertIn("XXII", evo.label)

    def test_multiple_term_label(self):
        """Test label for multiple Pauli terms in a single SparseObservable."""
        evo = PauliEvolutionGate(
            SparseObservable.from_list([("IIXX", 1), ("IYYI", 2), ("ZZII", 3)]), time=1
        )
        self.assertIn("IIXX", evo.label)
        self.assertIn("IYYI", evo.label)
        self.assertIn("ZZII", evo.label)

    def test_list_of_observables_label(self):
        """Test label when given a list of multiple SparseObservables."""
        evo = PauliEvolutionGate(
            [
                SparseObservable.from_list([("IIXX", 1), ("IYYI", 2), ("ZZII", 3)]),
                SparseObservable.from_list([("XXII", 4)]),
            ],
            time=1,
        )
        label = evo.label
        self.assertIn("IIXX", label)
        self.assertIn("XXII", label)

    def test_circuit_display_labels(self):
        """Test that circuit drawing correctly displays the Pauli labels."""
        evo = PauliEvolutionGate(SparseObservable.from_list([("XXII", 1)]), time=1)
        qc = QuantumCircuit(4)
        qc.append(evo, [0, 1, 2, 3])
        text = qc.draw(output="text")
        self.assertIn("XXII", text)
