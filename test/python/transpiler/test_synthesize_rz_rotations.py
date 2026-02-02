# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SynthesizeRzRotations pass."""

### [WIP Note] More tests need to be added, these are just basic sanity checks.

import unittest
import numpy as np

from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    RZGate,
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
)
from qiskit.quantum_info import Operator
from qiskit.quantum_info import get_clifford_gate_names


# from qiskit.synthesis import gridsynth_rz, gridsynth_unitary
# from qiskit.synthesis import synthesize_rz_rotations
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.synthesis import SynthesizeRZRotations

from test import QiskitTestCase  # pylint: disable=wrong-import-order


# Set of single-qubit Clifford gates
CLIFFORD_GATES_1Q_SET = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}

# Set of Clifford+T gates
CLIFFORD_T_GATES_SET = set(get_clifford_gate_names() + ["t", "tdg"])


@ddt
class TestSynthesizeRzRotations(QiskitTestCase):
    """Test Synthesize Rz rotations method"""

    def test_synthesize_rz_rotations(self):
        """Test that synthesize_rz_rotations works correctly."""
        num_trials = 40
        for angle in np.linspace(-2 * np.pi, 2 * np.pi, num_trials):
            with self.subTest(angle=angle):
                # Approximate RZ-rotation
                qc = QuantumCircuit(1)
                qc.rz(angle, 0)
                synthesized_circ = SynthesizeRZRotations()(qc)
                # Check the operators are (almost) equal
                self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(10, -10)
    def test_synthesize_rz_with_nonstandard_angles(self, angle):
        """Test that synthesize_rz_rotations works correctly."""
        # Approximate RZ-rotation
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        synthesized_circ = SynthesizeRZRotations()(qc)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    # [TO DO] could combine tests for angle and epsilon

    @data(1e-9, 1e-10, 1e-11)
    def test_synthesize_rz_with_epsilon(self, epsilon):
        """Test that synthesize_rz_rotations works correctly."""
        # Approximate RZ-rotation
        qc = QuantumCircuit(1)
        angle = np.random.uniform(0, 4 * np.pi)
        qc.rz(angle, 0)
        synthesized_circ = SynthesizeRZRotations(epsilon=epsilon)(qc)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    def test_gridsynth_rz_deterministic(self):
        """Test that calling synthesize_rz_rotations multiple times produces the same circuit."""
        angle = 1.2345
        num_trials = 10

        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        approximate_circuits = [SynthesizeRZRotations()(qc) for _ in range(num_trials)]

        for idx in range(1, len(approximate_circuits)):
            self.assertEqual(approximate_circuits[idx], approximate_circuits[0])

    ## test to check all ops are clifford+T
    ## test to check
