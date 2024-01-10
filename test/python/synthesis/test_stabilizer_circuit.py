# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for synth_circuit_from_stabilizers function."""
from __future__ import annotations

import unittest
from collections.abc import Collection

from test import combine

from ddt import ddt

import numpy as np

from qiskit.synthesis import synth_circuit_from_stabilizers
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Pauli, StabilizerState, random_clifford
from qiskit.quantum_info.operators import Clifford

from qiskit.test import QiskitTestCase


@ddt
class TestStabilizerCircuits(QiskitTestCase):
    """Tests for synth_circuit_from_stabilizers function."""

    def verify_stabilizers(self, stabilizers: Collection[str], **kwargs) -> None:
        """
        Verify that the circuit generated from stabilizers is correct.

        Args:
            stabilizers (Collection[str]): list of stabilizer strings
            kwargs: keyword arguments for synth_circuit_from_stabilizer_list
        """
        circuit = synth_circuit_from_stabilizers(stabilizers, **kwargs)
        clifford = Clifford(circuit)
        state = StabilizerState(circuit)
        for stabilizer in stabilizers:
            composed = clifford.compose(Pauli(stabilizer).to_instruction())
            # Test that the stabilizer is a stabilizer of the state by applying it to state
            self.assertTrue(state.equiv(StabilizerState(composed)))

    def test_stabilizer_to_circuit_simple(self):
        """Simple test case"""
        stabilizers = {"+ZXX", "+XYX", "+ZYY"}
        self.verify_stabilizers(stabilizers)

    def test_stabilizer_to_circuit_with_sign(self):
        """Simple test case with signs stabilizer"""
        stabilizers = {"ZXX", "-XYX", "+ZYY"}
        self.verify_stabilizers(stabilizers)

    def test_stabilizer_to_circuit_larger(self):
        """Larger test case"""
        stabilizer_list = [
            "YXIZZIXY",
            "ZIXYYXIZ",
            "IZYXXYZI",
            "XIIZIIXZ",
            "IIXZXIIZ",
            "IZXIXZII",
            "YYYYIIII",
            "XXIIXXII",
        ]
        self.verify_stabilizers(stabilizer_list)

    def test_stabilizer_to_circuit_underconstrained(self):
        """Underconstrained test case"""
        stabilizer_list = ["ZXX", "ZYY"]

        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list)
        self.assertEqual(
            cm.exception.message,
            "Stabilizers are underconstrained and allow_underconstrained is False. Add "
            "allow_underconstrained=True  to the function call if you want to allow "
            "underconstrained stabilizers.",
        )
        self.verify_stabilizers(stabilizer_list, allow_underconstrained=True)

    def test_stabilizer_to_circuit_redundant(self):
        """Redundant test case"""
        stabilizer_list = ["ZZXX", "XXXX", "XXZZ", "ZZZZ"]
        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list)
        self.assertEqual(
            cm.exception.message,
            f"Stabilizer 3 ({stabilizer_list[3]}) is a product of the others "
            "and allow_redundant is False. Add allow_redundant=True "
            "to the function call if you want to allow redundant stabilizers.",
        )
        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list, allow_redundant=True)
        self.assertEqual(
            cm.exception.message,
            "Stabilizers are underconstrained and allow_underconstrained is False. Add "
            "allow_underconstrained=True  to the function call if you want to allow "
            "underconstrained stabilizers.",
        )
        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list, allow_underconstrained=True)
        self.assertEqual(
            cm.exception.message,
            f"Stabilizer 3 ({stabilizer_list[3]}) is a product of the others "
            "and allow_redundant is False. Add allow_redundant=True "
            "to the function call if you want to allow redundant stabilizers.",
        )
        self.verify_stabilizers(stabilizer_list, allow_redundant=True, allow_underconstrained=True)

    def test_stabilizer_to_circuit_redundant_swap(self):
        """Redundant test case that requires to swap qubits during Gaussian elimination
        (i.e., some pivot is off-diagonal)"""
        stabilizer_list = ["ZZXX", "ZZZZ", "XXXX", "YYYY", "XZXZ"]
        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list)
        self.assertEqual(
            cm.exception.message,
            f"Stabilizer 3 ({stabilizer_list[3]}) is a product of the others "
            "and allow_redundant is False. Add allow_redundant=True "
            "to the function call if you want to allow redundant stabilizers.",
        )
        self.verify_stabilizers(stabilizer_list, allow_redundant=True)

    def test_stabilizer_to_circuit_non_commuting(self):
        """Non-commuting stabilizers"""
        stabilizer_list = ["ZXX", "XYX", "YYY"]
        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list)
        self.assertEqual(
            cm.exception.message,
            f"Stabilizers 0 ({stabilizer_list[0]}) and {2} ({stabilizer_list[2]}) do not commute",
        )

    def test_stabilizer_to_circuit_contradicting(self):
        """Contradicting stabilizers"""
        stabilizer_list = ["ZXX", "-ZXX"]
        with self.assertRaises(QiskitError) as cm:
            self.verify_stabilizers(stabilizer_list)
        self.assertEqual(
            cm.exception.message,
            f"Stabilizer 1 ({stabilizer_list[1]}) contradicts some of the previous stabilizers",
        )

    @combine(num_qubits=[4, 5, 6, 7])
    def test_regenerate_clifford(self, num_qubits):
        """Create a circuit from Clifford-generated list of stabilizers and verify that the
        circuit output is equivalent to the original state."""
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            clifford = random_clifford(num_qubits, seed=rng)
            state = StabilizerState(clifford)

            stabilizer_list = clifford.to_labels(mode="S")
            state_syn = StabilizerState.from_stabilizer_list(stabilizer_list)
            self.assertTrue(state.equiv(StabilizerState(state_syn)))


if __name__ == "__main__":
    unittest.main()
