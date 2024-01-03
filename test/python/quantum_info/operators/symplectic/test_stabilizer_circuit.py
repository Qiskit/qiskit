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
"""Tests for stabilizer_to_circuit function."""
from __future__ import annotations

import unittest

from qiskit.quantum_info.operators.symplectic.stabilizer_circuit import stabilizer_to_circuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Pauli, StabilizerState
from qiskit.quantum_info.operators import Clifford

from qiskit.test import QiskitTestCase


class TestStabilizerCircuits(QiskitTestCase):
    """Tests for stabilizer_to_circuit function."""

    def verify_stabilizers(self, stabilizer_list: list[str], **kwargs):
        """
        Verify that circuit generated from stabilizer_list is correct.
        :param stabilizer_list: list of stabilizer strings
        :param kwargs: keyword arguments for stabilizer_to_circuit
        """
        circuit = stabilizer_to_circuit(stabilizer_list, **kwargs)
        clifford = Clifford(circuit)
        state = StabilizerState(circuit)
        for stabilizer in stabilizer_list:
            composed = clifford.compose(Pauli(stabilizer).to_instruction())
            # Test that the stabilizer is a stabilizer of the state by applying it to state
            self.assertTrue(state.equiv(StabilizerState(composed)))

    def test_stabilizer_to_circuit_simple(self):
        """Simple test case"""
        stabilizer_list = ["ZXX", "XYX", "ZYY"]
        self.verify_stabilizers(stabilizer_list)

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
        """Redundant test case that requires to swap qubits"""
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


if __name__ == "__main__":
    unittest.main()
