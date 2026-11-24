# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Smoke test of a compiled extension module."""

import unittest

import qk_smoke

import qiskit
from qiskit import capi


class TestCAPIExtension(unittest.TestCase):
    def test_api_version_matches(self):
        """Test that the version read through the function table matches the library symbol."""
        self.assertEqual(qk_smoke.api_version(), capi.qk_api_version())

    def test_circuit_to_python(self):
        """Test that we get a bonafide ``QuantumCircuit`` out of the extension."""
        qc = qk_smoke.bell()
        self.assertIsInstance(qc, qiskit.QuantumCircuit)
        self.assertEqual([inst.operation.name for inst in qc.data], ["h", "cx"])

    def test_circuit_borrow_from_python(self):
        """Test that C can read a circuit built in Python."""
        qc = qiskit.QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        self.assertEqual(qk_smoke.num_instructions(qc._data), 3)


if __name__ == "__main__":
    unittest.main()
