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

"""Regression tests for ftir_mir_to_qiskit_circuit_translator.py."""

import os
import unittest

from circuit_emitter import InMemoryCircuitEmitter
from ftir_mir_to_qiskit_circuit_translator import translate

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "mlir_examples")
EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "expected")


class TestFtirMirToQiskitCircuitTranslator(unittest.TestCase):
    def test_all_fixtures_match_expected(self):
        mlir_files = sorted(f for f in os.listdir(FIXTURES_DIR) if f.endswith(".mlir"))
        self.assertTrue(mlir_files, f"no .mlir fixtures found in {FIXTURES_DIR}")

        for mlir_file in mlir_files:
            with self.subTest(fixture=mlir_file):
                source_path = os.path.join(FIXTURES_DIR, mlir_file)
                expected_path = os.path.join(EXPECTED_DIR, mlir_file.replace(".mlir", ".txt"))

                circuit = translate(source_path, InMemoryCircuitEmitter())
                actual = str(circuit)

                with open(expected_path) as expected_file:
                    expected = expected_file.read()

                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
