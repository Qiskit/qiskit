# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Class for testing evolution result."""
import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.algorithms.evolvers.evolution_result import EvolutionResult
from qiskit.opflow import Zero


class TestEvolutionResult(QiskitAlgorithmsTestCase):
    """Class for testing evolution result and relevant metadata."""

    def test_init_state(self):
        """Tests that a class is initialized correctly with an evolved_state."""
        evolved_state = Zero
        with self.assertWarns(DeprecationWarning):
            evo_result = EvolutionResult(evolved_state=evolved_state)

        expected_state = Zero
        expected_aux_ops_evaluated = None

        self.assertEqual(evo_result.evolved_state, expected_state)
        self.assertEqual(evo_result.aux_ops_evaluated, expected_aux_ops_evaluated)

    def test_init_observable(self):
        """Tests that a class is initialized correctly with an evolved_observable."""
        evolved_state = Zero
        evolved_aux_ops_evaluated = [(5j, 5j), (1.0, 8j), (5 + 1j, 6 + 1j)]
        with self.assertWarns(DeprecationWarning):
            evo_result = EvolutionResult(evolved_state, evolved_aux_ops_evaluated)

        expected_state = Zero
        expected_aux_ops_evaluated = [(5j, 5j), (1.0, 8j), (5 + 1j, 6 + 1j)]

        self.assertEqual(evo_result.evolved_state, expected_state)
        self.assertEqual(evo_result.aux_ops_evaluated, expected_aux_ops_evaluated)


if __name__ == "__main__":
    unittest.main()
