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

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.algorithms.time_evolution.evolution_result import EvolutionResult
from qiskit.opflow import Zero, X


class TestEvolutionResult(QiskitAlgorithmsTestCase):
    """Class for testing evolution result and relevant metadata."""

    def test_init_state(self):
        """Tests that a class is initialized correctly with an evolved_state."""
        evolved_state = Zero
        evo_result = EvolutionResult(evolved_state=evolved_state)

        expected_state = Zero
        expected_observable = None

        self.assertEqual(evo_result.evolved_state, expected_state)
        self.assertEqual(evo_result.evolved_observable, expected_observable)

    def test_init_observable(self):
        """Tests that a class is initialized correctly with an evolved_observable."""
        evolved_observable = X
        evo_result = EvolutionResult(evolved_observable=evolved_observable)

        expected_state = None
        expected_observable = X

        self.assertEqual(evo_result.evolved_state, expected_state)
        self.assertEqual(evo_result.evolved_observable, expected_observable)
