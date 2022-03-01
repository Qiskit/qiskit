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

"""Test gradient evolution problem class."""
import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data, unpack
from qiskit.algorithms.time_evolution.problems.gradient_evolution_problem import (
    GradientEvolutionProblem,
)
from qiskit.circuit import Parameter
from qiskit.opflow import Y, Z, One, Gradient, Zero


@ddt
class TestGradientEvolutionProblem(QiskitAlgorithmsTestCase):
    """Test gradient evolution problem class."""

    def test_init_default(self):
        """Tests that all default fields are initialized correctly."""
        hamiltonian = Y
        time = 2.5
        gradient_object = Gradient()
        initial_state = Zero

        evo_problem = GradientEvolutionProblem(hamiltonian, time, gradient_object, initial_state)

        expected_hamiltonian = Y
        expected_time = 2.5
        expected_gradient_object = gradient_object
        expected_initial_state = Zero
        expected_observable = None
        expected_t_param = None
        expected_hamiltonian_value_dict = None
        expected_gradient_params = None

        self.assertEqual(evo_problem.hamiltonian, expected_hamiltonian)
        self.assertEqual(evo_problem.time, expected_time)
        self.assertEqual(evo_problem.gradient_object, expected_gradient_object)
        self.assertEqual(evo_problem.initial_state, expected_initial_state)
        self.assertEqual(evo_problem.observable, expected_observable)
        self.assertEqual(evo_problem.t_param, expected_t_param)
        self.assertEqual(evo_problem.hamiltonian_value_dict, expected_hamiltonian_value_dict)
        self.assertEqual(evo_problem.gradient_params, expected_gradient_params)

    def test_init_all(self):
        """Tests that all fields are initialized correctly."""
        t_parameter = Parameter("t")
        param = Parameter("x")
        hamiltonian = t_parameter * Z + param * Y
        time = 2
        gradient_object = Gradient()

        initial_state = One
        observable = None
        hamiltonian_value_dict = {t_parameter: 3.2}
        gradient_params = [param]

        evo_problem = GradientEvolutionProblem(
            hamiltonian,
            time,
            gradient_object,
            initial_state,
            observable,
            t_parameter,
            hamiltonian_value_dict,
            gradient_params,
        )

        expected_hamiltonian = param * Y + t_parameter * Z
        expected_time = 2
        expected_gradient_object = gradient_object
        expected_initial_state = One
        expected_observable = None
        expected_t_param = t_parameter
        expected_hamiltonian_value_dict = {t_parameter: 3.2}
        expected_gradient_params = [param]

        self.assertEqual(evo_problem.hamiltonian, expected_hamiltonian)
        self.assertEqual(evo_problem.time, expected_time)
        self.assertEqual(evo_problem.gradient_object, expected_gradient_object)
        self.assertEqual(evo_problem.initial_state, expected_initial_state)
        self.assertEqual(evo_problem.observable, expected_observable)
        self.assertEqual(evo_problem.t_param, expected_t_param)
        self.assertEqual(evo_problem.hamiltonian_value_dict, expected_hamiltonian_value_dict)
        self.assertEqual(evo_problem.gradient_params, expected_gradient_params)

    @data((One, Y), (None, None))
    @unpack
    def test_init_error(self, initial_state, observable):
        """Tests that an error is raised when both or none initial_state and observable provided."""
        t_parameter = Parameter("t")
        hamiltonian = t_parameter * Z + Y
        time = 2
        gradient_object = Gradient()

        with self.assertRaises(ValueError):
            _ = GradientEvolutionProblem(
                hamiltonian, time, gradient_object, initial_state, observable
            )


if __name__ == "__main__":
    unittest.main()
