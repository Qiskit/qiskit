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

"""Test evolver problem class."""
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt, unpack
from numpy.testing import assert_raises

from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.circuit import Parameter
from qiskit.opflow import Y, Z, One, X, Zero


@ddt
class TestEvolutionProblem(QiskitAlgorithmsTestCase):
    """Test evolver problem class."""

    def test_init_default(self):
        """Tests that all default fields are initialized correctly."""
        hamiltonian = Y
        time = 2.5
        initial_state = One

        evo_problem = EvolutionProblem(hamiltonian, time, initial_state)

        expected_hamiltonian = Y
        expected_time = 2.5
        expected_initial_state = One
        expected_aux_operators = None
        expected_t_param = None
        expected_param_value_dict = None

        self.assertEqual(evo_problem.hamiltonian, expected_hamiltonian)
        self.assertEqual(evo_problem.time, expected_time)
        self.assertEqual(evo_problem.initial_state, expected_initial_state)
        self.assertEqual(evo_problem.aux_operators, expected_aux_operators)
        self.assertEqual(evo_problem.t_param, expected_t_param)
        self.assertEqual(evo_problem.param_value_dict, expected_param_value_dict)

    def test_init_all(self):
        """Tests that all fields are initialized correctly."""
        t_parameter = Parameter("t")
        hamiltonian = t_parameter * Z + Y
        time = 2
        initial_state = One
        aux_operators = [X, Y]
        param_value_dict = {t_parameter: 3.2}

        evo_problem = EvolutionProblem(
            hamiltonian,
            time,
            initial_state,
            aux_operators,
            t_param=t_parameter,
            param_value_dict=param_value_dict,
        )

        expected_hamiltonian = Y + t_parameter * Z
        expected_time = 2
        expected_initial_state = One
        expected_aux_operators = [X, Y]
        expected_t_param = t_parameter
        expected_param_value_dict = {t_parameter: 3.2}

        self.assertEqual(evo_problem.hamiltonian, expected_hamiltonian)
        self.assertEqual(evo_problem.time, expected_time)
        self.assertEqual(evo_problem.initial_state, expected_initial_state)
        self.assertEqual(evo_problem.aux_operators, expected_aux_operators)
        self.assertEqual(evo_problem.t_param, expected_t_param)
        self.assertEqual(evo_problem.param_value_dict, expected_param_value_dict)

    @data([Y, -1, One], [Y, -1.2, One], [Y, 0, One])
    @unpack
    def test_init_errors(self, hamiltonian, time, initial_state):
        """Tests expected errors are thrown on invalid time argument."""
        with assert_raises(ValueError):
            _ = EvolutionProblem(hamiltonian, time, initial_state)

    def test_validate_params(self):
        """Tests expected errors are thrown on parameters mismatch."""
        param_x = Parameter("x")
        param_y = Parameter("y")
        with self.subTest(msg="Parameter missing in dict."):
            hamiltonian = param_x * X + param_y * Y
            param_dict = {param_y: 2}
            evolution_problem = EvolutionProblem(hamiltonian, 2, Zero, param_value_dict=param_dict)
            with assert_raises(ValueError):
                evolution_problem.validate_params()

        with self.subTest(msg="Empty dict."):
            hamiltonian = param_x * X + param_y * Y
            param_dict = {}
            evolution_problem = EvolutionProblem(hamiltonian, 2, Zero, param_value_dict=param_dict)
            with assert_raises(ValueError):
                evolution_problem.validate_params()

        with self.subTest(msg="Extra parameter in dict."):
            hamiltonian = param_x * X + param_y * Y
            param_dict = {param_y: 2, param_x: 1, Parameter("z"): 1}
            evolution_problem = EvolutionProblem(hamiltonian, 2, Zero, param_value_dict=param_dict)
            with assert_raises(ValueError):
                evolution_problem.validate_params()


if __name__ == "__main__":
    unittest.main()
