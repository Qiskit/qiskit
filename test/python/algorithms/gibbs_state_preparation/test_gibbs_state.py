# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests GibbsState class."""
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data
import numpy as np

from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.opflow import Zero, X, H, SummedOp, Z, I


@ddt
class TestGibbsState(QiskitAlgorithmsTestCase):
    """Tests GibbsState class."""

    def test_gibbs_state_init(self):
        """Initialization test."""
        gibbs_state_function = Zero
        hamiltonian = X
        temperature = 42

        gibbs_state = GibbsState(gibbs_state_function, hamiltonian, temperature)

        np.testing.assert_equal(gibbs_state.gibbs_state_function, Zero)
        np.testing.assert_equal(gibbs_state.hamiltonian, X)
        np.testing.assert_equal(gibbs_state.temperature, 42)

    def test_calc_ansatz_gradients(self):
        """Tests if ansatz gradients are calculated correctly."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42

        depth = 1
        entangler_map = [[i + 1, i] for i in range(hamiltonian.num_qubits - 1)]
        ansatz = EfficientSU2(4, reps=depth, entanglement=entangler_map)
        qr = ansatz.qregs[0]
        for i in range(int(len(qr) / 2)):
            ansatz.cx(qr[i], qr[i + int(len(qr) / 2)])

        # Initialize the Ansatz parameters
        param_values_init = np.zeros(2 * hamiltonian.num_qubits * (depth + 1))
        for j in range(2 * H.num_qubits * depth, int(len(param_values_init) - H.num_qubits - 2)):
            param_values_init[int(j)] = np.pi / 2.0

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsState(
            gibbs_state_function, hamiltonian, temperature, ansatz, params_dict
        )

        gradient_params = list(ansatz.parameters)
        measurement_op = X
        gradient_method = "param_shift"
        gradients = gibbs_state.calc_ansatz_gradients(
            gradient_params, measurement_op, gradient_method
        )
        expected_gradients = [
            (-3.0000000000000007e-17 + 0j),
            (-2.05e-17 + 0j),
            (-4.199999999999999e-17 + 0j),
            (-0.9999999999999998 + 0j),
            0j,
            0j,
            (1.5250000000000001e-16 + 0j),
            (-1.0999999999999994e-17 + 0j),
            (-7.000000000000003e-18 + 0j),
            (-2.4999999999999913e-18 + 0j),
            0j,
            (5.4999999999999994e-17 + 0j),
            (-2.2499999999999996e-17 + 0j),
            (-3.5e-17 + 0j),
            (-1.500000000000001e-17 + 0j),
            (-6.95e-17 + 0j),
        ]

        np.testing.assert_almost_equal(gradients, expected_gradients)

    def test_calc_ansatz_gradients_missing_ansatz(self):
        """Tests if an expected error is raised when an ansatz is missing when calculating
        ansatz gradients."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42
        param_values_init = np.zeros(2)

        params_dict = dict(zip([Parameter("a"), Parameter("b")], param_values_init))
        gibbs_state = GibbsState(
            gibbs_state_function, hamiltonian, temperature, ansatz_params_dict=params_dict
        )

        gradient_params = [Parameter("a"), Parameter("b")]
        measurement_op = X
        gradient_method = "param_shift"
        np.testing.assert_raises(
            ValueError,
            gibbs_state.calc_ansatz_gradients,
            gradient_params,
            measurement_op,
            gradient_method,
        )

    @data(None, [Parameter("w")])
    def test_calc_ansatz_gradients_invalid_gradient_params(self, gradient_params):
        """Tests if expected errors are raised when gradient_params is invalid in ansatz
        gradients."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42

        depth = 1
        entangler_map = [[i + 1, i] for i in range(hamiltonian.num_qubits - 1)]
        ansatz = EfficientSU2(4, reps=depth, entanglement=entangler_map)
        qr = ansatz.qregs[0]
        for i in range(int(len(qr) / 2)):
            ansatz.cx(qr[i], qr[i + int(len(qr) / 2)])

        # Initialize the Ansatz parameters
        param_values_init = np.zeros(2 * hamiltonian.num_qubits * (depth + 1))
        for j in range(2 * H.num_qubits * depth, int(len(param_values_init) - H.num_qubits - 2)):
            param_values_init[int(j)] = np.pi / 2.0

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsState(
            gibbs_state_function, hamiltonian, temperature, ansatz, params_dict
        )

        measurement_op = X
        gradient_method = "param_shift"
        np.testing.assert_raises(
            ValueError,
            gibbs_state.calc_ansatz_gradients,
            gradient_params,
            measurement_op,
            gradient_method,
        )

    def test_calc_hamiltonian_gradients(self):
        """Tests if hamiltonian gradients are calculated correctly."""
        gibbs_state_function = Zero
        param = Parameter("w")
        hamiltonian = SummedOp([param * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42

        depth = 1
        entangler_map = [[i + 1, i] for i in range(hamiltonian.num_qubits - 1)]
        ansatz = EfficientSU2(4, reps=depth, entanglement=entangler_map)
        qr = ansatz.qregs[0]
        for i in range(int(len(qr) / 2)):
            ansatz.cx(qr[i], qr[i + int(len(qr) / 2)])

        # Initialize the Ansatz parameters
        param_values_init = np.zeros(2 * hamiltonian.num_qubits * (depth + 1))
        for j in range(2 * H.num_qubits * depth, int(len(param_values_init) - H.num_qubits - 2)):
            param_values_init[int(j)] = np.pi / 2.0

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))

        hamiltonian_gradients = {param: [0.1] * len(ansatz.parameters)}  # TODO np array
        gibbs_state = GibbsState(
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz,
            params_dict,
            hamiltonian_gradients,
        )

        gradient_params = list(ansatz.parameters)
        measurement_op = X
        gradient_method = "param_shift"
        final_gradients = gibbs_state.calc_hamiltonian_gradients(
            gradient_params, measurement_op, gradient_method
        )

        expected_gradients = {param: (-0.09999999999999999 + 0j)}

        np.testing.assert_equal(final_gradients, expected_gradients)


if __name__ == "__main__":
    unittest.main()
