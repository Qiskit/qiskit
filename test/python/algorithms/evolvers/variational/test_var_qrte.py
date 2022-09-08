# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Variational Quantum Real Time Evolution algorithm."""

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt
import numpy as np
from qiskit.test import slow_test
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import EvolutionProblem, VarQRTE
from qiskit.algorithms.evolvers.variational import (
    RealMcLachlanPrinciple,
)
from qiskit import BasicAer
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
    ExpectationFactory,
)


@ddt
class TestVarQRTE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Real Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)
        backend_statevector = BasicAer.get_backend("statevector_simulator")
        backend_qasm = BasicAer.get_backend("qasm_simulator")
        self.quantum_instance = QuantumInstance(
            backend=backend_statevector,
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.quantum_instance_qasm = QuantumInstance(
            backend=backend_qasm,
            shots=4000,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.backends_dict = {
            "qi_sv": self.quantum_instance,
            "qi_qasm": self.quantum_instance_qasm,
            "b_sv": backend_statevector,
        }

        self.backends_names = ["qi_qasm", "b_sv", "qi_sv"]

    @slow_test
    def test_run_d_1_with_aux_ops(self):
        """Test VarQRTE for d = 1 and t = 0.1 with evaluating auxiliary operators and the Forward
        Euler solver."""
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        )
        aux_ops = [X ^ X, Y ^ Z]
        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = RealMcLachlanPrinciple()

        time = 0.1

        evolution_problem = EvolutionProblem(observable, time, aux_operators=aux_ops)

        thetas_expected_sv = [
            0.88967020378258,
            1.53740751016451,
            1.57076759018861,
            1.58893301221363,
            1.60100970594142,
            1.57008242207638,
            1.63791241090936,
            1.53741371076912,
        ]

        thetas_expected_qasm = [
            0.88967811203145,
            1.53745130248168,
            1.57206794045495,
            1.58901347342829,
            1.60101431615503,
            1.57138020823337,
            1.63796000651177,
            1.53742227084076,
        ]

        expected_aux_ops_evaluated_sv = [(0.06675, 0.0), (0.772636, 0.0)]

        expected_aux_ops_evaluated_qasm = [
            (0.06450000000000006, 0.01577846435810532),
            (0.7895000000000001, 0.009704248425303218),
        ]

        for backend_name in self.backends_names:
            with self.subTest(msg=f"Test {backend_name} backend."):
                algorithm_globals.random_seed = self.seed
                backend = self.backends_dict[backend_name]
                expectation = ExpectationFactory.build(
                    operator=observable,
                    backend=backend,
                )
                var_qrte = VarQRTE(
                    ansatz,
                    var_principle,
                    init_param_values,
                    expectation=expectation,
                    num_timesteps=25,
                    quantum_instance=backend,
                )
                evolution_result = var_qrte.evolve(evolution_problem)

                evolved_state = evolution_result.evolved_state
                aux_ops = evolution_result.aux_ops_evaluated

                parameter_values = evolved_state.data[0][0].params
                if backend_name == "qi_qasm":
                    thetas_expected = thetas_expected_qasm
                    expected_aux_ops = expected_aux_ops_evaluated_qasm
                else:
                    thetas_expected = thetas_expected_sv
                    expected_aux_ops = expected_aux_ops_evaluated_sv

                for i, parameter_value in enumerate(parameter_values):
                    np.testing.assert_almost_equal(
                        float(parameter_value), thetas_expected[i], decimal=3
                    )
                np.testing.assert_array_almost_equal(aux_ops, expected_aux_ops)

    @slow_test
    @data(
        SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ),
        0.2252 * (I ^ I)
        + 0.5716 * (Z ^ Z)
        + 0.3435 * (I ^ Z)
        + -0.4347 * (Z ^ I)
        + 0.091 * (Y ^ Y)
        + 0.091 * (X ^ X),
    )
    def test_run_d_2(self, observable):
        """Test VarQRTE for d = 2 and t = 1 with RK45 ODE solver."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = RealMcLachlanPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        backend = BasicAer.get_backend("statevector_simulator")

        time = 1
        var_qrte = VarQRTE(
            ansatz,
            var_principle,
            param_dict,
            ode_solver="RK45",
            num_timesteps=25,
            quantum_instance=backend,
        )

        thetas_expected = [
            0.348407744196573,
            0.919404626262464,
            1.18189219371626,
            0.771011177789998,
            0.734384256533924,
            0.965289520781899,
            1.14441687204195,
            1.17231927568571,
            1.03014771379412,
            0.867266309056347,
            0.699606368428206,
            0.610788576398685,
        ]

        self._test_helper(observable, thetas_expected, time, var_qrte)

    def _test_helper(self, observable, thetas_expected, time, var_qrte):
        evolution_problem = EvolutionProblem(observable, time)
        evolution_result = var_qrte.evolve(evolution_problem)
        evolved_state = evolution_result.evolved_state

        parameter_values = evolved_state.data[0][0].params
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)


if __name__ == "__main__":
    unittest.main()
