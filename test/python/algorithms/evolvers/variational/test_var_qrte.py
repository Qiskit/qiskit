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
import numpy as np
from qiskit.algorithms.evolvers.variational.solvers.ode.ode_function_factory import (
    OdeFunctionFactory,
    OdeFunctionType,
)
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.algorithms import EvolutionProblem
from qiskit.algorithms.evolvers.variational import VarQRTE
from qiskit.algorithms.evolvers.variational.variational_principles.real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
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

    def test_run_d_1_with_aux_ops(self):
        """Test VarQRTE for d = 1 and t = 0.1 with evaluating auxiliary operators."""
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()
        aux_ops = [X ^ X, Y ^ Z]
        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = RealMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        ode_function = OdeFunctionFactory(OdeFunctionType.STANDARD_ODE)
        time = 0.1

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, param_value_dict=param_dict, aux_operators=aux_ops
        )

        thetas_expected_sv = [
            0.886892332536579,
            1.53841937804971,
            1.57100495332575,
            1.58891646585018,
            1.59947862625132,
            1.57016298432375,
            1.63950734048655,
            1.53843155700668,
        ]

        thetas_expected_qasm = [
            1.0,
            1.5707963267949,
            1.5707963267949,
            1.5707963267949,
            1.5707963267949,
            1.5707963267949,
            1.5707963267949,
            1.5707963267949,
        ]

        expected_aux_ops_evaluated_sv = [(0.06836956278706763, 0.0), (0.7711878222922857, 0.0)]

        expected_aux_ops_evaluated_qasm = [
            (-0.008500000000000008, 0.0158108171041221),
            (0.858, 0.008121514637061244),
        ]

        for backend_name in self.backends_names:
            with self.subTest(msg=f"Test {backend_name} backend."):
                algorithm_globals.random_seed = self.seed
                backend = self.backends_dict[backend_name]
                expectation = ExpectationFactory.build(
                    operator=observable,
                    backend=backend,
                )
                var_qite = VarQRTE(
                    var_principle, ode_function, quantum_instance=backend, expectation=expectation
                )
                evolution_result = var_qite.evolve(evolution_problem)

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

    def test_run_d_2(self):
        """Test VarQRTE for d = 2 and t = 1."""
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = RealMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        backend = BasicAer.get_backend("statevector_simulator")

        ode_function = OdeFunctionFactory(OdeFunctionType.STANDARD_ODE)
        var_qrte = VarQRTE(var_principle, ode_function, quantum_instance=backend)
        time = 1

        evolution_problem = EvolutionProblem(observable, time, ansatz, param_value_dict=param_dict)

        evolution_result = var_qrte.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

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
        # TODO remove print before merging
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)


if __name__ == "__main__":
    unittest.main()
