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

"""Test Variational Quantum Imaginary Time Evolution algorithm."""

import unittest

from ddt import data, ddt

from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.algorithms.evolvers.variational.solvers.ode.ode_function_factory import (
    OdeFunctionFactory,
    OdeFunctionType,
)
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit import BasicAer
from qiskit.algorithms import EvolutionProblem
from qiskit.algorithms.evolvers.variational import VarQITE
from qiskit.algorithms.evolvers.variational.variational_principles.imaginary_mc_lachlan_principle import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
    ExpectationFactory,
)
from qiskit.quantum_info import state_fidelity, Statevector


@ddt
class TestVarQITE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Imaginary Time Evolution algorithm."""

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
        """Test VarQITE for d = 1 and t = 1 with evaluating auxiliary operator."""
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
        var_principle = ImaginaryMcLachlanPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        ode_function = OdeFunctionFactory(OdeFunctionType.STANDARD_ODE)

        time = 1

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, aux_operators=aux_ops, param_value_dict=param_dict
        )

        # values from the prototype
        thetas_expected_sv = [
            0.905901128153194,
            2.00336012271211,
            2.69398302961789,
            2.74463415808318,
            2.33136908953802,
            1.7469421463254,
            2.10770301585949,
            1.92775891861825,
        ]

        thetas_expected_qasm = [
            0.92928079575943,
            1.70504715213865,
            2.6955702603652,
            2.70730190763444,
            2.29297852544896,
            1.51004167136585,
            2.200898726865,
            1.97021569662524,
        ]

        expected_aux_ops_evaluated_sv = [(-0.204155479846185, 0.0), (0.25191789852257596, 0.0)]
        expected_aux_ops_evaluated_qasm = [
            (0.011999999999999927, 0.015810249839898167),
            (0.32800000000000024, 0.014936666294725873),
        ]

        for backend_name in self.backends_names:
            with self.subTest(msg=f"Test {backend_name} backend."):
                algorithm_globals.random_seed = self.seed
                backend = self.backends_dict[backend_name]
                expectation = ExpectationFactory.build(
                    operator=observable,
                    backend=backend,
                )
                var_qite = VarQITE(
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

    def test_run_d_1_t_7(self):
        """Test VarQITE for d = 1 and t = 7."""
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

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = ImaginaryMcLachlanPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        backend = BasicAer.get_backend("statevector_simulator")

        ode_function = OdeFunctionFactory(OdeFunctionType.STANDARD_ODE)
        var_qite = VarQITE(var_principle, ode_function, quantum_instance=backend)
        time = 7

        evolution_problem = EvolutionProblem(observable, time, ansatz, param_value_dict=param_dict)

        evolution_result = var_qite.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

        # values from the prototype
        thetas_expected = [
            0.828917365718767,
            1.88481074798033,
            3.14111335991238,
            3.14125849601269,
            2.33768562678401,
            1.78670990729437,
            2.04214275514208,
            2.04009918594422,
        ]
        # TODO remove print before merging
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(list(ansatz.parameters), thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=2)

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
        """Test VarQITE for d = 2 and t = 1."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = ImaginaryMcLachlanPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        backend = BasicAer.get_backend("statevector_simulator")

        ode_function = OdeFunctionFactory(OdeFunctionType.STANDARD_ODE)
        var_qite = VarQITE(var_principle, ode_function, quantum_instance=backend)
        time = 1

        evolution_problem = EvolutionProblem(observable, time, ansatz, param_value_dict=param_dict)

        evolution_result = var_qite.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

        # values from the prototype
        thetas_expected = [
            1.29495364023786,
            1.08970061333559,
            0.667488228710748,
            0.500122687902944,
            1.4377736672043,
            1.22881086103085,
            0.729773048146251,
            1.01698854755226,
            0.050807780587492,
            0.294828474947149,
            0.839305697704923,
            0.663689581255428,
        ]
        # TODO remove print before merging
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(list(ansatz.parameters), thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)


if __name__ == "__main__":
    unittest.main()
