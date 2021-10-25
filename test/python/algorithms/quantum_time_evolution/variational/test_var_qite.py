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

import unittest

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.var_qite import VarQite
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
)
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestVarQite(QiskitAlgorithmsTestCase):
    def test_run(self):
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

        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        reg = "ridge"

        var_qite = VarQite(var_principle, regularization=reg, backend=None, error_based_ode=False)
        time = 1

        evolution_result = var_qite.evolve(
            observable,
            time,
            # TODO design api assumes it is provided here, currently calculated within VarQte
            ansatz,  # ansatz is a state in this case
            hamiltonian_value_dict=param_dict,
        )

        print(evolution_result)

    # def test_compare(self):
    #     observable = SummedOp([(Z ^ X), 3.0 * (Y ^ Y), (Z ^ X), (I ^ Z), (Z ^ I)]).reduce()
    #
    #     d = 1
    #     ansatz = EfficientSU2(observable.num_qubits, reps=d)
    #
    #     parameters = ansatz.ordered_parameters
    #     init_param_values = np.zeros(len(ansatz.ordered_parameters))
    #     for i in range(ansatz.num_qubits):
    #         init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2
    #
    #     param_dict = dict(zip(parameters, init_param_values))
    #
    #     var_principle = ImaginaryMcLachlanVariationalPrinciple(
    #         grad_method="lin_comb", qfi_method="lin_comb_full"
    #     )
    #
    #     reg = None
    #
    #     var_qite = VarQite(
    #         var_principle,
    #         regularization=reg,
    #         backend=Aer.get_backend("statevector_simulator"),
    #         error_based_ode=False,
    #     )
    #     time = 1
    #
    #     evolution_result = var_qite.evolve(
    #         observable,
    #         time,
    #         initial_state=ansatz,  # ansatz is a state in this case
    #         hamiltonian_value_dict=param_dict,
    #     )
    #
    #     print(evolution_result)


if __name__ == "__main__":
    unittest.main()
