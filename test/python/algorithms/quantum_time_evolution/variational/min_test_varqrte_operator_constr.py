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

from qiskit.algorithms.quantum_time_evolution.variational.principles.real.implementations\
    .real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
)
from qiskit import Aer
from qiskit.opflow.gradients.circuit_gradients import LinComb
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
    ListOp,
    StateFn,
    PauliExpectation,
)
from test.python.algorithms import QiskitAlgorithmsTestCase

np.random.seed = 11
from qiskit.utils import algorithm_globals
algorithm_globals.random_seed = 11

hamiltonian = SummedOp(
    [
        0.2252 * (I ^ I),
        0.5716 * (Z ^ Z),
        0.3435 * (I ^ Z),
        -0.4347 * (Z ^ I),
        0.091 * (Y ^ Y),
        0.091 * (X ^ X),
    ]
).reduce()

d = 1
ansatz = EfficientSU2(hamiltonian.num_qubits, reps=d)

exp = PauliExpectation().convert(~StateFn(hamiltonian) @ StateFn(ansatz))

hamiltonian_ = SummedOp([hamiltonian, ListOp([I ^ hamiltonian.num_qubits, exp], combo_fn=lambda x: \
                x[0] * x[1])])

parameters = ansatz.ordered_parameters
init_param_values = np.zeros(len(ansatz.ordered_parameters))
for i in range(len(ansatz.ordered_parameters)):
    init_param_values[i] = np.pi / 2
init_param_values[0] = 1



param_dict = dict(zip(parameters, init_param_values))

exp_val = ~StateFn(hamiltonian_) @ StateFn(ansatz)

grad = LinComb().convert(exp_val, parameters)

grad = grad.assign_parameters(param_dict)
grad = grad.eval()