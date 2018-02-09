# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Variational Quantum Eigensolver (VQE).
Generates many small circuits, thus good for profiling compiler overhead.
"""

import sys
import os
import numpy as np
from scipy import linalg as la
from functools import partial

# import qiskit modules
from qiskit import QuantumProgram

# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ryrz, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Hamiltonian_from_file, make_Hamiltonian
from qiskit.tools.apps.optimization import eval_hamiltonian, group_paulis

# Ignore warnings due to chopping of small imaginary part of the energy
import warnings
warnings.filterwarnings('ignore')

n_qubits = 2    # size of molecule
depth = 6       # single_qubit_gate_layers - entangler_layers + 1
device = 'local_qiskit_simulator' 

initial_theta = np.random.randn(2 * n_qubits * depth)
entangler_map = {1: [0]}    # map of two-qubit gates (key: control, values: target)
shots = 1
max_trials = 200
ham_name = os.path.join(os.path.dirname(__file__), 'H2/H2Equilibrium.txt')

# Exact Energy
pauli_list = Hamiltonian_from_file(ham_name)
pauli_list_grouped = group_paulis(pauli_list)
H = make_Hamiltonian(pauli_list)
exact = np.amin(la.eig(H)[0]).real
print('The exact ground state energy is: {}'.format(exact))

# Optimization
qp = QuantumProgram()

def cost_function(qp, H, n_qubits, depth, entangler_map, shots, device, theta):
    return eval_hamiltonian(qp, H,
                            trial_circuit_ryrz(n_qubits, depth, theta, entangler_map,
                                               None, False), shots, device).real


def optimize():
    initial_c = 0.01
    target_update = 2 * np.pi * 0.1
    save_step = 20

    if shots == 1:
        SPSA_params = SPSA_calibration(partial(cost_function,
                                               qp,
                                               H,
                                               n_qubits,
                                               depth,
                                               entangler_map, 
                                               shots, 
                                               device),
                                       initial_theta, 
                                       initial_c, 
                                       target_update, 
                                       25)
        output = SPSA_optimization(partial(cost_function,
                                           qp,
                                           H,
                                           n_qubits,
                                           depth,
                                           entangler_map,
                                           shots,
                                           device),
                                   initial_theta,
                                   SPSA_params,
                                   max_trials,
                                   save_step,
                                   1)

    else:
        SPSA_params = SPSA_calibration(partial(cost_function,
                                               qp,
                                               pauli_list_grouped,
                                               n_qubits,
                                               depth,
                                               entangler_map,
                                               shots,
                                               device),
                                       initial_theta,
                                       initial_c,
                                       target_update,
                                       25)
        output = SPSA_optimization(partial(cost_function,
                                           qp,
                                           pauli_list_grouped,
                                           n_qubits,
                                           depth,
                                           entangler_map,
                                           shots,
                                           device),
                                   initial_theta,
                                   SPSA_params,
                                   max_trials,
                                   save_step,
                                   1)


optimize()
