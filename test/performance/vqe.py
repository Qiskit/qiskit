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
import argparse
import time
import numpy as np
from scipy import linalg as la
from functools import partial

# import qiskit modules
from qiskit import mapper
from qiskit import QuantumProgram
from qiskit import QISKitError

# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ryrz, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Hamiltonian_from_file, make_Hamiltonian
from qiskit.tools.apps.optimization import eval_hamiltonian, group_paulis


def cost_function(qp, H, n_qubits, depth, entangler_map, shots, device, theta):
    trial_circuit = trial_circuit_ryrz(n_qubits, depth, theta, entangler_map,
                                       meas_string=None, measurement=False)

    energy = eval_hamiltonian(qp, H, trial_circuit, shots, device).real

    return energy


def vqe(molecule='H2', depth=6, max_trials=200, shots=1):
    if molecule == 'H2':
        n_qubits = 2
        Z1 = 1
        Z2 = 1
        min_distance = 0.2
        max_distance = 4

    elif molecule == 'LiH':
        n_qubits = 4
        Z1 = 1
        Z2 = 3
        min_distance = 0.5
        max_distance = 5

    else:
        raise QISKitError("Unknown molecule for VQE.")

    # Read Hamiltonian
    ham_name = os.path.join(os.path.dirname(__file__),
                            molecule + '/' + molecule + 'Equilibrium.txt')
    pauli_list = Hamiltonian_from_file(ham_name)
    H = make_Hamiltonian(pauli_list)

    # Exact Energy
    exact = np.amin(la.eig(H)[0]).real
    print('The exact ground state energy is: {}'.format(exact))

    # Optimization
    device = 'local_qiskit_simulator'
    qp = QuantumProgram()

    if shots != 1:
        H = group_paulis(pauli_list)

    entangler_map = qp.configuration(device)['coupling_map']

    if entangler_map == 'all-to-all':
        entangler_map = {i: [j for j in range(n_qubits) if j != i] for i in range(n_qubits)}
    else:
        entangler_map = mapper.coupling_list2dict(entangler_map)

    initial_theta = np.random.randn(2 * n_qubits * depth)   # initial angles
    initial_c = 0.01                                        # first theta perturbations
    target_update = 2 * np.pi * 0.1                         # aimed update on first trial
    save_step = 20                                          # print optimization trajectory

    cost = partial(cost_function, qp, H, n_qubits, depth, entangler_map, shots, device)

    SPSA_params = SPSA_calibration(cost, initial_theta, initial_c, target_update, stat=25)
    output = SPSA_optimization(cost, initial_theta, SPSA_params, max_trials, save_step, last_avg=1)

    return qp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Performance testing for compiler, using the VQE application.")
    parser.add_argument('--molecule', default='H2', help='molecule to calculate')
    parser.add_argument('--depth', type=int, default=6, help='depth of trial circuit')
    parser.add_argument('--max_trials', type=int, default=200, help='how many trials')
    parser.add_argument('--shots', type=int, default=1, help='shots per circuit')
    args = parser.parse_args()

    tstart = time.time()
    qp = vqe(args.molecule, args.depth, args.max_trials, args.shots)
    tend = time.time()

    all_circuits = list(qp.get_circuit_names())
    avg = sum(qp.get_qasm(c).count("\n") for c in all_circuits) / len(all_circuits)

    print("---- Number of circuits: {}".format(len(all_circuits)))
    print("---- Avg circuit size: {}".format(avg))
    print("---- Elapsed time: {}".format(tend - tstart))
