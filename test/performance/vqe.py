# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
from qiskit import QISKitError

# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ryrz, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Hamiltonian_from_file, make_Hamiltonian
from qiskit.tools.apps.optimization import eval_hamiltonian, group_paulis
from qiskit import get_backend


def cost_function(H, n_qubits, depth, entangler_map, shots, device, theta):
    trial_circuit = trial_circuit_ryrz(n_qubits, depth, theta, entangler_map,
                                       meas_string=None, measurement=False)

    energy, circuits = eval_hamiltonian(H, trial_circuit, shots, device)

    return energy.real, circuits


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
    device = 'qasm_simulator'
    if shots == 1:
        device = 'statevector_simulator'

    if 'statevector' not in device:
        H = group_paulis(pauli_list)

    entangler_map = getattr(get_backend(device).configuration(),
                            'coupling_map', 'all-to-all')

    if entangler_map == 'all-to-all':
        entangler_map = {i: [j for j in range(n_qubits) if j != i] for i in range(n_qubits)}
    else:
        entangler_map = mapper.coupling_list2dict(entangler_map)

    initial_theta = np.random.randn(2 * n_qubits * depth)   # initial angles
    initial_c = 0.01                                        # first theta perturbations
    target_update = 2 * np.pi * 0.1                         # aimed update on first trial
    save_step = 20                                          # print optimization trajectory

    cost = partial(cost_function, H, n_qubits, depth, entangler_map, shots, device)

    SPSA_params, circuits_cal = SPSA_calibration(cost, initial_theta, initial_c,
                                                 target_update, stat=25)
    output, circuits_opt = SPSA_optimization(cost, initial_theta, SPSA_params, max_trials,
                                             save_step, last_avg=1)

    return circuits_cal + circuits_opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Performance testing for compiler, using the VQE application.")
    parser.add_argument('--molecule', default='H2', help='molecule to calculate')
    parser.add_argument('--depth', type=int, default=6, help='depth of trial circuit')
    parser.add_argument('--max_trials', type=int, default=200, help='how many trials')
    parser.add_argument('--shots', type=int, default=1, help='shots per circuit')
    args = parser.parse_args()

    tstart = time.time()
    circuits = vqe(args.molecule, args.depth, args.max_trials, args.shots)
    tend = time.time()

    avg = sum(c.qasm().count("\n") for c in circuits) / len(circuits)

    print("---- Number of circuits: {}".format(len(circuits)))
    print("---- Avg circuit size: {}".format(avg))
    print("---- Elapsed time: {}".format(tend - tstart))
