# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum State Tomography.
Generates many small circuits, thus good for profiling compiler overhead.
Number of circuits grows like 3^n_qubits
"""

import sys
import numpy as np
import argparse
import time

# import qiskit modules
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import QiskitError, execute

# import tomography libary and other useful tools
import qiskit.tools.qcvv.tomography as tomo
from qiskit.tools.qi.qi import purity, outer, random_unitary_matrix
from qiskit.quantum_info import state_fidelity


# circuit that outputs the target state
def target_prep(state, target):
    # quantum circuit to make an entangled cat state
    if state == 'cat':
        n_qubits = int(np.log2(target.size))
        qr = QuantumRegister(n_qubits, 'qr')
        cr = ClassicalRegister(n_qubits, 'cr')
        circ = QuantumCircuit(qr, cr, name='cat')
        circ.h(qr[0])
        for i in range(1, n_qubits):
            circ.cx(qr[0], qr[i])

    # quantum circuit to prepare arbitrary given state
    elif state == 'random':
        n_qubits = int(np.log2(target.size))
        qr = QuantumRegister(n_qubits, 'qr')
        cr = ClassicalRegister(n_qubits, 'cr')
        circ = QuantumCircuit(qr, cr, name='random')
        circ.initialize("Qinit", target, [qr[i] for i in range(n_qubits)])

    return circ


# add basis measurements to the circuit for tomography
# XX..X, XX..Y, .., ZZ..Z
def add_tomo_circuits(circ):
    # Construct state tomography set for measurement of qubits in the register
    qr = next(iter(circ.get_qregs().values()))
    cr = next(iter(circ.get_cregs().values()))
    tomo_set = tomo.state_tomography_set(list(range(qr.size)))

    # Add the state tomography measurement circuits
    tomo_circuits = tomo.create_tomography_circuits(circ, qr, cr, tomo_set)

    return tomo_set, tomo_circuits


# perform quantum state tomography and assess quality of reconstructed vector
def state_tomography(state, n_qubits, shots):
    # cat target state: [1. 0. 0. ... 0. 0. 1.]/sqrt(2.)
    if state == 'cat':
        target = np.zeros(pow(2, n_qubits))
        target[0] = 1
        target[pow(2, n_qubits)-1] = 1.0
        target /= np.sqrt(2.0)
    # random target state: first column of a random unitary
    elif state == 'random':
        target = random_unitary_matrix(pow(2, n_qubits))[0]
    else:
        raise QiskitError("Unknown state for tomography.")

    print("target: {}".format(target))

    # Use the local qasm simulator
    backend = 'qasm_simulator'

    # Prepared target state and assess quality
    prep_circ = target_prep(state, target)
    prep_result = execute(prep_circ, backend='statevector_simulator').result()
    prep_state = prep_result.get_statevector(prep_circ)
    F_prep = state_fidelity(prep_state, target)
    print('Prepared state fidelity =', F_prep)

    # Run state tomography simulation and fit data to reconstruct circuit
    tomo_set, tomo_circuits = add_tomo_circuits(prep_circ)
    tomo_result = execute(tomo_circuits, backend=backend, shots=shots).result()
    tomo_data = tomo.tomography_data(tomo_result, prep_circ.name, tomo_set)
    rho_fit = tomo.fit_tomography_data(tomo_data)

    # calculate fidelity and purity of fitted state
    F_fit = state_fidelity(rho_fit, target)
    pur = purity(rho_fit)
    print('Fitted state fidelity =', F_fit)
    print('Fitted state purity =', str(pur))

    return tomo_circuits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Performance testing for compiler, using state tomography.")
    parser.add_argument('--state', default='cat', help='state for tomography')
    parser.add_argument('--n_qubits', type=int, default=5, help='num qubits')
    parser.add_argument('--shots', type=int, default=1024, help='shots per measurement basis')
    args = parser.parse_args()

    tstart = time.time()
    tomo_circuits = state_tomography(args.state, args.n_qubits, args.shots)
    tend = time.time()

    avg = sum(c.qasm().count("\n") for c in tomo_circuits) / len(tomo_circuits)

    print("---- Number of circuits: {}".format(len(tomo_circuits)))
    print("---- Avg circuit size: {}".format(avg))
    print("---- Elapsed time: {}".format(tend - tstart))
