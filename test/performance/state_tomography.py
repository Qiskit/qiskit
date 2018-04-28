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
Quantum State Tomography.
Generates many small circuits, thus good for profiling compiler overhead.
Number of circuits grows like 3^n_qubits
"""

import sys
import numpy as np
import argparse
import time

# import qiskit modules
from qiskit import QuantumProgram
from qiskit import QISKitError

# import tomography libary and other useful tools
import qiskit.tools.qcvv.tomography as tomo
from qiskit.tools.qi.qi import state_fidelity, purity
from qiskit.tools.qi.qi import outer, random_unitary_matrix


# circuit that outputs the target state
def target_prep(qp, state, target):
    # quantum circuit to make an entangled cat state
    if state == 'cat':
        n_qubits = int(np.log2(target.size))
        qr = qp.create_quantum_register('qr', n_qubits)
        cr = qp.create_classical_register('cr', n_qubits)
        cat = qp.create_circuit('prep', [qr], [cr])
        cat.h(qr[0])
        for i in range(1, n_qubits):
            cat.cx(qr[0], qr[i])

    # quantum circuit to prepare arbitrary given state
    elif state == 'random':
        n_qubits = int(np.log2(target.size))
        qr = qp.create_quantum_register('qr', n_qubits)
        cr = qp.create_classical_register('cr', n_qubits)
        random = qp.create_circuit('prep', [qr], [cr])
        random.initialize("Qinit", target, [qr[i] for i in range(n_qubits)])

    return qp


# add basis measurements to the Quantum Program for tomography
# XX..X, XX..Y, .., ZZ..Z
def add_tomo_circuits(qp):
    # Construct state tomography set for measurement of qubits in the register
    qr_name = list(qp.get_quantum_register_names())[0]
    cr_name = list(qp.get_classical_register_names())[0]
    qr = qp.get_quantum_register(qr_name)
    cr = qp.get_classical_register(cr_name)
    tomo_set = tomo.state_tomography_set(list(range(qr.size)))

    # Add the state tomography measurement circuits to the Quantum Program
    tomo_circuits = tomo.create_tomography_circuits(qp, 'prep', qr, cr, tomo_set)

    return qp, tomo_set, tomo_circuits


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
        raise QISKitError("Unknown state for tomography.")

    print("target: {}".format(target))

    # Use the local qasm simulator
    backend = 'local_qasm_simulator'

    qp = QuantumProgram()

    # Prepared target state and assess quality
    qp = target_prep(qp, state, target)
    prep_result = qp.execute(['prep'], backend=backend, shots=1)
    prep_state = prep_result.get_data('prep')['statevector']
    F_prep = state_fidelity(prep_state, target)
    print('Prepared state fidelity =', F_prep)

    # Run state tomography simulation and fit data to reconstruct circuit
    qp, tomo_set, tomo_circuits = add_tomo_circuits(qp)
    tomo_result = qp.execute(tomo_circuits, backend=backend, shots=shots)
    tomo_data = tomo.tomography_data(tomo_result, 'prep', tomo_set)
    rho_fit = tomo.fit_tomography_data(tomo_data)

    # calculate fidelity and purity of fitted state
    F_fit = state_fidelity(rho_fit, target)
    pur = purity(rho_fit)
    print('Fitted state fidelity =', F_fit)
    print('Fitted state purity =', str(pur))

    return qp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Performance testing for compiler, using state tomography.")
    parser.add_argument('--state', default='cat', help='state for tomography')
    parser.add_argument('--n_qubits', type=int, default=5, help='num qubits')
    parser.add_argument('--shots', type=int, default=1024, help='shots per measurement basis')
    args = parser.parse_args()

    tstart = time.time()
    qp = state_tomography(args.state, args.n_qubits, args.shots)
    tend = time.time()

    all_circuits = list(qp.get_circuit_names())
    avg = sum(qp.get_qasm(c).count("\n") for c in all_circuits) / len(all_circuits)

    print("---- Number of circuits: {}".format(len(all_circuits)))
    print("---- Avg circuit size: {}".format(avg))
    print("---- Elapsed time: {}".format(tend - tstart))
