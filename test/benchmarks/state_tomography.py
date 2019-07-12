# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,invalid-name,no-member,broad-except
# pylint: disable=no-else-return

"""
Quantum State Tomography.
Generates many small circuits, thus good for profiling compiler overhead.
Number of circuits grows like 3^n_qubits
"""

import numpy as np

import qiskit
# import tomography libary and other useful tools
try:
    import qiskit.tools.qcvv.tomography as tomo
    NO_TOMO = False
except ImportError:
    NO_TOMO = True
from qiskit.quantum_info import state_fidelity
try:
    from qiskit.tools.qi.qi import random_unitary_matrix, purity
except ImportError:
    NO_TOMO = True


class StateTomographyBench:
    params = [2, 3, 4, 5]
    timeout = 360.0
    use_quantum_program = False

    def setup(self, _):
        if hasattr(qiskit, 'QuantumProgram'):
            self.use_quantum_program = True
        else:
            self.use_quantum_program = False
        if NO_TOMO:
            raise NotImplementedError

    # circuit that outputs the target state
    def target_prep(self, state, target, n_qubits, qp=None):
        if not self.use_quantum_program:
            # quantum circuit to make an entangled cat state
            if state == 'cat':
                n_qubits = int(np.log2(target.size))
                qr = qiskit.QuantumRegister(n_qubits, 'qr')
                cr = qiskit.ClassicalRegister(n_qubits, 'cr')
                circ = qiskit.QuantumCircuit(qr, cr, name='cat')
                circ.h(qr[0])
                for i in range(1, n_qubits):
                    circ.cx(qr[0], qr[i])
            # quantum circuit to prepare arbitrary given state
            elif state == 'random':
                n_qubits = int(np.log2(target.size))
                qr = qiskit.QuantumRegister(n_qubits, 'qr')
                cr = qiskit.ClassicalRegister(n_qubits, 'cr')
                circ = qiskit.QuantumCircuit(qr, cr, name='random')
                circ.initialize(target, [qr[i] for i in range(n_qubits)])
            return circ
        else:
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
                try:
                    random.initialize(
                        "Qinit", target, [qr[i] for i in range(n_qubits)])
                except Exception:
                    random.initialize(
                        target, [qr[i] for i in range(n_qubits)])
            return qp

    # add basis measurements to the circuit for tomography
    # XX..X, XX..Y, .., ZZ..Z
    def add_tomo_circuits(self, circ):
        if not self.use_quantum_program:
            # Construct state tomography set for measurement of qubits in the
            # register
            qr = next(iter(circ.qregs))
            cr = next(iter(circ.cregs))
            tomo_set = tomo.state_tomography_set(list(range(qr.size)))
            # Add the state tomography measurement circuits
            tomo_circuits = tomo.create_tomography_circuits(
                circ, qr, cr, tomo_set)
            return tomo_set, tomo_circuits
        if self.use_quantum_program:
            # Construct state tomography set for measurement of qubits in the
            # register
            qr_name = list(circ.get_quantum_register_names())[0]
            cr_name = list(circ.get_classical_register_names())[0]
            qr = circ.get_quantum_register(qr_name)
            cr = circ.get_classical_register(cr_name)
            tomo_set = tomo.state_tomography_set(list(range(qr.size)))
            # Add the state tomography measurement circuits to the Quantum
            # Program
            tomo_circuits = tomo.create_tomography_circuits(
                circ, qr, cr, tomo_set)
            return circ, tomo_set, tomo_circuits
        raise Exception

    def time_state_tomography_cat(self, n_qubits):
        # cat target state: [1. 0. 0. ... 0. 0. 1.]/sqrt(2.)
        target = np.zeros(pow(2, n_qubits))
        target[0] = 1
        target[pow(2, n_qubits)-1] = 1.0
        target /= np.sqrt(2.0)
        if not self.use_quantum_program:
            self._state_tomography(target, 'cat', n_qubits)
        else:
            self._state_tomography_quantum_program(target, 'cat', n_qubits)

    def time_state_tomography_random(self, n_qubits):
        # random target state: first column of a random unitary
        target = random_unitary_matrix(pow(2, n_qubits))[0]
        if not self.use_quantum_program:
            self._state_tomography(target, 'random', n_qubits)
        else:
            self._state_tomography_quantum_program(target, 'random', n_qubits)

    def _state_tomography_quantum_program(self, target, state, n_qubits,
                                          shots=1):
        qp = qiskit.QuantumProgram()
        try:
            backend = 'local_qiskit_simulator'
            qp.get_backend_configuration(backend)
        except LookupError:
            backend = 'local_qasm_simulator'

        # Prepared target state and assess quality
        qp = self.target_prep(state, target, n_qubits, qp=qp)
        prep_result = qp.execute(['prep'],
                                 backend=backend, shots=1)
        prep_state = prep_result.get_data('prep')['quantum_state']
        F_prep = state_fidelity(prep_state, target)
        print('Prepared state fidelity =', F_prep)

        # Run state tomography simulation and fit data to reconstruct circuit
        qp, tomo_set, tomo_circuits = self.add_tomo_circuits(qp)
        tomo_result = qp.execute(tomo_circuits, backend=backend, shots=shots)
        tomo_data = tomo.tomography_data(tomo_result, 'prep', tomo_set)
        rho_fit = tomo.fit_tomography_data(tomo_data)

        # calculate fidelity and purity of fitted state
        F_fit = state_fidelity(rho_fit, target)
        pur = purity(rho_fit)
        print('Fitted state fidelity =', F_fit)
        print('Fitted state purity =', str(pur))

    # perform quantum state tomography and assess quality of reconstructed
    # vector
    def _state_tomography(self, target, state, n_qubits, shots=1):
        # Use the local qasm simulator
        backend = qiskit.BasicAer.get_backend('statevector_simulator')

        # Prepared target state and assess quality
        prep_circ = self.target_prep(state, target, n_qubits)
        prep_result = qiskit.execute(
            prep_circ, backend=backend).result()
        prep_state = prep_result.get_statevector(prep_circ)
        F_prep = state_fidelity(prep_state, target)
        print('Prepared state fidelity =', F_prep)

        # Run state tomography simulation and fit data to reconstruct circuit
        tomo_set, tomo_circuits = self.add_tomo_circuits(prep_circ)
        tomo_result = qiskit.execute(
            tomo_circuits, backend=backend, shots=shots).result()
        tomo_data = tomo.tomography_data(tomo_result, prep_circ.name, tomo_set)
        rho_fit = tomo.fit_tomography_data(tomo_data)

        # calculate fidelity and purity of fitted state
        F_fit = state_fidelity(rho_fit, target)
        pur = purity(rho_fit)
        print('Fitted state fidelity =', F_fit)
        print('Fitted state purity =', str(pur))
