# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring,invalid-name,no-member,broad-except
# pylint: disable=no-else-return, attribute-defined-outside-init

from qiskit.ignis.verification import tomography as tomo

import qiskit
from qiskit.quantum_info import state_fidelity


class StateTomographyBench:
    params = [2, 3, 4, 5]
    param_names = ['n_qubits']
    version = '0.1.1'
    timeout = 120.0

    def setup(self, _):
        self.sv_backend = qiskit.BasicAer.get_backend('statevector_simulator')
        self.qasm_backend = qiskit.BasicAer.get_backend('qasm_simulator')

    def time_state_tomography_bell(self, n_qubits):
        qr = qiskit.QuantumRegister(2)
        bell = qiskit.QuantumCircuit(qr)
        bell.h(qr[0])
        bell.cx(qr[0], qr[1])
        psi_bell = qiskit.execute(
            bell, self.sv_backend).result().get_statevector(bell)
        qr_full = qiskit.QuantumRegister(n_qubits)
        bell = qiskit.QuantumCircuit(qr_full)
        bell.h(qr_full[n_qubits - 2])
        bell.cx(qr_full[n_qubits - 2], qr_full[n_qubits - 1])
        qst_bell = tomo.state_tomography_circuits(bell,
                                                  [qr_full[n_qubits - 2],
                                                   qr_full[n_qubits - 1]])
        job = qiskit.execute(qst_bell, self.qasm_backend, shots=5000)
        rho_bell = tomo.StateTomographyFitter(job.result(), qst_bell).fit()
        state_fidelity(psi_bell, rho_bell)

    def time_state_tomography_cat(self, n_qubits):
        qr = qiskit.QuantumRegister(n_qubits, 'qr')
        circ = qiskit.QuantumCircuit(qr, name='cat')
        circ.h(qr[0])
        for i in range(1, n_qubits):
            circ.cx(qr[0], qr[i])
        psi = qiskit.execute(circ, self.sv_backend).result().get_statevector()
        qst_circ = tomo.state_tomography_circuits(circ, qr)
        tomo_result = qiskit.execute(
            qst_circ, self.qasm_backend, shots=5000).result()
        rho = tomo.StateTomographyFitter(tomo_result, qst_circ).fit()
        state_fidelity(psi, rho)
