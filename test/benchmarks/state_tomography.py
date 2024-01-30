# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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
# pylint: disable=import-error

from qiskit_experiments.library import StateTomography

import qiskit


class StateTomographyBench:
    params = [2, 3, 4, 5]
    param_names = ["n_qubits"]
    version = "0.3.0"
    timeout = 120.0

    def setup(self, _):
        provider = qiskit.providers.basic_provider.BasicProvider()
        self.qasm_backend = provider.get_backend("basic_simulator")

    def time_state_tomography_bell(self, n_qubits):
        meas_qubits = [n_qubits - 2, n_qubits - 1]
        qr_full = qiskit.QuantumRegister(n_qubits)
        bell = qiskit.QuantumCircuit(qr_full)
        bell.h(qr_full[meas_qubits[0]])
        bell.cx(qr_full[meas_qubits[0]], qr_full[meas_qubits[1]])

        qst_exp = StateTomography(bell, measurement_qubits=meas_qubits)
        expdata = qst_exp.run(self.qasm_backend, shots=5000).block_for_results()
        expdata.analysis_results("state")
        expdata.analysis_results("state_fidelity")

    def time_state_tomography_cat(self, n_qubits):
        qr = qiskit.QuantumRegister(n_qubits, "qr")
        circ = qiskit.QuantumCircuit(qr, name="cat")
        circ.h(qr[0])
        for i in range(1, n_qubits):
            circ.cx(qr[0], qr[i])
        qst_exp = StateTomography(circ)
        expdata = qst_exp.run(self.qasm_backend, shots=5000).block_for_results()
        expdata.analysis_results("state")
        expdata.analysis_results("state_fidelity")
