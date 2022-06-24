# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Circuit utility functions """

import numpy as np

from qiskit.utils.entanglement import compute_Q_ptrace_qiskit
from qiskit.utils.entanglement import compute_vn_entropy_qiskit


def summarize_circuits(circuits):
    """Summarize circuits based on QuantumCircuit, and five metrics are summarized.
        - Number of qubits
        - Number of classical bits
        - Number of operations
        - Depth of circuits
        - Counts of different gate operations

    The average statistic of the first four is provided if multiple circuits are provided.

    Args:
        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits

    Returns:
        str: a formatted string records the summary
    """
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ""
    ret += f"Submitting {len(circuits)} circuits.\n"
    ret += "============================================================================\n"
    stats = np.zeros(4)
    for i, circuit in enumerate(circuits):
        depth = circuit.depth()
        size = circuit.size()
        num_qubits = sum(reg.size for reg in circuit.qregs)
        num_clbits = sum(reg.size for reg in circuit.cregs)
        op_counts = circuit.count_ops()
        stats[0] += num_qubits
        stats[1] += num_clbits
        stats[2] += size
        stats[3] += depth
        ret = "".join(
            [
                ret,
                "{}-th circuit: {} qubits, {} classical bits and {} "
                "operations with depth {}\nop_counts: {}\n".format(
                    i, num_qubits, num_clbits, size, depth, op_counts
                ),
            ]
        )
    if len(circuits) > 1:
        stats /= len(circuits)
        ret = "".join(
            [
                ret,
                "Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} "
                "operations with depth {:.2f}\n".format(stats[0], stats[1], stats[2], stats[3]),
            ]
        )
    ret += "============================================================================\n"
    return ret


class entanglement:
    def __init__(self, parametric_circuit, backend, ent_measure=1, feature_dim=4, num_params=2000):

        self.parametric_circuit = parametric_circuit
        self.num_params = num_params
        self.feature_dim = feature_dim
        self.backend = backend
        self.ent_measure = ent_measure

    def entanglement_capibility_qiskit(self):

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import transpile

        entanglement_cap = []

        for samples in range(self.num_params):
            transpiled_circ = transpile(self.parametric_circuit, self.backend, optimization_level=3)
            job_sim = self.backend.run(transpiled_circ)
            result_sim = job_sim.result()

            state_vector = result_sim.get_statevector(transpiled_circ)
            state_vector = np.array(state_vector)

            Q_value = compute_Q_ptrace_qiskit(ket=state_vector, N=self.feature_dim)
            entanglement_cap.append(Q_value)

        entanglement_cap = np.array(entanglement_cap)
        net_entanglement_cap = np.sum(entanglement_cap) / self.num_params

        return net_entanglement_cap

    def von_neumann_entanglement_qiskit(self):

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import transpile

        entanglement_cap = []

        for samples in range(self.num_params):

            transpiled_circ = transpile(self.parametric_circuit, self.backend, optimization_level=3)
            job_sim = self.backend.run(transpiled_circ)
            result_sim = job_sim.result()

            state_vector = result_sim.get_statevector(transpiled_circ)
            state_vector = np.array(state_vector)

            Q_value = compute_vn_entropy_qiskit(ket=state_vector, N=self.feature_dim)
            entanglement_cap.append(Q_value)

        entanglement_cap = np.array(entanglement_cap)
        net_entanglement_cap = np.sum(entanglement_cap) / self.num_params

        return net_entanglement_cap

    def get_entanglement(self):

        ent_cap = {
            1: self.entanglement_capibility_qiskit(),
            2: self.von_neumann_entanglement_qiskit(),
        }

        return ent_cap[self.ent_measure]
