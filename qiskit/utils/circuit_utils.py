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

from qiskit.utils.entanglement import compute_ptrace
from qiskit.utils.entanglement import compute_vn_entropy


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


class Entanglement:

    """
    Class to measure the entanglement capacity of a parametric
    quantum circuit using two different measures:Meyer-Wallach Measure :
    Q-value = 1 => Maximum entangling capacity , Q-value = 0 => Minimum Entangling Capacity
    Von-Nuemann Measure   : Q-value = log_e(2) => Maximum entangling capacity ,
    Q-value = 0 => Minimum Entangling Capacity
    """

    def __init__(
        self,
        parametric_circuit,
        backend,
        ent_measure="meyer-wallach",
        feature_dim=4,
        num_params=2000,
    ) -> None:
        """
        Args:
            parametric_circuit : A parameterized circuit used input to
            calculate the entangling capacity.
            backend: The backend for running the circuit
            ent_measure: The type of entanglement measure;
                         ent_measure = "meyer-wallach" => Meyer-Wallach Measure
                         (Default Measure),
                         ent_measure = "von-neumann" => Von-Neumann Measure
            feature_dim: The total no. of feature of parametric_circuit i.e. the
                         no. of qubit in the parametric circuit; feature_dim = 4(Default Qubits)
            num_params: The total no. of parameter to sample over to get mean entangling capacity
                        over sampling again a sample size of num_params
                        num_params = 2000 (Default Value)
        """

        self.parametric_circuit = parametric_circuit
        self.num_params = num_params
        self.feature_dim = feature_dim
        self.backend = backend
        self.ent_measure = ent_measure

    def meyer_wallach(self) -> float:
        """
        Returns:
                net_entanglement_cap: The meyer-wallach entangling capacity of the
                                       given parametric circuit.
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import transpile

        entanglement_cap = []

        for _ in range(self.num_params):
            transpiled_circ = transpile(self.parametric_circuit, self.backend, optimization_level=3)
            job_sim = self.backend.run(transpiled_circ)
            result_sim = job_sim.result()

            state_vector = result_sim.get_statevector(transpiled_circ)
            state_vector = np.array(state_vector)

            q_value = compute_ptrace(ket=state_vector, num_qubits=self.feature_dim)
            entanglement_cap.append(q_value)

        entanglement_cap = np.array(entanglement_cap)
        net_entanglement_cap = np.sum(entanglement_cap) / self.num_params

        return net_entanglement_cap

    def von_neumann(self) -> float:

        """
        Returns:
              net_entanglement_cap: The von_neumann entangling capacity of the given parametric circuit.
        """

        # Runtime imports to avoid circular imports causeed by QuantumInstance
        # getting initialized by imported utils/__init__ which is imported
        # by qiskit.circuit
        from qiskit import transpile

        entanglement_cap = []

        for _ in range(self.num_params):

            transpiled_circ = transpile(self.parametric_circuit, self.backend, optimization_level=3)
            job_sim = self.backend.run(transpiled_circ)
            result_sim = job_sim.result()

            state_vector = result_sim.get_statevector(transpiled_circ)
            state_vector = np.array(state_vector)

            q_value = compute_vn_entropy(ket=state_vector, num_qubits=self.feature_dim)
            entanglement_cap.append(q_value)

        entanglement_cap = np.array(entanglement_cap)
        net_entanglement_cap = np.sum(entanglement_cap) / self.num_params

        return net_entanglement_cap

    def get_entanglement(self) -> float:
        """
        Returns:
            ent_cap: The entangling capacity of the given parametric circuit ;
                     Default: meyer-wallach measure
        """

        ent_cap = {
            "meyer-wallach": self.meyer_wallach(),
            "von-neumann": self.von_neumann(),
        }

        return ent_cap[self.ent_measure]
