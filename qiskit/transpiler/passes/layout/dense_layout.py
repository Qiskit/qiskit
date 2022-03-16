# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Choose a Layout by finding the most connected subset of qubits."""


import numpy as np
import retworkx

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit._accelerate.dense_layout import best_subset  # pylint: disable=import-error


class DenseLayout(AnalysisPass):
    """Choose a Layout by finding the most connected subset of qubits.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit).

    Note:
        Even though a 'layout' is not strictly a property of the DAG,
        in the transpiler architecture it is best passed around between passes
        by being set in `property_set`.
    """

    def __init__(self, coupling_map, backend_prop=None):
        """DenseLayout initializer.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.
            backend_prop (BackendProperties): backend properties object
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.backend_prop = backend_prop
        num_qubits = 0
        self.adjacency_matrix = None
        if self.coupling_map:
            num_qubits = self.coupling_map.size()
            self.adjacency_matrix = retworkx.adjacency_matrix(self.coupling_map.graph)
        self.error_mat = np.zeros((num_qubits, num_qubits))
        if self.backend_prop and self.coupling_map:
            error_dict = {
                tuple(gate.qubits): gate.parameters[0].value
                for gate in self.backend_prop.gates
                if len(gate.qubits) == 2
            }
            for edge in self.coupling_map.get_edges():
                gate_error = error_dict.get(edge)
                if gate_error is not None:
                    self.error_mat[edge[0]][edge[1]] = gate_error
            for index, qubit_data in enumerate(self.backend_prop.qubits):
                # Handle faulty qubits edge case
                if index >= num_qubits:
                    break
                for item in qubit_data:
                    if item.name == "readout_error":
                        self.error_mat[index][index] = item.value

    def run(self, dag):
        """Run the DenseLayout pass on `dag`.

        Pick a convenient layout depending on the best matching
        qubit connectivity, and set the property `layout`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        num_dag_qubits = len(dag.qubits)
        if num_dag_qubits > self.coupling_map.size():
            raise TranspilerError("Number of qubits greater than device.")
        num_cx = 0
        num_meas = 0

        # Get avg number of cx and meas per qubit
        ops = dag.count_ops()
        if "cx" in ops.keys():
            num_cx = ops["cx"]
        if "measure" in ops.keys():
            num_meas = ops["measure"]

        best_sub = self._best_subset(num_dag_qubits, num_meas, num_cx)
        layout = Layout()
        map_iter = 0
        for qreg in dag.qregs.values():
            for i in range(qreg.size):
                layout[qreg[i]] = int(best_sub[map_iter])
                map_iter += 1
            layout.add_register(qreg)
        self.property_set["layout"] = layout

    def _best_subset(self, num_qubits, num_meas, num_cx):
        """Computes the qubit mapping with the best connectivity.

        Args:
            num_qubits (int): Number of subset qubits to consider.

        Returns:
            ndarray: Array of qubits to use for best connectivity mapping.
        """
        from scipy.sparse import coo_matrix, csgraph

        if num_qubits == 1:
            return np.array([0])
        if num_qubits == 0:
            return []

        rows, cols, best_map = best_subset(
            num_qubits,
            self.adjacency_matrix,
            num_meas,
            num_cx,
            bool(self.backend_prop),
            self.coupling_map.is_symmetric,
            self.error_mat,
        )
        data = [1] * len(rows)
        sp_sub_graph = coo_matrix((data, (rows, cols)), shape=(num_qubits, num_qubits)).tocsr()
        perm = csgraph.reverse_cuthill_mckee(sp_sub_graph)
        best_map = best_map[perm]
        return best_map
