# -*- coding: utf-8 -*-

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
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


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
        self.cx_mat = None
        self.meas_arr = None
        self.num_cx = 0
        self.num_meas = 0

    def run(self, dag):
        """Run the DenseLayout pass on `dag`.

        Pick a convenient layout depending on the best matching
        qubit connectivity, and set the property `layout`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        num_dag_qubits = sum([qreg.size for qreg in dag.qregs.values()])
        if num_dag_qubits > self.coupling_map.size():
            raise TranspilerError('Number of qubits greater than device.')

        # Get avg number of cx and meas per qubit
        ops = dag.count_ops()
        if 'cx' in ops.keys():
            self.num_cx = ops['cx']
        if 'measure' in ops.keys():
            self.num_meas = ops['measure']

        # Compute the sparse cx_err matrix and meas array
        device_qubits = self.coupling_map.size()
        if self.backend_prop:
            rows = []
            cols = []
            cx_err = []

            for edge in self.coupling_map.get_edges():
                for gate in self.backend_prop.gates:
                    if gate.qubits == edge:
                        rows.append(edge[0])
                        cols.append(edge[1])
                        cx_err.append(gate.parameters[0].value)
                        break
                else:
                    continue

            self.cx_mat = sp.coo_matrix((cx_err, (rows, cols)),
                                        shape=(device_qubits,
                                               device_qubits)).tocsr()

            # Set measurement array
            meas_err = []
            for qubit_data in self.backend_prop.qubits:
                for item in qubit_data:
                    if item.name == 'readout_error':
                        meas_err.append(item.value)
                        break
                else:
                    continue
            self.meas_arr = np.asarray(meas_err)

        best_sub = self._best_subset(num_dag_qubits)
        layout = Layout()
        map_iter = 0
        for qreg in dag.qregs.values():
            for i in range(qreg.size):
                layout[qreg[i]] = int(best_sub[map_iter])
                map_iter += 1
        self.property_set['layout'] = layout

    def _best_subset(self, num_qubits):
        """Computes the qubit mapping with the best connectivity.

        Args:
            num_qubits (int): Number of subset qubits to consider.

        Returns:
            ndarray: Array of qubits to use for best connectivity mapping.
        """
        if num_qubits == 1:
            return np.array([0])
        if num_qubits == 0:
            return []

        device_qubits = self.coupling_map.size()

        cmap = np.asarray(self.coupling_map.get_edges())
        data = np.ones_like(cmap[:, 0])
        sp_cmap = sp.coo_matrix((data, (cmap[:, 0], cmap[:, 1])),
                                shape=(device_qubits, device_qubits)).tocsr()
        best = 0
        best_map = None
        best_error = np.inf
        best_sub = None
        # do bfs with each node as starting point
        for k in range(sp_cmap.shape[0]):
            bfs = cs.breadth_first_order(sp_cmap, i_start=k, directed=False,
                                         return_predecessors=False)

            connection_count = 0
            sub_graph = []
            for i in range(num_qubits):
                node_idx = bfs[i]
                for j in range(sp_cmap.indptr[node_idx],
                               sp_cmap.indptr[node_idx + 1]):
                    node = sp_cmap.indices[j]
                    for counter in range(num_qubits):
                        if node == bfs[counter]:
                            connection_count += 1
                            sub_graph.append([node_idx, node])
                            break

            if self.backend_prop:
                curr_error = 0
                # compute meas error for subset
                avg_meas_err = np.mean(self.meas_arr)
                meas_diff = np.mean(
                    self.meas_arr[bfs[0:num_qubits]])-avg_meas_err
                if meas_diff > 0:
                    curr_error += self.num_meas*meas_diff

                cx_err = np.mean([self.cx_mat[edge[0], edge[1]]
                                  for edge in sub_graph])
                if self.coupling_map.is_symmetric:
                    cx_err /= 2
                curr_error += self.num_cx*cx_err
                if connection_count >= best and curr_error < best_error:
                    best = connection_count
                    best_error = curr_error
                    best_map = bfs[0:num_qubits]
                    best_sub = sub_graph

            else:
                if connection_count > best:
                    best = connection_count
                    best_map = bfs[0:num_qubits]
                    best_sub = sub_graph

        # Return a best mapping that has reduced bandwidth
        mapping = {}
        for edge in range(best_map.shape[0]):
            mapping[best_map[edge]] = edge
        new_cmap = [[mapping[c[0]], mapping[c[1]]] for c in best_sub]
        rows = [edge[0] for edge in new_cmap]
        cols = [edge[1] for edge in new_cmap]
        data = [1]*len(rows)
        sp_sub_graph = sp.coo_matrix((data, (rows, cols)),
                                     shape=(num_qubits, num_qubits)).tocsr()
        perm = cs.reverse_cuthill_mckee(sp_sub_graph)
        best_map = best_map[perm]
        return best_map
