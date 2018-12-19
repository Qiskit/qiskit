# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A pass for choosing a Layout of a circuit onto a Coupling graph.

This pass associates a physical qubit (int) to each virtual qubit
of the circuit (tuple(QuantumRegister, int)).

Note: even though a 'layout' is not strictly a property of the DAG,
in the transpiler architecture it is best passed around between passes by
being set in `property_set`.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

from qiskit.mapper import Layout
from qiskit.transpiler._basepasses import AnalysisPass
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler import TranspilerError


class DenseLayout(AnalysisPass):
    """
    Chooses a Layout by finding the most connected subset of qubits.
    """

    def __init__(self, coupling_map):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.

        Raises:
            TranspilerError: if invalid options
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.requires.append(CheckMap(self.coupling_map))

    def run(self, dag):
        """
        Run the DenseLayout pass on `dag`, and set the property `layout`.

        The following scenarios will be taken in order:
        1. layout=qr[i]->i, if the dag can be trivially laid out
        2. layout=_pick_best_layout(), otherwise

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: with malformed inputs
        """
        if self.property_set['is_direction_mapped']:
            trivial_layout = Layout()
            for qreg in dag.qregs.values():
                trivial_layout.add_register(qreg)
            self.property_set['layout'] = trivial_layout
        else:
            self.property_set['layout'] = self._pick_best_layout(dag)

    def _pick_best_layout(self, dag):
        """Pick a convenient layout depending on the best matching
        qubit connectivity.

        Args:
            dag (DAGCircuit): DAG representation of circuit.

        Returns:
            Layout: a good layout for the virtual qubits in DAG to physical qubits
                in coupling_map

        Raises:
            TranspilerError: if wrong number of qubits given.
        """
        num_dag_qubits = sum([qreg.size for qreg in dag.qregs.values()])
        if num_dag_qubits > self.coupling_map.size():
            raise TranspilerError('Number of qubits greater than device.')
        best_sub = self._best_subset(num_dag_qubits)
        layout = Layout()
        map_iter = 0
        for qreg in dag.qregs.values():
            for i in range(qreg.size):
                layout[(qreg, i)] = int(best_sub[map_iter])
                map_iter += 1
        return layout

    def _best_subset(self, n_qubits):
        """Computes the qubit mapping with the best connectivity.

        Args:
            n_qubits (int): Number of subset qubits to consider.

        Returns:
            ndarray: Array of qubits to use for best connectivity mapping.
        """
        if n_qubits == 1:
            return np.array([0])

        device_qubits = self.coupling_map.size()

        cmap = np.asarray(self.coupling_map.get_edges())
        data = np.ones_like(cmap[:, 0])
        sp_cmap = sp.coo_matrix((data, (cmap[:, 0], cmap[:, 1])),
                                shape=(device_qubits, device_qubits)).tocsr()
        best = 0
        best_map = None
        # do bfs with each node as starting point
        for k in range(sp_cmap.shape[0]):
            bfs = cs.breadth_first_order(sp_cmap, i_start=k, directed=False,
                                         return_predecessors=False)

            connection_count = 0
            for i in range(n_qubits):
                node_idx = bfs[i]
                for j in range(sp_cmap.indptr[node_idx],
                               sp_cmap.indptr[node_idx + 1]):
                    node = sp_cmap.indices[j]
                    for counter in range(n_qubits):
                        if node == bfs[counter]:
                            connection_count += 1
                            break

            if connection_count > best:
                best = connection_count
                best_map = bfs[0:n_qubits]
        return best_map
