# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A pass implementing a basic mapper.

The basic mapper is a minimum effort to insert swap gates to map the DAG into a coupling map. When
a cx is not in the coupling map possibilities, it inserts one or more swaps in front to make it
compatible.
"""

from copy import copy

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import Layout
from qiskit.extensions.standard import SwapGate


class BasicMapper(TransformationPass):
    """
    Maps (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.
    """

    def __init__(self,
                 coupling_map,
                 swap_gate=None,
                 initial_layout=None):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates.
        Args:
            coupling_map (Coupling): Directed graph represented a coupling map.
            swap_instruction (Type):  Default: SwapGate. The Gate class that defines a swap gate.
            initial_layout (Layout): initial layout of qubits in mapping
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.swap_gate = swap_gate if swap_gate is not None else SwapGate

    def run(self, dag):
        """
        Runs the BasicMapper pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.
        """
        new_dag = DAGCircuit()

        if self.initial_layout is None:
            # create a one-to-one layout
            self.initial_layout = Layout()
            physical_qubit = 0
            for qreg in dag.qregs.values():
                for index in range(qreg.size):
                    self.initial_layout[(qreg, index)] = physical_qubit
                    physical_qubit += 1
        current_layout = copy(self.initial_layout)

        for layer in dag.serial_layers():
            subdag = layer['graph']

            for a_cx in subdag.get_cnot_nodes():
                physical_q0 = current_layout[a_cx['op'].qargs[0]]
                physical_q1 = current_layout[a_cx['op'].qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        closest_qubit = current_layout[path[swap + 1]]
                        farthest_qubit = current_layout[path[swap + 2]]

                        # create the involved registers
                        if closest_qubit[0] not in swap_layer.qregs.values():
                            swap_layer.add_qreg(closest_qubit[0])
                        if farthest_qubit[0] not in swap_layer.qregs.values():
                            swap_layer.add_qreg(farthest_qubit[0])

                        # create the swap operation
                        swap_layer.add_basis_element('swap', 2, 0, 0)
                        swap_layer.apply_operation_back(
                            self.swap_gate(closest_qubit, farthest_qubit))

                        # update current_layout
                        current_layout.swap(closest_qubit, farthest_qubit)

                        # swap the order in shortest path
                        path[swap + 2], path[swap + 1] = path[swap + 1], path[swap + 2]

                    # layer insertion
                    edge_map = current_layout.combine_into_edge_map(self.initial_layout)
                    new_dag.extends_at_the_end(swap_layer, edge_map)

            edge_map = current_layout.combine_into_edge_map(self.initial_layout)
            new_dag.extends_at_the_end(subdag, edge_map)

        return new_dag
