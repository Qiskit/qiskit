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
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements


class BasicSwap(TransformationPass):
    """
    Maps (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.
    """

    def __init__(self,
                 coupling_map,
                 initial_layout=None):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates.
        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            initial_layout (Layout): initial layout of qubits in mapping
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.swap_gate = SwapGate
        self.requires.append(BarrierBeforeFinalMeasurements())

    def run(self, dag):
        """
        Runs the BasicSwap pass on `dag`.
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
                physical_q0 = current_layout[a_cx['qargs'][0]]
                physical_q1 = current_layout[a_cx['qargs'][1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the involved registers
                        if qubit_1[0] not in swap_layer.qregs.values():
                            swap_layer.add_qreg(qubit_1[0])
                        if qubit_2[0] not in swap_layer.qregs.values():
                            swap_layer.add_qreg(qubit_2[0])

                        # create the swap operation
                        swap_layer.add_basis_element('swap', 2, 0, 0)
                        swap_layer.apply_operation_back(self.swap_gate(qubit_1, qubit_2),
                                                        qargs=[qubit_1, qubit_2])

                    # layer insertion
                    edge_map = current_layout.combine_into_edge_map(self.initial_layout)
                    new_dag.compose_back(swap_layer, edge_map)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])

            edge_map = current_layout.combine_into_edge_map(self.initial_layout)
            new_dag.extend_back(subdag, edge_map)

        return new_dag
