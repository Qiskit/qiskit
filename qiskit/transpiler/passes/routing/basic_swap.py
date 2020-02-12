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

"""Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.extensions.standard import SwapGate


class BasicSwap(TransformationPass):
    """Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.

    The basic mapper is a minimum effort to insert swap gates to map the DAG onto
    a coupling map. When a cx is not in the coupling map possibilities, it inserts
    one or more swaps in front to make it compatible.
    """

    def __init__(self, coupling_map):
        """BasicSwap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """Run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        new_dag = DAGCircuit()

        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Basic swap runs on physical circuits only')

        if len(dag.qubits()) > len(self.coupling_map.physical_qubits):
            raise TranspilerError('The layout does not match the amount of qubits in the DAG')

        canonical_register = dag.qregs['q']
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer['graph']

            for gate in subdag.twoQ_gates():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(SwapGate(),
                                                        qargs=[qubit_1, qubit_2],
                                                        cargs=[])

                    # layer insertion
                    edge_map = current_layout.combine_into_edge_map(trivial_layout)
                    new_dag.compose_back(swap_layer, edge_map)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])

            edge_map = current_layout.combine_into_edge_map(trivial_layout)
            new_dag.extend_back(subdag, edge_map)

        return new_dag
