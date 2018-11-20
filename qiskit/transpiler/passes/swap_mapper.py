# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
TODO
"""

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import Layout

class SwapMapper(TransformationPass):
    """
    Maps a DAGCircuit onto a `coupling_map` using swap gates.
    """

    def __init__(self,
                 coupling_map,
                 swap_basis_element='swap',
                 swap_data=None,
                 initial_layout=None):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates.
        Args:
            coupling_map (Coupling): Directed graph represented a coupling map.
            swap_basis_element (string):  Default: 'swap' the name of the gate
               that will be used for swaping.
            swap_data (dict): The swap "gate data". Default: the swap gate is opaque.
        """
        super().__init__()
        self.layout = initial_layout
        self.coupling_map = coupling_map
        self.swap_basis_element = swap_basis_element
        self.swap_data = swap_data if swap_data is not None else {"opaque": True,
                                                                  "n_args": 0,
                                                                  "n_bits": 2,
                                                                  "args": [],
                                                                  "bits": ["a", "b"]}

    def run(self, dag):
        """
        Runs the SwapMapper pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.
        """
        new_dag = DAGCircuit()

        if self.layout is None:
            # create a one-to-one layout
            self.layout = Layout()
            wire_no = 0
            for qreg in dag.qregs.values():
                for index in range(qreg.size):
                    self.layout[(qreg.name, index)] = wire_no
                    wire_no += 1

        for layer in dag.serial_layers():
            subdag = layer['graph']
            cxs = subdag.get_cnot_nodes()
            if not cxs:
                # Trivial layer, there is no entanglement in this layer, just leave it like this.
                new_dag.add_dag_at_the_end(subdag, self.layout)
                continue
            for a_cx in subdag.get_cnot_nodes():
                physical_q0 = ('q', self.layout[a_cx['qargs'][0]])
                physical_q1 = ('q', self.layout[a_cx['qargs'][1]])
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert the SWAP when the CXs are not already together.
                    path = self.coupling_map.shortest_path(physical_q0, physical_q1)
                    new_dag.add_basis_element(self.swap_basis_element, 2)
                    closest_physical = self.layout[path[1]['name'][1]]
                    farthest_physical = self.layout[path[-1]['name'][1]]
                    new_dag.apply_operation_back(self.swap_basis_element,
                                                 [closest_physical, farthest_physical])
                    new_dag.add_gate_data(self.swap_basis_element, self.swap_data)
                    self.layout.swap(farthest_physical[1], closest_physical[1])
                new_dag.add_dag_at_the_end(subdag, self.layout)
        return new_dag
