# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
This pass checks if a DAG is mapped to a coupling map.

It checks that all 2-qubit interactions are laid out to be physically close.
"""

from qiskit.transpiler._basepasses import AnalysisPass
from qiskit.mapper import Layout
from qiskit.extensions.standard.swap import SwapGate


class CheckMap(AnalysisPass):
    """
    Checks if a DAGCircuit is mapped to `coupling_map`.

    It checks that all 2-qubit interactions are laid out to be physically close.
    """

    def __init__(self, coupling_map, initial_layout=None):
        """
        Checks if a DAGCircuit is mapped to `coupling_map`.
        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            initial_layout (Layout): The initial layout of the DAG to analyze.
        """
        super().__init__()
        self.layout = initial_layout
        self.coupling_map = coupling_map
        self.results = {'is_swap_mapped': [],
                        'is_direction_mapped': []}

    def run(self, dag):
        """
        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).
        If `dag` is mapped and the direction is correct the property
        `is_direction_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        """
        if self.layout is None:
            self.layout = Layout()
            for qreg in dag.qregs.values():
                self.layout.add_register(qreg)

        self.property_set['is_swap_mapped'] = True
        self.property_set['is_direction_mapped'] = True

        for gate in dag.get_2q_nodes():
            physical_q0 = self.layout[gate['qargs'][0]]
            physical_q1 = self.layout[gate['qargs'][1]]

            if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                self.property_set['is_swap_mapped'] = False
                self.property_set['is_direction_mapped'] = False
                return
            else:
                if (physical_q0, physical_q1) not in self.coupling_map.get_edges():
                    self.property_set['is_direction_mapped'] = False

            if isinstance(gate['op'], SwapGate):
                if (physical_q1, physical_q0) not in self.coupling_map.get_edges():
                    self.property_set['is_direction_mapped'] = False
