# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
This pass checks if the CNOTs (or any other 2Q) in the DAG follow the right
direction with respect to thecoupling map.
"""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.mapper import Layout
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.cxbase import CXBase


class CheckCnotDirection(AnalysisPass):
    """
    Checks if the CNOTs in the DAG circuit follow the right
    direction with respect to the coupling map.
    """

    def __init__(self, coupling_map, initial_layout=None):
        """
        Checks if the CNOTs in DAGCircuit are in the allowed direction with
        respect to `coupling_map`.
        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            initial_layout (Layout): The initial layout of the DAG to analyze.
        """
        super().__init__()
        self.layout = initial_layout
        self.coupling_map = coupling_map

    def run(self, dag):
        """
        If `dag` is mapped and the direction is correct the property
        `is_direction_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to check.
        """
        if self.layout is None:
            if self.property_set["layout"]:
                self.layout = self.property_set["layout"]
            else:
                self.layout = Layout.generate_trivial_layout(*dag.qregs.values())

        self.property_set['is_direction_mapped'] = True
        edges = self.coupling_map.get_edges()

        for gate in dag.twoQ_gates():
            physical_q0 = self.layout[gate.qargs[0]]
            physical_q1 = self.layout[gate.qargs[1]]

            if isinstance(gate.op, (CXBase, CnotGate)) and (
                    physical_q0, physical_q1) not in edges:
                self.property_set['is_direction_mapped'] = False
                return
