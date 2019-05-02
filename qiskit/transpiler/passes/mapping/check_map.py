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

"""This pass checks if a DAG circuit is already mapped to a coupling map.

It checks that all 2-qubit interactions are laid out to be physically close.
"""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler import Layout


class CheckMap(AnalysisPass):
    """
    Checks if a DAGCircuit is mapped to `coupling_map`, setting `is_swap_mapped`
    in the property set as True if mapped. False otherwise.
    """

    def __init__(self, coupling_map, initial_layout=None):
        """
        Checks if a DAGCircuit is mapped to `coupling_map`.
        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            initial_layout (Layout): The initial layout of the DAG to analyze.
        """
        super().__init__()
        self.layout = initial_layout  # TODO: this arg is never used, remove.
        self.coupling_map = coupling_map

    def run(self, dag):
        """
        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        """
        if self.layout is None:
            if self.property_set["layout"]:
                self.layout = self.property_set["layout"]
            else:
                self.layout = Layout.generate_trivial_layout(*dag.qregs.values())

        self.property_set['is_swap_mapped'] = True

        for gate in dag.twoQ_gates():
            physical_q0 = self.layout[gate.qargs[0]]
            physical_q1 = self.layout[gate.qargs[1]]

            if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                self.property_set['is_swap_mapped'] = False
                return
