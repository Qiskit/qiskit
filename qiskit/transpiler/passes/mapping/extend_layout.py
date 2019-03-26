# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A pass for extending a Layout to use all the nodes in the  Coupling graph.
"""

from qiskit.transpiler.basepasses import AnalysisPass


class ExtendLayout(AnalysisPass):
    """
    Extends a Layout with the idle nodes from the coupling map.
    """

    def __init__(self, coupling_map):
        """
        Extends a Layout with the idle nodes from coupling_map.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """
        Args:
            dag (DAGCircuit): the parameter will be ignored.
        """
        physcial_qubits_in_layout = self.property_set['layout'].get_physical_bits().keys()
        for physcial_qubit in self.coupling_map.physical_qubits:
            if physcial_qubit not in physcial_qubits_in_layout:
                self.property_set['layout'][physcial_qubit] = None
