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

"""Check if a DAG circuit is already mapped to a coupling map."""

from qiskit.transpiler.basepasses import AnalysisPass


class CheckMap(AnalysisPass):
    """Check if a DAG circuit is already mapped to a coupling map.

    Check if a DAGCircuit is mapped to `coupling_map` by checking that all
    2-qubit interactions are laid out to be physically close, setting the
    property ``is_swap_mapped`` to ``True`` or ``False`` accordingly.
    """

    def __init__(self, coupling_map):
        """CheckMap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """Run the CheckMap pass on `dag`.

        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        """
        self.property_set['is_swap_mapped'] = True

        for gate in dag.twoQ_gates():
            physical_q0 = gate.qargs[0].index
            physical_q1 = gate.qargs[1].index

            if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                self.property_set['is_swap_mapped'] = False
                return
