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

"""Check if the gates follow the right direction with respect to the coupling map."""

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass


class CheckGateDirection(AnalysisPass):
    """Check if the two-qubit gates follow the right direction with
    respect to the coupling map.
    """

    def __init__(self, coupling_map):
        """CheckGateDirection initializer.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """Run the CheckGateDirection pass on `dag`.

        If `dag` is mapped and the direction is correct the property
        `is_direction_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to check.
        """
        self.property_set["is_direction_mapped"] = True
        edges = self.coupling_map.get_edges()

        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        for gate in dag.two_qubit_ops():
            physical_q0 = trivial_layout[gate.qargs[0]]
            physical_q1 = trivial_layout[gate.qargs[1]]

            if (physical_q0, physical_q1) not in edges:
                self.property_set["is_direction_mapped"] = False
                return
