# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Set the ``layout`` property to the given layout."""


from qiskit.transpiler.basepasses import AnalysisPass


class SetLayout(AnalysisPass):
    """Set the ``layout`` property to the given layout.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit) in increasing order.
    """

    def __init__(self, layout):
        """SetLayout initializer.

        Args:
            layout (Layout): the layout to set.
        """
        super().__init__()
        self.layout = layout

    def run(self, dag):
        """Run the SetLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: the original DAG.
        """
        self.property_set['layout'] = self.layout
        return dag
