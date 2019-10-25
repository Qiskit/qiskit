# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A pass to evaluate how good the layout selection was.
No CX direction is considered.
Saves in `layout_score` (or `property` param) the sum of the distance off for each CX.
The lower the number, the better the selection.
Therefore, 0 is a perfect layout selection.
"""

from qiskit.transpiler.basepasses import AnalysisPass


class LayoutScore(AnalysisPass):
    """
    Saves in `layout_score` (or `property` param) the
    sum of the distance off for each CX.
    """
    def __init__(self, coupling_map, initial_layout=None, property=None):
        super().__init__()
        self.layout = initial_layout
        self.coupling_map = coupling_map
        self.property = property

    def run(self, dag):
        self.layout = self.layout or self.property_set["layout"]

        property = self.property or 'layout_score'

        if self.layout is None:
            return

        distances = []
        self.property_set[property] = 0

        for gate in dag.twoQ_gates():
            physical_q0 = self.layout[gate.qargs[0]]
            physical_q1 = self.layout[gate.qargs[1]]

            distances.append(self.coupling_map.distance(physical_q0, physical_q1)-1)

        self.property_set[property] += sum(distances)
