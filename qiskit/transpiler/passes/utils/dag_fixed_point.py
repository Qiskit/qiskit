# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if the DAG has reached a fixed point."""

from copy import deepcopy

from qiskit.transpiler.basepasses import AnalysisPass


class DAGFixedPoint(AnalysisPass):
    """Check if the DAG has reached a fixed point.

    A dummy analysis pass that checks if the DAG a fixed point (the DAG is not
    modified anymore). The results is saved in
    ``property_set['dag_fixed_point']`` as a boolean.
    """

    def run(self, dag):
        """Run the DAGFixedPoint pass on `dag`."""
        if self.property_set['_dag_fixed_point_previous_dag'] is None:
            self.property_set['dag_fixed_point'] = False
        else:
            fixed_point_reached = self.property_set['_dag_fixed_point_previous_dag'] == dag
            self.property_set['dag_fixed_point'] = fixed_point_reached

        self.property_set['_dag_fixed_point_previous_dag'] = deepcopy(dag)
