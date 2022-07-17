# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Depth optimization using commutativity analysis."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag


class OptimizeDepth(TransformationPass):
    """Optimize depth using commutativity analysis."""

    def run(self, dag):
        """Run the OptimizeDepth pass on `dag`.

        Internally, converts DagCircuit to DagDependency, and then
        converts DagDependency back to DagCircuit (using depth minimization heuristic).

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        original_depth = dag.depth()
        dagdependency = dag_to_dagdependency(dag)
        optimized_dag = dagdependency_to_dag(dagdependency, optimize_depth=True)
        optimized_depth = optimized_dag.depth()

        # Since "depth minimization" is heuristic, we return the optimized dag
        # only if it's depth is really smaller.
        if optimized_depth < original_depth:
            return optimized_dag
        else:
            return dag
