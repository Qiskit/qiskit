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


"""Commute gates to reduce circuit depth."""

from qiskit import converters
from qiskit.transpiler.basepasses import TransformationPass


class CommuteMinimumDepth(TransformationPass):
    r"""This pass converts a class:`~DagCircuit` to a class:`~DagDependency`
    and then back to a class:`~DagCircuit`.

    The optimization is done in the conversion from the class:`~DagDependency`
    to the class:`~DagCircuit
    """

    def run(self, dag):
        """Run the CommuteMinimumDepth pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        return converters.dagdependency_to_dag(converters.dag_to_dagdependency(dag))
