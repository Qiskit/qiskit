# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Synthesize high-level objects."""

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info.synthesis.clifford_decompose import decompose_clifford


class HighLevelSynthesis(TransformationPass):
    """Synthesize high-level objects by choosing the appropriate synthesis method based on
    the object's name.
    """

    # TODO: Currently, this class only contains the minimal functionality required to transpile
    #       Cliffords. In the near future, this class will be expanded to cover other higher-level
    #       objects (as these become available). Additionally, the plan is to make HighLevelSynthesis
    #       "pluggable", so that the users would be able to "plug in" their own synthesis methods
    #       for higher-level objects (which would be called during transpilation).

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with high level objects synthesized.
        """

        for node in dag.named_nodes("clifford"):
            decomposition = circuit_to_dag(decompose_clifford(node.op))
            dag.substitute_node_with_dag(node, decomposition)

        return dag
