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
