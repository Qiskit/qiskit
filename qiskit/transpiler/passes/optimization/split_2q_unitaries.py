# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

import numpy as np

from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
from qiskit.synthesis.two_qubit.two_qubit_decompose import decompose_two_qubit_product_gate
from qiskit.transpiler import TransformationPass


class Split2QUnitaries(TransformationPass):
    """Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

    def run(self, dag: DAGCircuit):
        """Run the Split2QUnitaries pass on `dag`."""
        sq_id = np.eye(2)
        for node in dag.topological_op_nodes():
            # skip operations without two-qubits and for which we can not determine a potential 1q split
            if (
                len(node.cargs) > 0
                or len(node.qargs) != 2
                or node.matrix is None
                or node.is_parameterized()
            ):
                continue

            # check if the node can be represented by single-qubit gates
            nmat = node.matrix
            local_invariants = two_qubit_local_invariants(nmat)
            if local_invariants[0] == 1 and local_invariants[1] == 0 and local_invariants[2] == 3:
                ul, ur, phase = decompose_two_qubit_product_gate(nmat)

                if not np.allclose(ur, sq_id):
                    ur_node = DAGOpNode.from_instruction(
                        CircuitInstruction(UnitaryGate(ur), qubits=(node.qargs[0],)), dag=dag
                    )
                    ur_node._node_id = dag._multi_graph.add_node(ur_node)
                    dag._increment_op("unitary")
                    dag._multi_graph.insert_node_on_in_edges(ur_node._node_id, node._node_id)

                if not np.allclose(ul, sq_id):
                    ul_node = DAGOpNode.from_instruction(
                        CircuitInstruction(UnitaryGate(ul), qubits=(node.qargs[1],)), dag=dag
                    )
                    ul_node._node_id = dag._multi_graph.add_node(ul_node)
                    dag._increment_op("unitary")
                    dag._multi_graph.insert_node_on_in_edges(ul_node._node_id, node._node_id)

                dag.global_phase += phase
                dag.remove_op_node(node)

        return dag
