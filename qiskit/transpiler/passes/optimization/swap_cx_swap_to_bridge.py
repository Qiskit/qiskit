# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Convert SWAP-CX-SWAP sub-circuit to bridge gate"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.generalized_gates import BridgeGate


class SwapCXSwaptoBridge(TransformationPass):
    """Runs the SwapCXSwaptoBridge pass on `dag`.

    Transpiler pass to convert SWAP-CX-SWAP subcircuit to BridgeGate.
    This runs post-routing to find instances where a routing pass inserted a SWAP-CX-SWAP and replaces it
    with a bridge gate. It allows for all non-bridge aware routing passes to take advantage
    of the bridge gate, without requiring them to be updated individually.
    """

    def run(self, dag):

        """Run the CircuittoBridge pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        for node in dag.op_nodes():

            # Find any CX node in the input circuit (DAG) and get the list of Successors and Predecessors

            if node.op.name == "cx":  # First element of the pattern is CX

                successors_of_cx = list(dag.successors(node))
                predecessors_of_cx = list(dag.predecessors(node))

                # Check if a child node is a swap gate.
                for succ in successors_of_cx:

                    # Second element of the pattern = SWAP1(Swap predecessor of CX)
                    if succ.type == "op" and succ.op.name == "swap":

                        # check the SWAP gate in the parent nodes  .
                        for pred in predecessors_of_cx:

                            # Third Element of the pattern = SWAP2(Swap sucessor of CX)
                            if pred.type == "op" and pred.op.name == "swap":
                                succ_of_predecessors_of_cx = dag.successors(pred)

                                # This check the common parent, and we have the pattern here
                                if succ in succ_of_predecessors_of_cx:

                                    # The list of the predecessors and successors of the pattern
                                    pred_of_predecessors_of_cx = dag.predecessors(
                                        pred
                                    )  # predecessor of the SWAP1

                                    succ_of_successors_of_cx = dag.successors(
                                        succ
                                    )  # Sucessors of the SWAP2

                                    # Remove SWAP1 and SWAP2 from the list
                                    # of successors and predecessors of CX
                                    successors_of_cx.remove(succ)
                                    predecessors_of_cx.remove(pred)

                                    set_predecessors_of_the_pattern = {
                                        *pred_of_predecessors_of_cx,
                                        predecessors_of_cx[0],
                                    }
                                    set_successors_of_the_pattern = {
                                        *succ_of_successors_of_cx,
                                        successors_of_cx[0],
                                    }

                                    # Insert the bridge gate
                                    bridge_gate = BridgeGate(3).to_instruction()

                                    print(bridge_gate)

                                    # Find first the ordered qargs of the new bridge Gate.
                                    # The target qubit will be the target qubit of the CX gate.
                                    # The ancilla qubit will be the control qubit of the CX gate.
                                    # The control qubit will the qarg of either SWAP
                                    # gate which is /NOT/ the ancilla qubit.

                                    if node.qargs[0] in pred.qargs:
                                        bridge_target_qarg = node.qargs[1]
                                        bridge_ancilla_qarg = node.qargs[0]
                                        bridge_control_qarg = [
                                            q for q in pred.qargs if q != bridge_ancilla_qarg
                                        ][0]
                                        bridge_qargs = [
                                            bridge_control_qarg,
                                            bridge_ancilla_qarg,
                                            bridge_target_qarg,
                                        ]
                                    else:
                                        bridge_control_qarg = node.qargs[0]
                                        bridge_ancilla_qarg = node.qargs[1]
                                        bridge_target_qarg = [
                                            q for q in pred.qargs if q != bridge_ancilla_qarg
                                        ][0]
                                        bridge_qargs = [
                                            bridge_control_qarg,
                                            bridge_ancilla_qarg,
                                            bridge_target_qarg,
                                        ]

                                    bridge_node_index = dag._add_op_node(
                                        bridge_gate, bridge_qargs, []
                                    )

                                    for pattern_pred_node in set_predecessors_of_the_pattern:
                                        shared_qargs = set(bridge_qargs).intersection(
                                            pattern_pred_node.qargs
                                        )

                                        for qarg in shared_qargs:
                                            dag._multi_graph.add_edge(
                                                pattern_pred_node._node_id, bridge_node_index, qarg
                                            )

                                    for pattern_succ_node in set_successors_of_the_pattern:
                                        shared_qargs = set(bridge_qargs).intersection(
                                            pattern_succ_node.qargs
                                        )

                                        for qarg in shared_qargs:
                                            dag._multi_graph.add_edge(
                                                pattern_succ_node._node_id, bridge_node_index, qarg
                                            )

                                    # Removing the old nodes will remove the old edges as well.
                                    dag.remove_op_node(node)
                                    dag.remove_op_node(pred)
                                    dag.remove_op_node(succ)

        return dag
