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

"""Replace each SWAP-CX-SWAP sequence by a single Bridge gate."""

from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.circuit.library.standard_gates import CXGate, SwapGate
from qiskit.circuit.library.generalized_gates import BridgeGate
from qiskit.transpiler.basepasses import TransformationPass


class SwapCXSwapToBridge(TransformationPass):
    """Replace each SWAP-CX-SWAP sequence by a single Bridge gate.

    This pass finds CX gates surrounded by SWAP compute/uncompute pairs
    that act on either the control or target qubit of CX and replaces them
    by a Bridge Gate which has fewer CNOTs. Note that it's able to handle
    any sequence of such SWAP pairs, not just one pair. For example, the
    following subcircuit will be collected:

                                                           ┌─────────┐
        q_0: ─X─────────────X─                        q_0: ┤0        ├
              │             │                              │         │
        q_1: ─X──X───────X──X─                        q_1: ┤1        ├
                 │       │             =>                  │  Bridge │
        q_2: ────X───■───X────                        q_2: ┤2        ├
                   ┌─┴─┐                                   │         │
        q_3: ──────┤ X ├──────                        q_3: ┤3        ├
                   └───┘                                   └─────────┘

    This transpiler pass is particularly useful since routing passes may
    insert such subcircuits and hence it should be run after routing.
    """

    def run(self, dag):
        """Run the SwapCXSwapToBridge pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        marked = set()
        qubit_indices = {qb: ix for ix, qb in enumerate(dag.qubits)}

        def is_pred_unmarked_and_on_qubit(edge, qubit):
            pred, _, edge_qb = edge
            return isinstance(pred, DAGOpNode) and edge_qb == qubit and pred not in marked

        def is_succ_unmarked_and_on_qubit(edge, qubit):
            _, succ, edge_qb = edge
            return isinstance(succ, DAGOpNode) and edge_qb == qubit and succ not in marked

        def is_swap_pair(nd_a, nd_b):
            return (
                isinstance(nd_a.op, SwapGate)
                and nd_a.op.condition is None
                and isinstance(nd_b.op, SwapGate)
                and nd_b.op.condition is None
                and sorted(nd_a.qargs, key=lambda q: qubit_indices[q])
                == sorted(nd_b.qargs, key=lambda q: qubit_indices[q])
                and dag.is_successor(nd_a, nd_b)
            )

        blocks = []
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, CXGate):
                continue

            def follow_qubit_through_swap_pairs(qubit):
                block_nd = []
                wires = [qubit]

                pred, succ = node, node
                while True:
                    try:
                        pred, _, _ = next(
                            filter(
                                lambda raw: is_pred_unmarked_and_on_qubit(raw, qubit),
                                dag.edges(pred, incoming=True),
                            )
                        )
                        _, succ, _ = next(
                            filter(
                                lambda raw: is_succ_unmarked_and_on_qubit(raw, qubit),
                                dag.edges(succ, incoming=False),
                            )
                        )

                        if not is_swap_pair(pred, succ):
                            break

                        qubits = pred.qargs
                        # move now to the qubit of this swap pair that we have not yet "explored".
                        qubit = qubits[1 - qubits.index(qubit)]

                        block_nd.extend([pred, succ])
                        wires.append(qubit)
                    except StopIteration:
                        break

                return block_nd, wires

            # follow control qubit
            c_qubit = node.qargs[0]
            c_block, c_wires = follow_qubit_through_swap_pairs(c_qubit)
            marked.update(c_block)

            # follow target qubit
            t_qubit = node.qargs[1]
            t_block, t_wires = follow_qubit_through_swap_pairs(t_qubit)
            marked.update(t_block)

            if not (c_block or t_block):
                continue

            block_tot = c_block + [node] + t_block
            wires = c_wires[::-1] + t_wires
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(wires))
            blocks.append((block_tot, wire_pos_map, node.op.condition))

        # now replace the collected blocks with a Bridge gate.
        for block, wire_pos_map, cx_node_condition in blocks:
            op = BridgeGate(len(wire_pos_map))
            op.condition = cx_node_condition
            dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)

        return dag
