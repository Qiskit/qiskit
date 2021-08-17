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

"""Collect sequences of uninterrupted gates acting on 2 qubits."""

from collections import defaultdict

from qiskit.circuit import Gate
from qiskit.transpiler.basepasses import AnalysisPass


class Collect2qBlocks(AnalysisPass):
    """Collect two-qubit subcircuits."""

    def run(self, dag):
        """Run the Collect2qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['block_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set["commutation_set"] = defaultdict(list)
        pending_1q = [list() for _ in range(dag.num_qubits())]
        block_id = [-(i + 1) for i in range(dag.num_qubits())]
        current_id = 0
        block_list = list()
        to_qid = dict()
        for i, qubit in enumerate(dag.qubits):
            to_qid[qubit] = i
        for node in dag.topological_op_nodes():
            qids = [to_qid[q] for q in node.qargs]
            if (
                not isinstance(node.op, Gate)
                or len(qids) > 2
                or node.op.condition
                or node.op.is_parameterized()
            ):
                for qid in qids:
                    if block_id[qid] > 0:
                        block_list[block_id[qid]].extend(pending_1q[qid])
                    block_id[qid] = -(qid + 1)
                    pending_1q[qid].clear()
                continue

            if len(qids) == 1:
                b_id = block_id[qids[0]]
                if b_id < 0:
                    pending_1q[qids[0]].append(node)
                else:
                    block_list[b_id].append(node)
            elif block_id[qids[0]] == block_id[qids[1]]:
                block_list[block_id[qids[0]]].append(node)
            else:
                block_id[qids[0]] = current_id
                block_id[qids[1]] = current_id
                new_block = list()
                if pending_1q[qids[0]]:
                    new_block.extend(pending_1q[qids[0]])
                    pending_1q[qids[0]].clear()
                if pending_1q[qids[1]]:
                    new_block.extend(pending_1q[qids[1]])
                    pending_1q[qids[1]].clear()
                new_block.append(node)
                block_list.append(new_block)
                current_id += 1

        self.property_set["block_list"] = [tuple(block) for block in block_list]
        return dag
