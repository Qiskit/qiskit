# -*- coding: utf-8 -*-

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

from qiskit.transpiler.basepasses import AnalysisPass


class Collect2qBlocks(AnalysisPass):
    """Collect sequences of uninterrupted gates acting on 2 qubits.

    Traverse the DAG and find blocks of gates that act consecutively on
    pairs of qubits. Write the blocks to propert_set as a dictionary
    of the form::

        {(q0, q1): [[g0, g1, g2], [g5]],
         (q0, q2): [[g3, g4]]
         ..
         .
        }

    Based on implementation by Andrew Cross.
    """

    def run(self, dag):
        """Run the Collect2qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological sort order
        such that all gates in a block act on the same pair of
        qubits and are adjacent in the circuit. the blocks are built
        by examining predecessors and successors of "cx" gates in
        the circuit. u1, u2, u3, cx, id gates will be included.

        After the execution, ``property_set['block_list']`` is set to
        a list of tuples of "op" node labels.
        """
        # Initiate the commutation set
        self.property_set['commutation_set'] = defaultdict(list)

        good_names = ["cx", "u1", "u2", "u3", "id"]
        block_list = []
        nodes = list(dag.topological_nodes())
        nodes_seen = dict(zip(nodes, [False] * len(nodes)))
        for nd in dag.topological_op_nodes():

            group = []
            # Explore predecessors and successors of cx gates
            if nd.name == "cx" and nd.condition is None and not nodes_seen[nd]:
                these_qubits = set(nd.qargs)
                # Explore predecessors of the "cx" node
                pred = list(dag.quantum_predecessors(nd))
                explore = True
                while explore:
                    pred_next = []
                    # If there is one predecessor, add it if it's on the right qubits
                    if len(pred) == 1 and not nodes_seen[pred[0]]:
                        pnd = pred[0]
                        if pnd.name in good_names and pnd.condition is None:
                            if (pnd.name == "cx" and set(pnd.qargs) == these_qubits) or \
                                    pnd.name != "cx" and not pnd.op.is_parameterized():
                                group.append(pnd)
                                nodes_seen[pnd] = True
                                pred_next.extend(dag.quantum_predecessors(pnd))
                    # If there are two, then we consider cases
                    elif len(pred) == 2:
                        # First, check if there is a relationship
                        if pred[0] in dag.predecessors(pred[1]):
                            sorted_pred = [pred[1]]   # was [pred[1], pred[0]]
                        elif pred[1] in dag.predecessors(pred[0]):
                            sorted_pred = [pred[0]]   # was [pred[0], pred[1]]
                        else:
                            # We need to avoid accidentally adding a cx on these_qubits
                            # since these must have a dependency through the other predecessor
                            # in this case
                            if pred[0].name == "cx" and set(pred[0].qargs) == these_qubits:
                                sorted_pred = [pred[1]]
                            elif pred[1].name == "cx" and set(pred[1].qargs) == these_qubits:
                                sorted_pred = [pred[0]]
                            else:
                                sorted_pred = pred
                        if len(sorted_pred) == 2 and sorted_pred[0].name == "cx" and \
                           sorted_pred[1].name == "cx":
                            break  # stop immediately if we hit a pair of cx
                        # Examine each predecessor
                        for pnd in sorted_pred:
                            if pnd.name not in good_names or pnd.condition is not None:
                                # remove any qubits that are interrupted by a gate
                                # e.g. a measure in the middle of the circuit
                                these_qubits = list(set(these_qubits) -
                                                    set(pnd.qargs))
                                continue
                            # If a predecessor is a single qubit gate, add it
                            if pnd.name != "cx" and not pnd.op.is_parameterized():
                                if not nodes_seen[pnd]:
                                    group.append(pnd)
                                    nodes_seen[pnd] = True
                                    pred_next.extend(dag.quantum_predecessors(pnd))
                            # If cx, check qubits
                            else:
                                pred_qubits = set(pnd.qargs)
                                if pred_qubits == these_qubits and pnd.condition is None:
                                    # add if on same qubits
                                    if not nodes_seen[pnd]:
                                        group.append(pnd)
                                        nodes_seen[pnd] = True
                                        pred_next.extend(dag.quantum_predecessors(pnd))
                                else:
                                    # remove qubit from consideration if not
                                    these_qubits = list(set(these_qubits) -
                                                        set(pred_qubits))
                    # Update predecessors
                    # Stop if there aren't any more
                    pred = list(set(pred_next))
                    if not pred:
                        explore = False
                # Reverse the predecessor list and append the "cx" node
                group.reverse()
                group.append(nd)
                nodes_seen[nd] = True
                # Reset these_qubits
                these_qubits = set(nd.qargs)
                # Explore successors of the "cx" node
                succ = list(dag.quantum_successors(nd))
                explore = True
                while explore:
                    succ_next = []
                    # If there is one successor, add it if its on the right qubits
                    if len(succ) == 1 and not nodes_seen[succ[0]]:
                        snd = succ[0]
                        if snd.name in good_names and snd.condition is None:
                            if (snd.name == "cx" and set(snd.qargs) == these_qubits) or \
                                    snd.name != "cx" and not snd.op.is_parameterized():
                                group.append(snd)
                                nodes_seen[snd] = True
                                succ_next.extend(dag.quantum_successors(snd))
                    # If there are two, then we consider cases
                    elif len(succ) == 2:
                        # First, check if there is a relationship
                        if succ[0] in dag.successors(succ[1]):
                            sorted_succ = [succ[1]]  # was [succ[1], succ[0]]
                        elif succ[1] in dag.successors(succ[0]):
                            sorted_succ = [succ[0]]  # was [succ[0], succ[1]]
                        else:
                            # We need to avoid accidentally adding a cx on these_qubits
                            # since these must have a dependency through the other successor
                            # in this case
                            if succ[0].name == "cx" and set(succ[0].qargs) == these_qubits:
                                sorted_succ = [succ[1]]
                            elif succ[1].name == "cx" and set(succ[1].qargs) == these_qubits:
                                sorted_succ = [succ[0]]
                            else:
                                sorted_succ = succ
                        if len(sorted_succ) == 2 and \
                           sorted_succ[0].name == "cx" and \
                           sorted_succ[1].name == "cx":
                            break  # stop immediately if we hit a pair of cx
                        # Examine each successor
                        for snd in sorted_succ:
                            if snd.name not in good_names or snd.condition is not None:
                                # remove qubits from consideration if interrupted
                                # by a gate e.g. a measure in the middle of the circuit
                                these_qubits = list(set(these_qubits) -
                                                    set(snd.qargs))
                                continue

                            # If a successor is a single qubit gate, add it
                            # NB as we have eliminated all gates with names not in
                            # good_names, this check guarantees they are single qubit
                            if snd.name != "cx" and not snd.op.is_parameterized():
                                if not nodes_seen[snd]:
                                    group.append(snd)
                                    nodes_seen[snd] = True
                                    succ_next.extend(dag.quantum_successors(snd))
                            else:
                                # If cx, check qubits
                                succ_qubits = set(snd.qargs)
                                if succ_qubits == these_qubits and snd.condition is None:
                                    # add if on same qubits
                                    if not nodes_seen[snd]:
                                        group.append(snd)
                                        nodes_seen[snd] = True
                                        succ_next.extend(dag.quantum_successors(snd))
                                else:
                                    # remove qubit from consideration if not
                                    these_qubits = list(set(these_qubits) -
                                                        set(succ_qubits))
                    # Update successors
                    # Stop if there aren't any more
                    succ = list(set(succ_next))
                    if not succ:
                        explore = False

                block_list.append(tuple(group))

        self.property_set['block_list'] = block_list

        return dag
