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

from qiskit.circuit import Gate
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
        by examining predecessors and successors of 2q gates in
        the circuit.

        After the execution, ``property_set['block_list']`` is set to
        a list of tuples of "op" node labels.
        """
        # Initiate the commutation set
        self.property_set['commutation_set'] = defaultdict(list)

        block_list = []
        nodes = list(dag.topological_nodes())
        nodes_seen = dict(zip(nodes, [False] * len(nodes)))
        for nd in dag.topological_op_nodes():

            group = []
            # Explore predecessors and successors of 2q gates
            if (  # pylint: disable=too-many-boolean-expressions
                    nd.type == 'op'
                    and isinstance(nd.op, Gate)
                    and len(nd.qargs) == 2
                    and not nodes_seen[nd]
                    and nd.condition is None
                    and not nd.op.is_parameterized()
            ):
                these_qubits = set(nd.qargs)
                # Explore predecessors of the 2q node
                pred = list(dag.quantum_predecessors(nd))
                explore = True
                while explore:
                    pred_next = []
                    # If there is one predecessor, add it if it's on the right qubits
                    if len(pred) == 1 and not nodes_seen[pred[0]]:
                        pnd = pred[0]
                        if (
                                pnd.type == 'op'
                                and isinstance(pnd.op, Gate)
                                and len(pnd.qargs) <= 2
                                and pnd.condition is None
                                and not pnd.op.is_parameterized()
                        ):
                            if (len(pnd.qargs) == 2 and set(pnd.qargs) == these_qubits) \
                               or len(pnd.qargs) == 1:
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
                            # We need to avoid accidentally adding a 2q gate on these_qubits
                            # since these must have a dependency through the other predecessor
                            # in this case
                            if len(pred[0].qargs) == 2 and set(pred[0].qargs) == these_qubits:
                                sorted_pred = [pred[1]]
                            elif (len(pred[1].qargs) == 1
                                  and set(pred[1].qargs) == these_qubits):
                                sorted_pred = [pred[0]]
                            else:
                                sorted_pred = pred
                        if len(sorted_pred) == 2 and len(sorted_pred[0].qargs) == 2 and \
                           len(sorted_pred[1].qargs) == 2:
                            break  # stop immediately if we hit a pair of 2q gates
                        # Examine each predecessor
                        for pnd in sorted_pred:
                            if (
                                    pnd.type != 'op'
                                    or not isinstance(pnd.op, Gate)
                                    or len(pnd.qargs) > 2
                                    or pnd.condition is not None
                                    or pnd.op.is_parameterized()
                            ):
                                # remove any qubits that are interrupted by a gate
                                # e.g. a measure in the middle of the circuit
                                these_qubits = list(set(these_qubits) -
                                                    set(pnd.qargs))
                                continue
                            # If a predecessor is a single qubit gate, add it
                            if len(pnd.qargs) == 1 and not pnd.op.is_parameterized():
                                if not nodes_seen[pnd]:
                                    group.append(pnd)
                                    nodes_seen[pnd] = True
                                    pred_next.extend(dag.quantum_predecessors(pnd))
                            # If 2q, check qubits
                            else:
                                pred_qubits = set(pnd.qargs)
                                if (
                                        pred_qubits == these_qubits
                                        and pnd.condition is None
                                        and not pnd.op.is_parameterized()
                                ):
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
                # Reverse the predecessor list and append the 2q node
                group.reverse()
                group.append(nd)
                nodes_seen[nd] = True
                # Reset these_qubits
                these_qubits = set(nd.qargs)
                # Explore successors of the 2q node
                succ = list(dag.quantum_successors(nd))
                explore = True
                while explore:
                    succ_next = []
                    # If there is one successor, add it if its on the right qubits
                    if len(succ) == 1 and not nodes_seen[succ[0]]:
                        snd = succ[0]
                        if (
                                snd.type == 'op'
                                and isinstance(snd.op, Gate)
                                and len(snd.qargs) <= 2
                                and snd.condition is None
                                and not snd.op.is_parameterized()
                        ):
                            if (len(snd.qargs) == 2 and set(snd.qargs) == these_qubits) or \
                                    len(snd.qargs) == 1:
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
                            # We need to avoid accidentally adding a 2q gate on these_qubits
                            # since these must have a dependency through the other successor
                            # in this case
                            if (len(succ[0].qargs) == 2
                                    and set(succ[0].qargs) == these_qubits):
                                sorted_succ = [succ[1]]
                            elif (len(succ[1].qargs) == 2
                                  and set(succ[1].qargs) == these_qubits):
                                sorted_succ = [succ[0]]
                            else:
                                sorted_succ = succ
                        if len(sorted_succ) == 2 and \
                           len(sorted_succ[0].qargs) == 2 and \
                           len(sorted_succ[1].qargs) == 2:
                            break  # stop immediately if we hit a pair of 2q gates
                        # Examine each successor
                        for snd in sorted_succ:
                            if (
                                    snd.type != 'op'
                                    or not isinstance(snd.op, Gate)
                                    or len(snd.qargs) > 2
                                    or snd.condition is not None
                                    or snd.op.is_parameterized()
                            ):
                                # remove qubits from consideration if interrupted
                                # by a gate e.g. a measure in the middle of the circuit
                                these_qubits = list(set(these_qubits) -
                                                    set(snd.qargs))
                                continue

                            # If a successor is a single qubit gate, add it
                            # NB as we have eliminated all gates with names not in
                            # good_names, this check guarantees they are single qubit
                            if len(snd.qargs) == 1 and not snd.op.is_parameterized():
                                if not nodes_seen[snd]:
                                    group.append(snd)
                                    nodes_seen[snd] = True
                                    succ_next.extend(dag.quantum_successors(snd))
                            else:
                                # If 2q, check qubits
                                succ_qubits = set(snd.qargs)
                                if (
                                        succ_qubits == these_qubits
                                        and snd.condition is None
                                        and not snd.op.is_parameterized()
                                ):
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
