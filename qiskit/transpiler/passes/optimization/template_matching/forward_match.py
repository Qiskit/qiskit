# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Template matching in the forward direction, it takes an initial
match, a configuration of qubit and both circuit and template as inputs. The
result is a list of match between the template and the circuit.


**Reference:**

[1] Iten, R., Sutter, D. and Woerner, S., 2019.
Efficient template matching in quantum circuits.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""


class ForwardMatch:
    """
    Object to apply template matching in the forward direction.
    """

    def __init__(self, circuit_dag, template_dag, node_id_c, node_id_t, qubits, clbits=None):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            template_dag (DAGDependency): template in the dag dependency form.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_t (int): index of the first gate matched in the template.
            qubits (list): list of considered qubits in the circuit.
            clbits (list): list of considered clbits in the circuit.
        """

        # The dag depdendency representation of the circuit
        self.circuit_dag = circuit_dag.copy()

        # The dag dependency representation of the template
        self.template_dag = template_dag.copy()

        # List of qubit on which the node of the circuit is acting on
        self.qubits = qubits

        # List of qubit on which the node of the circuit is acting on
        self.clbits = clbits if clbits is not None else []

        # Id of the node in the circuit
        self.node_id_c = node_id_c

        # Id of the node in the template
        self.node_id_t = node_id_t

        # List of match
        self.match = []

        # List of candidates for the forward match
        self.candidates = []

        # List of nodes in circuit which are matched
        self.matched_nodes_list = []

        # Transformation of the qarg indices of the circuit to be adapted to the template indices
        self.qarg_indices = []

        # Transformation of the carg indices of the circuit to be adapted to the template indices
        self.carg_indices = []

    def _init_successors_to_visit(self):
        """
        Initialize the attribute list 'SuccessorsToVisit'
        """
        for i in range(0, self.circuit_dag.size()):
            if i == self.node_id_c:
                self.circuit_dag.get_node(i).successorstovisit = \
                    self.circuit_dag.direct_successors(i)

    def _init_matched_with_circuit(self):
        """
        Initialize the attribute 'MatchedWith' in the template DAG dependency.
        """
        for i in range(0, self.circuit_dag.size()):
            if i == self.node_id_c:
                self.circuit_dag.get_node(i).matchedwith = [self.node_id_t]
            else:
                self.circuit_dag.get_node(i).matchedwith = []

    def _init_matched_with_template(self):
        """
        Initialize the attribute 'MatchedWith' in the circuit DAG dependency.
        """
        for i in range(0, self.template_dag.size()):
            if i == self.node_id_t:
                self.template_dag.get_node(i).matchedwith = [self.node_id_c]
            else:
                self.template_dag.get_node(i).matchedwith = []

    def _init_is_blocked_circuit(self):
        """
        Initialize the attribute 'IsBlocked' in the circuit DAG dependency.
        """
        for i in range(0, self.circuit_dag.size()):
            self.circuit_dag.get_node(i).isblocked = False

    def _init_is_blocked_template(self):
        """
        Initialize the attribute 'IsBlocked' in the template DAG dependency.
        """
        for i in range(0, self.template_dag.size()):
            self.template_dag.get_node(i).isblocked = False

    def _init_list_match(self):
        """
        Initialize the list of matched nodes between the circuit and the template
        with the first match found.
        """
        self.match.append([self.node_id_t, self.node_id_c])

    def _find_forward_candidates(self, node_id_t):
        """
        Find the candidate nodes to be matched in the template for a given node.
        Args:
            node_id_t (int): considered node id.
        """
        matches = []

        for i in range(0, len(self.match)):
            matches.append(self.match[i][0])

        pred = matches.copy()
        if len(pred) > 1:
            pred.sort()
        pred.remove(node_id_t)

        if self.template_dag.direct_successors(node_id_t):
            maximal_index = self.template_dag.direct_successors(node_id_t)[-1]
            for elem in pred:
                if elem > maximal_index:
                    pred.remove(elem)

        block = []
        for node_id in pred:
            for dir_succ in self.template_dag.direct_successors(node_id):
                if dir_succ not in matches:
                    succ = self.template_dag.successors(dir_succ)
                    block = block + succ
        self.candidates = list(set(self.template_dag.direct_successors(node_id_t))
                               - set(matches) - set(block))

    def _init_matched_nodes(self):
        """
        Initialize the list of current matched nodes.
        """
        self.matched_nodes_list.append([self.node_id_c, self.circuit_dag.get_node(self.node_id_c)])

    def _get_node_forward(self, list_id):
        """
        Return a node from the matched_node_list for a given list id.
        Args:
            list_id (int): considered list id of the desired node.

        Returns:
            DAGDepNode: DAGDepNode object corresponding to i-th node of the matched_node_list.
        """
        node = self.matched_nodes_list[list_id][1]
        return node

    def _remove_node_forward(self, list_id):
        """
        Remove a node of the current matched list for a given list id.
        Args:
            list_id (int): considered list id of the desired node.
        """
        self.matched_nodes_list.pop(list_id)

    def _update_successor(self, node, successor_id):
        """
        Return a node with an updated attribute 'SuccessorToVisit'.
        Args:
            node (DAGDepNode): current node.
            successor_id (int): successor id to remove.

        Returns:
            DAGNode: DAGNode with updated attribute 'SuccessorToVisit'.
        """
        node_update = node
        node_update.successorstovisit.pop(successor_id)
        return node_update

    def _get_successors_to_visit(self, node, list_id):
        """
        Return the successor for a given node and id.
        Args:
            node (DAGNode): current node.
            list_id (int): id in the list for the successor to get.

        Returns:
            int: id of the successor to get.
        """
        successor_id = node.successorstovisit[list_id]
        return successor_id

    def _update_qarg_indices(self, qarg):
        """
        Change qubits indices of the current circuit node in order to
        be comparable with the indices of the template qubits list.
        Args:
            qarg (list): list of qubits indices from the circuit for a given node.
        """
        self.qarg_indices = []
        for q in qarg:
            if q in self.qubits:
                self.qarg_indices.append(self.qubits.index(q))
        if len(qarg) != len(self.qarg_indices):
            self.qarg_indices = []

    def _update_carg_indices(self, carg):
        """
        Change clbits indices of the current circuit node in order to
        be comparable with the indices of the template qubits list.
        Args:
            carg (list): list of clbits indices from the circuit for a given node.
        """
        self.carg_indices = []
        if carg:
            for q in carg:
                if q in self.clbits:
                    self.carg_indices.append(self.clbits.index(q))
            if len(carg) != len(self.carg_indices):
                self.carg_indices = []

    def run_forward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubits configuration.
        """

        # Initialize the new attributes
        self._init_successors_to_visit()

        self._init_matched_with_circuit()
        self._init_matched_with_template()

        self._init_is_blocked_circuit()
        self._init_is_blocked_template()

        self._init_list_match()

        self._init_matched_nodes()

        # while over the list of matches to be checked
        while self.matched_nodes_list:

            # Return first element of the matched_nodes_list and removes it from the list
            v_first = self._get_node_forward(0)
            self._remove_node_forward(0)

            # If no successors to visit go to the end
            if not v_first.successorstovisit:
                continue

            # Get the label and the node of the first successor
            label = self._get_successors_to_visit(v_first, 0)
            v = [label, self.circuit_dag.get_node(label)]

            # Update of the SuccessorsToVisit attribute
            v_first = self._update_successor(v_first, 0)

            # Update the matched_nodes_list

            self.matched_nodes_list.append([v_first.node_id, v_first])
            self.matched_nodes_list.sort(key=lambda x: x[1].successorstovisit)

            # If the node is blocked and already matched go to the end
            if v[1].isblocked | (v[1].matchedwith != []):
                continue

            # Search for potential candidates in the template
            self._find_forward_candidates(v_first.matchedwith[0])

            # Get the list of qubit on which the node operation is acting on
            # (circuit) and update the indices

            qarg1 = self.circuit_dag.get_node(label).qindices
            carg1 = self.circuit_dag.get_node(label).cindices

            self._update_qarg_indices(qarg1)
            self._update_carg_indices(carg1)

            # Loop over the list of candidates
            match = False

            for i in self.candidates:

                # Break the for loop if a match is found
                if match:
                    break

                # Get the list of qubit on which the node operation is acting (template)
                qarg2 = self.template_dag.get_node(i).qindices
                carg2 = self.template_dag.get_node(i).cindices

                # Compare the indices of qubits and the operation,
                # if both are True; a match is found
                if (set(self.qarg_indices) == set(qarg2)) and (self.qarg_indices[-1] == qarg2[-1])\
                        and (self.circuit_dag.get_node(label).op ==
                             self.template_dag.get_node(i).op) \
                        and (set(self.carg_indices) == set(carg2)):
                    if self.circuit_dag.get_node(label).condition or \
                            self.template_dag.get_node(i).condition:
                        if self.circuit_dag.get_node(label).condition[1] != \
                                self.template_dag.get_node(i).condition[1]:
                            continue
                    else:
                        v[1].matchedwith = [i]

                        self.template_dag.get_node(i).matchedwith = [label]

                        self.match.append([i, label])

                        potential = self.circuit_dag.direct_successors(label)

                        for potential_id in potential:
                            if self.circuit_dag.get_node(potential_id).isblocked | \
                                    (self.circuit_dag.get_node(potential_id).matchedwith != []):
                                potential.remove(potential_id)

                        sorted_potential = sorted(potential)
                        v[1].successorstovisit = sorted_potential

                        self.matched_nodes_list.append([v[0], v[1]])
                        self.matched_nodes_list.sort(key=lambda x: x[1].successorstovisit)
                        match = True
                        continue

            # If no match is found, block the node and all the successors
            if not match:
                v[1].isblocked = True
                for succ in v[1].successors:
                    self.circuit_dag.get_node(succ).isblocked = True
                    if self.circuit_dag.get_node(succ).matchedwith:
                        self.match.remove([self.circuit_dag.get_node(succ).matchedwith[0], succ])
                        match_id = self.circuit_dag.get_node(succ).matchedwith[0]
                        self.template_dag.get_node(match_id).matchedwith = []
                        self.circuit_dag.get_node(succ).matchedwith = []
