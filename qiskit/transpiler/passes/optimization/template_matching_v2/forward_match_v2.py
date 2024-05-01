# This code is part of Qiskit.
#
# (C) Copyright IBM 2020-2024.
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
match, a configuration of qubits and both circuit and template as inputs. The
result is a list of matches between the template and the circuit.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""

from qiskit.transpiler.passes.optimization.template_matching_v2.template_utils_v2 import (
    get_node,
    get_qindices,
    get_cindices,
    get_descendants,
    get_successors,
)

from qiskit.circuit.controlledgate import ControlledGate


class ForwardMatch:
    """
    Object to apply template matching in the forward direction.
    """

    def __init__(
        self,
        circuit_dag_dep,
        template_dag_dep,
        node_c,
        node_t,
        temp_match_class,
        qubits,
        clbits=None,
    ):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag_dep (DAGDependencyV2): circuit in the dag dependency form.
            template_dag_dep (DAGDependencyV2): template in the dag dependency form.
            node_c (DAGOpNode): node of first gate matched in the circuit.
            node_t (DAGOpNode): node of first gate matched in the template.
            qubits (list): list of considered qubits in the circuit.
            clbits (list): list of considered clbits in the circuit.
        """

        # The dag dependency representation of the circuit
        self.circuit_dag_dep = circuit_dag_dep

        # The dag dependency representation of the template
        self.template_dag_dep = template_dag_dep

        # List of qubits on which the node of the circuit is acting
        self.qubits = qubits

        # List of clbits on which the node of the circuit is acting
        self.clbits = clbits if clbits is not None else []

        # Id of the node in the circuit
        self.node_c = node_c

        # Id of the node in the template
        self.node_t = node_t

        # List of matches
        self.match = []

        # List of candidates for the forward match
        self.candidates = []

        # List of nodes in circuit which are matched
        self.matched_nodes_list = []

        # Transformation of the qarg indices of the circuit to be adapted to the template indices
        self.qarg_indices = []

        # Transformation of the carg indices of the circuit to be adapted to the template indices
        self.carg_indices = []

        # Class instance of TemplateMatching caller
        self.temp_match_class = temp_match_class

        # Dicts for storing lists of node ids
        self.successorstovisit = {}
        self.matchedwith = {}

        # Bool indicating if a node is blocked due to no match
        self.isblocked = {}

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

        temp_succs = get_successors(self.template_dag_dep, node_id_t)
        if temp_succs:
            maximal_index = temp_succs[-1]
            pred = [elem for elem in pred if elem <= maximal_index]

        block = []
        for node_id in pred:
            for succ in get_successors(self.template_dag_dep, node_id):
                if succ not in matches:
                    node = get_node(self.template_dag_dep, succ)
                    if node not in self.temp_match_class.descendants:
                        self.temp_match_class.descendants[node] = get_descendants(
                            self.template_dag_dep, succ
                        )
                    block = block + self.temp_match_class.descendants[node]
        self.candidates = list(set(temp_succs) - set(matches) - set(block))

    def _update_qarg_indices(self, qarg):
        """
        Change qubit indices of the current circuit node in order to
        be comparable with the indices of the template qubits list.
        Args:
            qarg (list): list of qubit indices from the circuit for a given node.
        """
        self.qarg_indices = []
        for q in qarg:
            if q in self.qubits:
                self.qarg_indices.append(self.qubits.index(q))
        if len(qarg) != len(self.qarg_indices):
            self.qarg_indices = []

    def _update_carg_indices(self, carg):
        """
        Change clbit indices of the current circuit node in order to
        be comparable with the indices of the template qubit list.
        Args:
            carg (list): list of clbit indices from the circuit for a given node.
        """
        self.carg_indices = []
        if carg:
            for q in carg:
                if q in self.clbits:
                    self.carg_indices.append(self.clbits.index(q))
            if len(carg) != len(self.carg_indices):
                self.carg_indices = []

    def _is_same_q_conf(self, node_circuit, node_template):
        """
        Check if the qubit configurations are compatible.
        Args:
            node_circuit (DAGOpNode): node in the circuit.
            node_template (DAGOpNode): node in the template.
        Returns:
            bool: True if possible, False otherwise.
        """

        node_temp_qind = get_qindices(self.template_dag_dep, node_template)
        if isinstance(node_circuit.op, ControlledGate):

            c_template = node_template.op.num_ctrl_qubits

            if c_template == 1:
                return self.qarg_indices == node_temp_qind

            else:
                control_qubits_template = node_temp_qind[:c_template]
                control_qubits_circuit = self.qarg_indices[:c_template]

                if set(control_qubits_circuit) == set(control_qubits_template):

                    target_qubits_template = node_temp_qind[c_template::]
                    target_qubits_circuit = self.qarg_indices[c_template::]

                    if node_template.op.base_gate.name in [
                        "rxx",
                        "ryy",
                        "rzz",
                        "swap",
                        "iswap",
                        "ms",
                    ]:
                        return set(target_qubits_template) == set(target_qubits_circuit)
                    else:
                        return target_qubits_template == target_qubits_circuit
                else:
                    return False
        else:
            if node_template.op.name in ["rxx", "ryy", "rzz", "swap", "iswap", "ms"]:
                return set(self.qarg_indices) == set(
                    get_qindices(self.template_dag_dep, node_template)
                )
            else:
                return self.qarg_indices == node_temp_qind

    def _is_same_c_conf(self, node_circuit, node_template):
        """
        Check if the clbit configurations are compatible.
        Args:
            node_circuit (DAGOpNode): node in the circuit.
            node_template (DAGOpNode): node in the template.
        Returns:
            bool: True if possible, False otherwise.
        """
        if (
            getattr(node_circuit.op, "condition", None)
            and node_template.type == "op"
            and getattr(node_template.op, "condition", None)
        ):
            if set(self.carg_indices) != set(node_template.cindices):
                return False
            if (
                getattr(node_circuit.op, "condition", None)[1]
                != getattr(node_template.op, "condition", None)[1]
            ):
                return False
        return True

    def run_forward_match(self):
        """
        Apply the forward match algorithm and return the list of matches given an initial match
        and a circuit qubit configuration.
        """

        # Initialize certain attributes
        for node in self.circuit_dag_dep.topological_nodes():
            self.successorstovisit[node] = get_successors(self.circuit_dag_dep, node._node_id)

        self.matchedwith[self.node_c] = [self.node_t._node_id]
        self.matchedwith[self.node_t] = [self.node_c._node_id]

        # Initialize the list of matches and the stack of matched nodes (circuit)
        self.match.append([self.node_t._node_id, self.node_c._node_id])
        self.matched_nodes_list.append(get_node(self.circuit_dag_dep, self.node_c._node_id))

        # While the list of matched nodes is not empty
        while self.matched_nodes_list:

            first_matched = self.matched_nodes_list.pop(0)
            if self.successorstovisit[first_matched] == []:
                continue

            # Get the id and the node of the first successor to visit
            trial_successor_id = self.successorstovisit[first_matched].pop(0)
            trial_successor = get_node(self.circuit_dag_dep, trial_successor_id)

            # Update the matched_nodes_list with new attribute successor to visit and sort the list.
            self.matched_nodes_list.append(first_matched)
            self.matched_nodes_list.sort(key=lambda x: self.successorstovisit[x])

            # If the node is blocked and already matched go to the end
            if self.isblocked.get(trial_successor) or self.matchedwith.get(trial_successor):
                continue

            # Search for potential candidates in the template
            self._find_forward_candidates(self.matchedwith[first_matched][0])

            qarg1 = get_qindices(
                self.circuit_dag_dep, get_node(self.circuit_dag_dep, trial_successor_id)
            )
            carg1 = get_cindices(
                self.circuit_dag_dep, get_node(self.circuit_dag_dep, trial_successor_id)
            )

            # Update the indices for both qubits and clbits in order to be comparable with the
            # indices in the template circuit.
            self._update_qarg_indices(qarg1)
            self._update_carg_indices(carg1)

            match = False

            # For loop over the candidates (template) to find a match.
            for i in self.candidates:

                # Break the for loop if a match is found.
                if match:
                    break

                node_circuit = get_node(self.circuit_dag_dep, trial_successor_id)
                node_template = get_node(self.template_dag_dep, i)

                # Compare the indices of qubits and the operation name.
                # Necessary but not sufficient conditions for a match to happen.
                node_temp_qind = get_qindices(self.template_dag_dep, node_template)
                if (
                    len(self.qarg_indices) != len(node_temp_qind)
                    or set(self.qarg_indices) != set(node_temp_qind)
                    or node_circuit.name != node_template.name
                ):
                    continue

                # Check if the qubit, clbit configurations are compatible for a match,
                # also check if the operations are the same.
                if (
                    self._is_same_q_conf(node_circuit, node_template)
                    and self._is_same_c_conf(node_circuit, node_template)
                    and node_circuit.op.soft_compare(node_template.op)
                ):
                    self.matchedwith[trial_successor] = [i]
                    self.matchedwith[node_template] = [trial_successor_id]

                    # Append the new match to the list of matches.
                    self.match.append([i, trial_successor_id])

                    # Potential successors to visit (circuit) for a given match.
                    potential = get_successors(self.circuit_dag_dep, trial_successor_id)

                    # If the potential successors to visit are blocked or matched, it is removed.
                    for potential_id in potential:
                        if self.isblocked.get(get_node(self.circuit_dag_dep, potential_id)) or (
                            self.matchedwith.get(get_node(self.circuit_dag_dep, potential_id))
                        ):
                            potential.remove(potential_id)

                    sorted_potential = sorted(potential)

                    #  Update the successor to visit attribute
                    self.successorstovisit[trial_successor] = sorted_potential

                    # Add the updated node to the stack.
                    self.matched_nodes_list.append(trial_successor)
                    self.matched_nodes_list.sort(key=lambda x: self.successorstovisit[x])
                    match = True
                    continue

            # If no match is found, block the node and all the descendants.
            if not match:
                self.isblocked[trial_successor] = True
                if trial_successor not in self.temp_match_class.descendants:
                    self.temp_match_class.descendants[trial_successor] = get_descendants(
                        self.circuit_dag_dep, trial_successor_id
                    )

                for desc_id in self.temp_match_class.descendants[trial_successor]:
                    desc = get_node(self.circuit_dag_dep, desc_id)
                    self.isblocked[desc] = True
                    if self.matchedwith.get(desc):
                        match_id = self.matchedwith[desc][0]
                        self.match.remove([match_id, desc])
                        self.matchedwith[get_node(self.template_dag_dep, match_id)] = []
                        self.matchedwith[desc] = []

        return (self.matchedwith, self.isblocked)
