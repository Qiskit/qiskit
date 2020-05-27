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
Template matching in the backward direction, it takes an initial match, a
configuration of qubit, both circuit and template as inputs and the list
obtained from forward match. The result is a list of matches between the
template and the circuit.


**Reference:**

[1] Iten, R., Sutter, D. and Woerner, S., 2019.
Efficient template matching in quantum circuits.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""

from qiskit.circuit.controlledgate import ControlledGate


class Match:
    """
    Object to represent a match and its qubit configurations.
    """

    def __init__(self, match, qubit, clbit):
        """
        Create a Match class with necessary arguments.
        Args:
            match (list): list of matched gates.
            qubit (list): list of qubits configuration.
            clbit (list): list of clbits configuration.

        """
        # Match list
        self.match = match
        # Qubits list for circuit
        self.qubit = [qubit]
        # Clbits for template
        self.clbit = [clbit]


class MatchingScenarios:
    """
    Class to represent a matching scenario.
    """

    def __init__(self, circuit_matched,
                 circuit_blocked,
                 template_matched,
                 template_blocked,
                 matches,
                 counter):
        """
        Create a MatchingScenarios class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            template_dag (DAGDependency): template in the dag dependency form.
            matches (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """
        # DAGDependency object: circuit
        self.circuit_matched = circuit_matched

        self.template_matched = template_matched

        self.circuit_blocked = circuit_blocked
        # DAGDependency object: template
        self.template_blocked = template_blocked
        # List of matches
        self.matches = matches
        # Counter
        self.counter = counter


class MatchingScenariosList:
    """
    Object to define a list of MatchingScenarios, with method to append
    and pop elements.
    """

    def __init__(self):
        """
        Create an empty MatchingScenariosList.
        """
        # List
        self.matching_scenarios_list = []

    def append_scenario(self, matching):
        """
        Append a scenario to the list.
        Args:
            matching (MatchingScenarios): a scenario of match.
        """
        self.matching_scenarios_list.append(matching)

    def pop_scenario(self):
        """
        Pop the first scenario of the list.
        Returns:
            MatchingScenarios: a scenario of match.
        """
        # Pop the first MatchingScenario and returns it
        first = self.matching_scenarios_list[0]
        self.matching_scenarios_list.pop(0)
        return first


class BackwardMatch:
    """
    Class BackwardMatch allows to run backward direction part of template
    matching algorithm.
    """

    def __init__(self, circuit_dag, template_dag, forward_matches
                 , node_id_c, node_id_t, qubits, clbits=None):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            template_dag (DAGDependency): template in the dag dependency form.
            forward_matches (list): list of match obtained in the forward direction.
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

        self.forward_matches = forward_matches

        self.match_final = []

    def _gate_indices(self):
        """
        Function which returns the list of gates that are not match and not
        blocked for the first scenario.
        Returns:
            list: list of gate id.
        """
        gate_indices = []

        current_dag = self.circuit_dag

        for node in current_dag.get_nodes():
            if (not node.matchedwith) and (not node.isblocked):
                gate_indices.append(node.node_id)
        gate_indices.reverse()
        return gate_indices

    def _find_backward_candidates(self, template_blocked, matches):
        """
        Function which returns the list possible backward candidates in the template dag.
        Args:
            template_dag (DAGDependency): dag representation of the template circuit.
            matches (list): list of matches.
        Returns:
            list: list of backward candidates (id).
        """
        template_block = []

        for node_id in range(self.node_id_t, self.template_dag.size()):
            if template_blocked[node_id]:
                template_block.append(node_id)

        matches_template = sorted([match[0] for match in matches])

        successors = self.template_dag.get_node(self.node_id_t).successors
        potential = []
        for index in range(self.node_id_t + 1, self.template_dag.size()):
            if (index not in successors) and (index not in template_block):
                potential.append(index)

        candidates_indices = list(set(potential) - set(matches_template))
        candidates_indices = sorted(candidates_indices)
        candidates_indices.reverse()

        return candidates_indices

    def _update_qarg_indices(self, qarg):
        """
        Change qubits indices of the current circuit node in order to
        be comparable the indices of the template qubits list.
        Args:
            qarg (list): list of qubits indices from the circuit for a given gate.
        Returns:
            list: circuit indices update for qubits.
        """
        qarg_indices = []
        for q in qarg:
            if q in self.qubits:
                qarg_indices.append(self.qubits.index(q))
        if len(qarg) != len(qarg_indices):
            qarg_indices = []
        return qarg_indices

    def _update_carg_indices(self, carg):
        """
        Change clbits indices of the current circuit node in order to
        be comparable the indices of the template qubits list.
        Args:
            carg (list): list of clbits indices from the circuit for a given gate.
        Returns:
            list: circuit indices update for clbits.
        """
        carg_indices = []
        if carg:
            for q in carg:
                if q in self.clbits:
                    carg_indices.append(self.clbits.index(q))
            if len(carg) != len(carg_indices):
                carg_indices = []
        return carg_indices

    def _is_same_op(self, node_circuit, node_template):
        """
        Check if two instructions are the same.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
        Returns:
            bool: True if the same, False otherwise.
        """
        return node_circuit.op == node_template.op

    def _is_same_q_conf(self, node_circuit, node_template, qarg_circuit):
        """
        Check if the qubits configurations are compatible.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
            qarg_circuit (list): qubits configuration for the Instruction in the circuit.
        Returns:
            bool: True if possible, False otherwise.
        """

        if isinstance(node_circuit.op, ControlledGate):

            c_template = node_template.op.num_ctrl_qubits

            if c_template == 1:
                return qarg_circuit == node_template.qindices

            else:
                control_qubits_template = node_template.qindices[:c_template]
                control_qubits_circuit = qarg_circuit[:c_template]

                if set(control_qubits_circuit) == set(control_qubits_template):

                    target_qubits_template = node_template.qindices[c_template::]
                    target_qubits_circuit = qarg_circuit[c_template::]

                    if node_template.op.base_gate.name\
                            in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
                        return set(target_qubits_template) == set(target_qubits_circuit)
                    else:
                        return target_qubits_template == target_qubits_circuit
                else:
                    return False
        else:
            if node_template.op.name in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
                return set(qarg_circuit) == set(node_template.qindices)
            else:
                return qarg_circuit == node_template.qindices

    def _is_same_c_conf(self, node_circuit, node_template, carg_circuit):
        """
        Check if the clbits configurations are compatible.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
            carg_circuit (list): clbits configuration for the Instruction in the circuit.
        Returns:
            bool: True if possible, False otherwise.
        """
        if node_circuit.condition and node_template.conditon:
            if set(carg_circuit) != set(node_template.cindices):
                return False
            if node_circuit.condition[1] != node_template.conditon[1]:
                return False
        return True

    def _init_matched_blocked_list(self):
        circuit_matched = []
        circuit_blocked = []

        for node in self.circuit_dag.get_nodes():
            circuit_matched.append(node.matchedwith)
            circuit_blocked.append(node.isblocked)

        template_matched = []
        template_blocked = []

        for node in self.template_dag.get_nodes():
            template_matched.append(node.matchedwith)
            template_blocked.append(node.isblocked)

        return circuit_matched, circuit_blocked, template_matched, template_blocked

    def run_backward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubits configuration.

        """
        match_store_list = []

        counter = 1

        circuit_matched, circuit_blocked, template_matched, template_blocked = self._init_matched_blocked_list()

        if self.qubits == [6,5,4,3,2] and self.node_id_t==0 and self.node_id_c==7:
            print(self.forward_matches)
            print(circuit_matched, circuit_blocked, template_matched, template_blocked)

        first_match = MatchingScenarios(circuit_matched,
                                        circuit_blocked,
                                        template_matched,
                                        template_blocked,
                                        self.forward_matches,
                                        counter)

        matching_list = MatchingScenariosList()
        matching_list.append_scenario(first_match)

        gate_indices = self._gate_indices()

        number_of_gate_to_match = self.template_dag.size() - \
                                  (self.node_id_t - 1) - len(self.forward_matches)

        tree = 0

        while matching_list.matching_scenarios_list:

            scenario = matching_list.pop_scenario()

            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked

            template_matched = scenario.template_matched
            template_blocked = scenario.template_blocked

            matches_scenario = scenario.matches

            counter_scenario = scenario.counter

            match_backward = [match for match in matches_scenario
                              if match not in self.forward_matches]

            if counter_scenario > len(gate_indices) or \
                    len(match_backward) == number_of_gate_to_match:
                matches_scenario.sort(key=lambda x: x[0])
                match_store_list.append(Match(matches_scenario, self.qubits, self.clbits))
                continue

            circuit_id = gate_indices[counter_scenario - 1]
            node_circuit = self.circuit_dag.get_node(circuit_id)

            if circuit_blocked[circuit_id]:
                matching_scenario = MatchingScenarios(circuit_matched,
                                                      circuit_blocked,
                                                      template_matched,
                                                      template_blocked,
                                                      matches_scenario,
                                                      counter_scenario + 1)
                matching_list.append_scenario(matching_scenario)
                continue

            candidates_indices = self._find_backward_candidates(template_blocked,
                                                                matches_scenario)

            qarg1 = node_circuit.qindices
            carg1 = node_circuit.cindices

            qarg1 = self._update_qarg_indices(qarg1)
            carg1 = self._update_carg_indices(carg1)

            global_match = False

            for template_id in candidates_indices:
                actual_match = False

                node_template = self.template_dag.get_node(template_id)
                qarg2 = self.template_dag.get_node(template_id).qindices

                if len(qarg1) != len(qarg2) \
                        or set(qarg1) != set(qarg2)\
                        or node_circuit.name != node_template.name:
                    continue

                if self._is_same_q_conf(node_circuit, node_template, qarg1)\
                        and self._is_same_c_conf(node_circuit, node_template, carg1)\
                        and self._is_same_op(node_circuit, node_template):

                    tree = tree + 1

                    circuit_matched_match = circuit_matched.copy()
                    circuit_blocked_match = circuit_blocked.copy()

                    template_matched_match = template_matched.copy()
                    template_blocked_match = template_blocked.copy()

                    matches_scenario_match = matches_scenario.copy()

                    block_list = []
                    broken_matches_match = []

                    for potential_block in self.template_dag.successors(template_id):
                        if not template_matched_match[potential_block]:
                            template_blocked_match[potential_block] = True
                            block_list.append(potential_block)
                            for block_id in block_list:
                                for succ_id in \
                                        self.template_dag.successors(block_id):
                                    template_blocked_match[succ_id] = True
                                    if template_matched_match[succ_id]:
                                        new_id = \
                                            template_matched_match[succ_id][0]
                                        circuit_matched_match[new_id] = []
                                        template_matched_match[succ_id] = []
                                        broken_matches_match.append(succ_id)

                    new_matches_scenario_match = [elem for elem in matches_scenario_match
                                              if elem[0] not in broken_matches_match]

                    condition = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario_match:
                            condition = False
                            break

                    if ([self.node_id_t, self.node_id_c] in new_matches_scenario_match) \
                            and (condition or not match_backward):

                        template_matched_match[template_id] = [circuit_id]
                        circuit_matched_match[circuit_id] = [template_id]
                        new_matches_scenario_match.append([template_id, circuit_id])

                        new_matching_scenario = MatchingScenarios(circuit_matched_match,
                                                                  circuit_blocked_match,
                                                                  template_matched_match,
                                                                  template_blocked_match,
                                                                  new_matches_scenario_match,
                                                                  counter_scenario + 1)
                        matching_list.append_scenario(new_matching_scenario)

                        actual_match = True
                        global_match = True

                    if actual_match:
                        circuit_matched_block_s = circuit_matched.copy()
                        circuit_blocked_block_s = circuit_blocked.copy()

                        template_matched_block_s = template_matched.copy()
                        template_blocked_block_s = template_blocked.copy()

                        matches_scenario_block_s = matches_scenario.copy()

                        circuit_blocked_block_s[circuit_id] = True

                        broken_matches = []

                        for succ in self.circuit_dag.get_node(circuit_id).successors:
                            circuit_blocked_block_s[succ] = True
                            if circuit_matched_block_s[succ]:
                                broken_matches.append(succ)
                                new_id = circuit_matched_block_s[succ][0]
                                template_matched_block_s[new_id] = []
                                circuit_matched_block_s[succ] = []

                        new_matches_scenario_block_s = [elem for elem in matches_scenario_block_s
                                                        if elem[1] not in broken_matches]

                        condition_not_greedy = True

                        for back_match in match_backward:
                            if back_match not in new_matches_scenario_block_s:
                                condition_not_greedy = False
                                break

                        if ([self.node_id_t, self.node_id_c] in new_matches_scenario_block_s) and \
                                (condition_not_greedy or not match_backward):
                            new_matching_scenario = MatchingScenarios(circuit_matched_block_s,
                                                                      circuit_blocked_block_s,
                                                                      template_matched_block_s,
                                                                      template_blocked_block_s,
                                                                      new_matches_scenario_block_s,
                                                                      counter_scenario + 1)
                            matching_list.append_scenario(new_matching_scenario)

                        if broken_matches:

                            circuit_matched_block_p = circuit_matched.copy()
                            circuit_blocked_block_p = circuit_blocked.copy()

                            template_matched_block_p = template_matched.copy()
                            template_blocked_block_p = template_blocked.copy()

                            matches_scenario_block_p = matches_scenario.copy()

                            circuit_blocked_block_p[circuit_id] = True

                            for pred in self.circuit_dag.get_node(circuit_id).predecessors:
                                circuit_blocked_block_p[pred] = True

                            matching_scenario = MatchingScenarios(circuit_matched_block_p,
                                                                  circuit_blocked_block_p,
                                                                  template_matched_block_p,
                                                                  template_blocked_block_p,
                                                                  matches_scenario_block_p,
                                                                  counter_scenario + 1)
                            matching_list.append_scenario(matching_scenario)

            if not global_match:

                tree = tree + 1

                circuit_blocked[circuit_id] = True

                following_matches = []

                successors = self.circuit_dag.get_node(circuit_id).successors
                for succ in successors:
                    if circuit_matched[succ]:
                        following_matches.append(succ)

                predecessors = self.circuit_dag.get_node(circuit_id).predecessors

                if not predecessors or not following_matches:

                    matching_scenario = MatchingScenarios(circuit_matched,
                                                          circuit_blocked,
                                                          template_matched,
                                                          template_blocked,
                                                          matches_scenario,
                                                          counter_scenario + 1)
                    matching_list.append_scenario(matching_scenario)

                else:
                    # Option 1
                    for pred in predecessors:
                        circuit_blocked[pred] = True

                    matching_scenario = MatchingScenarios(circuit_matched,
                                                          circuit_blocked,
                                                          template_matched,
                                                          template_blocked,
                                                          matches_scenario,
                                                          counter_scenario + 1)
                    matching_list.append_scenario(matching_scenario)

                    # Option 2
                    circuit_matched_nomatch = circuit_matched.copy()
                    circuit_blocked_nomatch = circuit_blocked.copy()

                    template_matched_nomatch = template_matched.copy()
                    template_blocked_nomatch = template_blocked.copy()

                    matches_scenario_nomatch = matches_scenario.copy()

                    broken_matches = []

                    successors = self.circuit_dag.get_node(circuit_id).successors

                    for succ in successors:
                        circuit_blocked_nomatch[succ] = True
                        if circuit_matched_nomatch[succ]:
                            broken_matches.append(succ)
                            circuit_matched_nomatch[succ] = []

                    new_matches_scenario_nomatch = [elem for elem in matches_scenario_nomatch
                                            if elem[1] not in broken_matches]

                    condition_block = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario_nomatch:
                            condition_block = False
                            break

                    if ([self.node_id_t, self.node_id_c] in matches_scenario_nomatch) \
                            and (condition_block or not match_backward):
                        new_matching_scenario = MatchingScenarios(circuit_matched_nomatch,
                                                                  circuit_blocked_nomatch,
                                                                  template_matched_nomatch,
                                                                  template_blocked_nomatch,
                                                                  new_matches_scenario_nomatch,
                                                                  counter_scenario + 1)
                        matching_list.append_scenario(new_matching_scenario)

        length = max(len(m.match) for m in match_store_list)

        for scenario in match_store_list:
            if (len(scenario.match) == length) and \
                    not any(scenario.match == x.match for x in self.match_final):
                self.match_final.append(scenario)
