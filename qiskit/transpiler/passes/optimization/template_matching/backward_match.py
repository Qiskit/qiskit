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

    def __init__(self, circuit_dag, template_dag, matches, counter):
        """
        Create a MatchingScenarios class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            template_dag (DAGDependency): template in the dag dependency form.
            matches (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """
        # DAGDependency object: circuit
        self.circuit_dag = circuit_dag
        # DAGDependency object: template
        self.template_dag = template_dag
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

    def _find_backward_candidates(self, template_dag, matches):
        """
        Function which returns the list possible backward candidates in the template dag.
        Args:
            template_dag (DAGDependency): dag representation of the template circuit.
            matches (list): list of matches.
        Returns:
            list: list of backward candidates (id).
        """
        template_block = []

        for node_id in range(self.node_id_t, template_dag.size()):
            if template_dag.get_node(node_id).isblocked:
                template_block.append(node_id)

        matches_template = sorted([match[0] for match in matches])

        successors = template_dag.get_node(self.node_id_t).successors
        potential = []
        for index in range(self.node_id_t + 1, template_dag.size()):
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

    def run_backward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubits configuration.

        """
        match_store_list = []

        counter = 1

        first_match = MatchingScenarios(self.circuit_dag,
                                        self.template_dag,
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

            circuit_scenario = scenario.circuit_dag
            template_scenario = scenario.template_dag
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
            v = circuit_scenario.get_node(circuit_id)

            if v.isblocked:
                matching_scenario = MatchingScenarios(circuit_scenario,
                                                      template_scenario,
                                                      matches_scenario,
                                                      counter_scenario + 1)
                matching_list.append_scenario(matching_scenario)
                continue

            candidates_indices = self._find_backward_candidates(template_scenario,
                                                                matches_scenario)

            qarg1 = v.qindices
            carg1 = v.cindices

            qarg1 = self._update_qarg_indices(qarg1)
            carg1 = self._update_carg_indices(carg1)

            global_match = False

            for template_id in candidates_indices:
                actual_match = False

                node_template = self.template_dag.get_node(template_id)
                qarg2 = self.template_dag.get_node(template_id).qindices

                if len(qarg1) != len(qarg2) \
                        or set(qarg1) != set(qarg2)\
                        or v.name != node_template.name:
                    continue

                if self._is_same_q_conf(v, node_template, qarg1)\
                        and self._is_same_c_conf(v, node_template, carg1)\
                        and self._is_same_op(v, node_template):

                    tree = tree + 1

                    circuit_scenario_match = circuit_scenario.copy()
                    template_scenario_match = template_scenario.copy()
                    matches_scenario_match = matches_scenario.copy()

                    block_list = []
                    broken_matches_match = []

                    for potential_block in template_scenario_match.successors(template_id):
                        if not template_scenario_match.get_node(potential_block).matchedwith:
                            template_scenario_match.get_node(potential_block).isblocked = True
                            block_list.append(potential_block)
                            for block_id in block_list:
                                for succ_id in \
                                        template_scenario_match.successors(block_id):
                                    template_scenario_match.get_node(succ_id).isblocked \
                                        = True
                                    if template_scenario_match.get_node(succ_id).matchedwith:
                                        new_id = \
                                            template_scenario_match.get_node(succ_id).matchedwith[0]
                                        circuit_scenario_match.get_node(new_id).matchedwith = []
                                        template_scenario_match.get_node(succ_id).matchedwith \
                                            = []
                                        broken_matches_match.append(succ_id)

                    matches_scenario_match = [elem for elem in matches_scenario_match
                                              if elem[0] not in broken_matches_match]

                    condition = True

                    for back_match in match_backward:
                        if back_match not in matches_scenario_match:
                            condition = False
                            break

                    if ([self.node_id_t, self.node_id_c] in matches_scenario_match) \
                            and (condition or not match_backward):
                        template_scenario_match.get_node(template_id).matchedwith = [circuit_id]
                        circuit_scenario_match.get_node(circuit_id).matchedwith = [template_id]
                        matches_scenario_match.append([template_id, circuit_id])

                        new_matching_scenario = MatchingScenarios(circuit_scenario_match,
                                                                  template_scenario_match,
                                                                  matches_scenario_match
                                                                  , counter_scenario + 1)
                        matching_list.append_scenario(new_matching_scenario)

                        actual_match = True
                        global_match = True

                    if actual_match:

                        circuit_scenario_block_s = circuit_scenario.copy()
                        template_scenario_block_s = template_scenario.copy()
                        matches_scenario_block_s = matches_scenario.copy()

                        circuit_scenario_block_s.get_node(circuit_id).isblocked = True
                        new_v = circuit_scenario_block_s.get_node(circuit_id)

                        broken_matches = []

                        for succ in new_v.successors:
                            circuit_scenario_block_s.get_node(succ).isblocked = True
                            if circuit_scenario_block_s.get_node(succ).matchedwith:
                                broken_matches.append(succ)
                                circuit_scenario_block_s.get_node(succ).matchedwith = []

                        new_matches_scenario_block_s = [elem for elem in matches_scenario_block_s
                                                        if elem[1] not in broken_matches]

                        condition_not_greedy = True

                        for back_match in match_backward:
                            if back_match not in new_matches_scenario_block_s:
                                condition_not_greedy = False
                                break

                        if ([self.node_id_t, self.node_id_c] in new_matches_scenario_block_s) and \
                                (condition_not_greedy or not match_backward):
                            new_matching_scenario = MatchingScenarios(circuit_scenario_block_s,
                                                                      template_scenario_block_s,
                                                                      new_matches_scenario_block_s,
                                                                      counter_scenario + 1)
                            matching_list.append_scenario(new_matching_scenario)

                        if broken_matches:

                            circuit_scenario_block_p = circuit_scenario.copy()
                            template_scenario_block_p = template_scenario.copy()
                            matches_scenario_block_p = matches_scenario.copy()

                            circuit_scenario_block_p.get_node(circuit_id).isblocked = True
                            new_v = circuit_scenario_block_p.get_node(circuit_id)

                            for pred in new_v.predecessors:
                                circuit_scenario_block_p.get_node(pred).isblocked = True

                            matching_scenario = MatchingScenarios(circuit_scenario_block_p,
                                                                  template_scenario_block_p,
                                                                  matches_scenario_block_p,
                                                                  counter_scenario + 1)
                            matching_list.append_scenario(matching_scenario)

            if not global_match:

                tree = tree + 1

                v.isblocked = True

                successors = v.successors

                following_matches = []
                for succ in successors:
                    if circuit_scenario.get_node(succ).matchedwith:
                        following_matches.append(succ)

                predecessors = v.predecessors

                if not predecessors or not following_matches:

                    matching_scenario = MatchingScenarios(circuit_scenario,
                                                          template_scenario,
                                                          matches_scenario,
                                                          counter_scenario + 1)
                    matching_list.append_scenario(matching_scenario)

                else:
                    # Option 1
                    for pred in predecessors:
                        circuit_scenario.get_node(pred).isblocked = True

                    matching_scenario = MatchingScenarios(circuit_scenario,
                                                          template_scenario,
                                                          matches_scenario,
                                                          counter_scenario + 1)
                    matching_list.append_scenario(matching_scenario)

                    new_circuit_scenario = circuit_scenario.copy()
                    new_template_scenario = template_scenario.copy()
                    new_matches_scenario = matches_scenario.copy()

                    new_v = new_circuit_scenario.get_node(circuit_id)

                    broken_matches = []

                    for succ in new_v.successors:
                        new_circuit_scenario.get_node(succ).isblocked = True
                        if new_circuit_scenario.get_node(succ).matchedwith:
                            broken_matches.append(succ)
                            new_circuit_scenario.get_node(succ).matchedwith = []

                    new_matches_scenario = [elem for elem in new_matches_scenario
                                            if elem[1] not in broken_matches]

                    condition_block = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario:
                            condition_block = False
                            break

                    if ([self.node_id_t, self.node_id_c] in new_matches_scenario) \
                            and (condition_block or not match_backward):
                        new_matching_scenario = MatchingScenarios(new_circuit_scenario,
                                                                  new_template_scenario,
                                                                  new_matches_scenario,
                                                                  counter_scenario + 1)
                        matching_list.append_scenario(new_matching_scenario)

        length = max(len(m.match) for m in match_store_list)

        for scenario in match_store_list:
            if (len(scenario.match) == length) and \
                    not any(scenario.match == x.match for x in self.match_final):
                self.match_final.append(scenario)
