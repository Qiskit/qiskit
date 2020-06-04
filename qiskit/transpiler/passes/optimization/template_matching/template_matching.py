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
Template matching for all possible qubit configurations and initial matches. It
returns the list of all matches obtained from this algorithm.


**Reference:**

[1] Iten, R., Sutter, D. and Woerner, S., 2019.
Efficient template matching in quantum circuits.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""

import itertools

from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch


class TemplateMatching:
    """
    Class TemplatingMatching allows to apply the full template matching algorithm.
    """

    def __init__(self, circuit_dag_dep, template_dag_dep,
                 heuristics_qubits_param=None, heuristics_backward_param=None):
        """
        Create a TemplateMatching object with necessary arguments.
        Args:
            circuit_dag_dep (QuantumCircuit): circuit.
            template_dag_dep (QuantumCircuit): template.
            heuristics_backward_param (list[int]): [length, survivor]
            heuristics_qubits_param (list[int]): [length]
        """
        self.circuit_dag_dep = circuit_dag_dep
        self.template_dag_dep = template_dag_dep
        self.match_list = []
        self.heuristics_qubits_param = heuristics_qubits_param\
            if heuristics_qubits_param is not None else []
        self.heuristics_backward_param = heuristics_backward_param\
            if heuristics_backward_param is not None else []

    def _list_first_match(self, qarg_c, qarg_t, carg_c, carg_t, n_qubits_t, n_clbits_t):
        """
        Returns the list of qubit for circuit given the first match, the unknown qubit are
        replaced by -1.
        Args:
            qarg_c (list): list of qubits on which the first matched circuit gate is acting on.
            qarg_t (list): list of qubits on which the first matched template gate is acting on.
            carg_c (list): list of clbits on which the first matched circuit gate is acting on.
            carg_t (list): list of clbits on which the first matched template gate is acting on.
            n_qubits_t (int): number of qubit in the template.
            n_clbits_t (int): number of qubit in the template.
        Returns:
            list: list of qubits to consider in circuit (with specific order).
        """

        l_q = [-1] * n_qubits_t

        for q in qarg_t:
            l_q[q] = qarg_c[qarg_t.index(q)]

        if not carg_t or not carg_c:
            l_c = []
        else:
            l_c = [-1] * n_clbits_t
            for c in carg_t:
                l_c[c] = carg_c[carg_t.index(c)]

        return l_q, l_c

    def _sublist(self, lst, exclude, length):
        """
        Function that returns all possible combinations of a given length, considering an
        excluded list of elements.
        Args:
            lst (list): list of qubits indices from the circuit.
            exclude (list): list of qubits from the first matched circuit gate.
            length (int): length of the list to be returned (number of template qubit -
            number of qubit from the first matched template gate).
        Yield:
            iterator: Iterator of the possible lists.
        """
        for sublist in itertools.combinations([e for e in lst if e not in exclude], length):
            yield list(sublist)

    def _list_qubit_clbit_circuit(self, list_first_match, permutation):
        """
        Function that returns the list of the circuit qubits and clbits give a permutation
        and an initial match.
        Args:
            list_first_match (list): list of qubits indices for the initial match.
            permutation (list): possible permutation for the circuit qubit.
        Returns:
            list: list of circuit qubit for the given permutation and initial match.
        """
        list_circuit = []

        counter = 0

        for elem in list_first_match:
            if elem == -1:
                list_circuit.append(permutation[counter])
                counter = counter + 1
            else:
                list_circuit.append(elem)

        return list_circuit

    def _add_match(self, backward_match_list):
        """
        Method to add a match in list only if it is not already in it.
        If the match is already in the list, the qubit configuration
        is append to the existing match.
        Args:
            backward_match_list (list): match from the backward part of the
            algorithm.
        """

        already_in = False

        for b_match in backward_match_list:
            for l_match in self.match_list:
                if b_match.match == l_match.match:
                    index = self.match_list.index(l_match)
                    self.match_list[index].qubit.append(b_match.qubit[0])
                    already_in = True

            if not already_in:
                self.match_list.append(b_match)

    def _explore_circuit(self, node_id_c, n_qubits_t, length):
        """
        Explore the successors of the node_id_c (up to the given length).
        Args:
            node_id_c (int): first match id in the circuit.
            n_qubits_t (int): number of qubits in the template.
            length (int): length for exploration of the successors.
        Returns:
            list: qubits configuration for the 'length' successors of node_id_c.
        """
        successors = self.circuit_dag_dep.get_node(node_id_c).successors

        counter = 1
        qubit_set = set(self.circuit_dag_dep.get_node(node_id_c).qindices)
        for succ in successors:
            qarg = self.circuit_dag_dep.get_node(succ).qindices
            if (len(qubit_set | set(qarg))) < n_qubits_t and counter <= length:
                qubit_set = qubit_set | set(qarg)
            counter += 1
        return list(qubit_set)

    def run_template_matching(self):
        """
        Change qubits indices of the current circuit node in order to
        be comparable the indices of the template qubits list.
        """

        # Get the number of qubits/clbits for both circuit and template.
        n_qubits_c = len(self.circuit_dag_dep.qubits())
        n_clbits_c = len(self.circuit_dag_dep.clbits())

        n_qubits_t = len(self.template_dag_dep.qubits())
        n_clbits_t = len(self.template_dag_dep.clbits())

        # Loop over the indices of both template and circuit.
        for template_index in range(0, self.template_dag_dep.size()):
            for circuit_index in range(0, self.circuit_dag_dep.size()):
                # Operations match.
                if self.circuit_dag_dep.get_node(circuit_index).op == \
                        self.template_dag_dep.get_node(template_index).op:

                    qarg_c = self.circuit_dag_dep.get_node(circuit_index).qindices
                    carg_c = self.circuit_dag_dep.get_node(circuit_index).cindices

                    qarg_t = self.template_dag_dep.get_node(template_index).qindices
                    carg_t = self.template_dag_dep.get_node(template_index).cindices

                    node_id_c = circuit_index
                    node_id_t = template_index

                    # Fix the qubits and clbits configuration given the first match.
                    list_first_match_q, list_first_match_c = self._list_first_match(qarg_c, qarg_t,
                                                                                    carg_c, carg_t,
                                                                                    n_qubits_t,
                                                                                    n_clbits_t)

                    list_circuit_q = list(range(0, n_qubits_c))
                    list_circuit_c = list(range(0, n_clbits_c))

                    # If the parameter for qubits heuristics is given then extracts
                    # the list of qubits for the successors (length(int)) in the circuit.
                    if self.heuristics_qubits_param:
                        heuristics_qubits = self._explore_circuit(node_id_c, n_qubits_t,
                                                                  self.heuristics_qubits_param[0])
                    else:
                        heuristics_qubits = []

                    for sub_q in self._sublist(list_circuit_q, qarg_c, n_qubits_t - len(qarg_t)):
                        # If the heuristics qubits are a subset of the given qubits configuration,
                        # then this configuration is accepted.
                        if set(heuristics_qubits).issubset(set(sub_q) | set(qarg_c)):
                            # Permute the qubit configuration.
                            for perm_q in itertools.permutations(sub_q):
                                perm_q = list(perm_q)

                                list_qubit_circuit =\
                                    self._list_qubit_clbit_circuit(list_first_match_q, perm_q)

                                # Check for clbits configurations if there are clbits.
                                if list_circuit_c:
                                    for sub_c in self._sublist(list_circuit_c, carg_c,
                                                               n_clbits_t - len(carg_t)):
                                        for perm_c in itertools.permutations(sub_c):
                                            perm_c = list(perm_c)

                                            list_clbit_circuit =\
                                                self._list_qubit_clbit_circuit(list_first_match_c,
                                                                               perm_c)

                                            # Apply the forward match part of the algorithm.
                                            forward = ForwardMatch(self.circuit_dag_dep,
                                                                   self.template_dag_dep,
                                                                   node_id_c, node_id_t,
                                                                   list_qubit_circuit,
                                                                   list_clbit_circuit)
                                            forward.run_forward_match()

                                            # Apply the backward match part of the algorithm.
                                            backward = BackwardMatch(forward.circuit_dag_dep,
                                                                     forward.template_dag_dep,
                                                                     forward.match, node_id_c,
                                                                     node_id_t,
                                                                     list_qubit_circuit,
                                                                     list_clbit_circuit,
                                                                     self.heuristics_backward_param)
                                            backward.run_backward_match()

                                            # Add the matches to the list.
                                            self._add_match(backward.match_final)
                                else:
                                    # Apply the forward match part of the algorithm.
                                    forward = ForwardMatch(self.circuit_dag_dep,
                                                           self.template_dag_dep,
                                                           node_id_c, node_id_t,
                                                           list_qubit_circuit)
                                    forward.run_forward_match()

                                    # Apply the backward match part of the algorithm.
                                    backward = BackwardMatch(forward.circuit_dag_dep,
                                                             forward.template_dag_dep,
                                                             forward.match,
                                                             node_id_c,
                                                             node_id_t,
                                                             list_qubit_circuit,
                                                             self.heuristics_backward_param)
                                    backward.run_backward_match()

                                    # Add the matches to the list.
                                    self._add_match(backward.match_final)

        # Sort the list of matches according to the length of the matches (decreasing order).
        self.match_list.sort(key=lambda x: len(x.match), reverse=True)
