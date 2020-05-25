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
Template matching substitution, given a list of maximal matches it substitutes
them in circuit and creates a new optimized dag version of the circuit.
"""

from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag


class SubstitutionConfig:
    """
    Class to store the configuration of a given match substitution, which circuit
    gates, template gates, qubits, and clbits and predecessors of the match
    in the circuit.
    """

    def __init__(self, circuit_config, template_config, pred_block,
                 qubit_config, clbit_config=None):

        self.circuit_config = circuit_config
        self.template_config = template_config
        self.qubit_config = qubit_config
        self.clbit_config = clbit_config if clbit_config is not None else []
        self.pred_block = pred_block


class TemplateSubstitution:
    """
    Class to run the subsitution algorithm from the list of maximal matches.
    """

    def __init__(self, max_matches, circuit_dag_dep, template_dag_dep):
        """
        Initialize TemplateSubstitution with necessary arguments.
        Args:
            max_matches (list): list of maximal matches obtained from the running
             the template matching algorithm.
            circuit_dag_dep (DAGDependency): circuit in the dag dependency form.
            template_dag_dep (DAGDependency): template in the dag dependency form.
        """

        self.match_stack = max_matches
        self.circuit_dag_dep = circuit_dag_dep
        self.template_dag_dep = template_dag_dep

        self.substitution_list = []
        self.unmatched_list = []
        self.dag_dep_optimized = DAGDependency()
        self.dag_optimized = DAGCircuit()

    def _pred_block(self, circuit_sublist):
        """
        It returns the predecessors of a given part of the circuit.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
        Returns:
            list: List of predecessors of the current match circuit configuration.
        """
        predecessors = {}
        for index in circuit_sublist:
            predecessors = predecessors | set(self.circuit_dag_dep.get_node(index).predecessors)

        exclude = {}
        for elem in self.substitution_list:
            exclude = exclude | elem.circuit_config | elem.pred_block

        pred = list(predecessors - set(circuit_sublist) - exclude)
        pred.sort()

        return pred

    def _rules(self, circuit_sublist):
        """
        Set of rules to decide whether the match is to be substitute or not.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
        Returns:
            bool: True if the match respects the given rule for replacement, False otherwise.
        """
        if len(circuit_sublist) > (self.template_dag_dep.size() / 2):
            for elem in circuit_sublist:
                for config in self.substitution_list:
                    if any(elem == x for x in config.circuit_config):
                        return False
            return True
        else:
            return False

    def _template_inverse(self, template_sublist):
        """
        The template circuit realizes the identity operator, then given the list of
        matches in the template, it returns the inverse part of the template that
        will be replaced.
        Args:
            template_sublist (list): list of the gates matched in the circuit.
        Returns:
            list: the template inverse part that will substitute the circuit match.
        """
        template_list = range(0, self.template_dag_dep.size())
        inverse = list(set(template_list) - set(template_sublist))
        left = []
        right = []

        pred = set([])
        for index in template_sublist:
            pred = pred | set(self.template_dag_dep.get_node(index).predecessors)
        pred = list(pred - set(template_sublist))

        succ = set([])
        for index in template_sublist:
            succ = succ | set(self.template_dag_dep.get_node(index).successors)
        succ = list(succ - set(template_sublist))

        comm = list(set(template_list) - set(pred) - set(succ))

        for elem in inverse:
            if elem in pred:
                left.append(elem)
            elif elem in succ:
                right.append(elem)
            elif elem in comm:
                right.append(elem)

        left.sort()
        right.sort()

        left.reverse()
        right.reverse()

        total = left + right
        return total

    def _substitution(self):
        """
        From the list of maximal matches, it chooses which one will be used and gives the necessary
        details for each substitution(template inverse, predecessors of the match).
        """
        while self.match_stack:

            current = self.match_stack.pop(0)

            current_match = current.match
            current_qubit = current.qubit
            current_clbit = current.clbit

            template_sublist = [x[0] for x in current_match]

            circuit_sublist = [x[1] for x in current_match]
            circuit_sublist.sort()

            if self._rules(circuit_sublist):
                template_sublist_inverse = self._template_inverse(template_sublist)

                pred = self._pred_block(circuit_sublist)

                config = SubstitutionConfig(circuit_sublist,
                                            template_sublist_inverse,
                                            pred,
                                            current_qubit,
                                            current_clbit)
                self.substitution_list.append(config)

        circuit_list = []
        for elem in self.substitution_list:
            circuit_list = circuit_list + elem.circuit_config + elem.pred_block

        self.unmatched_list = list({range(0, self.circuit_dag_dep.size())}
                                   - set(circuit_list))

        self.substitution_list.sort(key=lambda x: x.circuit_config[0])

    def run_dag_opt(self):
        """
        It runs the substitution algorithm and creates the optimized DAGCircuit().
        """
        self._substitution()

        dag_dep_opt = DAGDependency()

        dag_dep_opt.name = self.circuit_dag_dep.name
        dag_dep_opt.cregs = self.circuit_dag_dep.cregs.copy()
        dag_dep_opt.qregs = self.circuit_dag_dep.qregs.copy()

        already_sub = []

        for node in self.circuit_dag_dep.get_nodes():
            if node.node_id in already_sub:
                pass
            elif node.node_id in self.unmatched_list:

                inst = node.op.copy()
                inst.condition = node.condition
                dag_dep_opt.add_op_node(inst, node.qargs, node.cargs)

                already_sub.append(node.node_id)

            else:
                bloc = self.substitution_list.pop(0)

                circuit_sub = bloc.circuit_config
                template_inverse = bloc.template_config
                pred = bloc.pred_block

                qubit = bloc.qubit_config[0]

                if bloc.clbit_config:
                    clbit = bloc.clbit_config[0]
                else:
                    clbit = []

                for elem in pred:
                    node = self.circuit_dag_dep.get_node(elem)
                    inst = node.op.copy()
                    inst.condition = node.condition
                    dag_dep_opt.add_op_node(inst, node.qargs, node.cargs)
                    already_sub.append(elem)

                already_sub = already_sub + circuit_sub

                for index in template_inverse:
                    all_qubits = self.circuit_dag_dep.qubits()
                    qarg_t = self.template_dag_dep.get_node(index).qindices
                    qarg_c = [qubit[x] for x in qarg_t]
                    qargs = [all_qubits[x] for x in qarg_c]

                    all_clbits = self.circuit_dag_dep.clbits()
                    carg_t = self.template_dag_dep.get_node(index).cindices

                    if all_clbits and clbit:
                        carg_c = [clbit[x] for x in carg_t]
                        cargs = [all_clbits[x] for x in carg_c]
                    else:
                        cargs = []

                    inst = node.op.copy()
                    inst.condition = node.condition

                    dag_dep_opt.add_op_node(inst.inverse(), qargs, cargs)

        self.dag_dep_optimized = dag_dep_opt
        self.dag_optimized = dagdependency_to_dag(dag_dep_opt)
