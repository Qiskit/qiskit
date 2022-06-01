# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This pass collapses 2-qubit gates that commute into :class:`.Commuting2qBlock`."""

from qiskit.circuit import Qubit, Gate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.passes.optimization.commutation_analysis import CommutationAnalysis
from qiskit.transpiler import TransformationPass, TranspilerError

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)


class Commuting2qGateGrouper(TransformationPass):
    """A pass that collapses 2-qubit gates that commute into :class:`.Commuting2qBlock`."""

    def __init__(self):
        super().__init__()
        self._commutation_set_by_node = None
        self._done_nodes = set()
        self.requires = [CommutationAnalysis()]

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Collapses 2-qubit gates that commute into :class:`.Commuting2qBlock`.

        Args:
            dag: A DagCircuit in which commutation analysis was performed.

        Returns:
            A dag where multi-qubit commuting groups are collapsed as :class:`.Commuting2qBlock`.

        Raises:
            TranspilerError: If `commutation_set` is not in the property set.
        """
        if self.property_set["commutation_set"] is None:
            raise TranspilerError("commutation_set is not in the property set.")

        new_dag = dag.copy_empty_like()

        for nd in dag.layered_topological_op_nodes():
            if nd in self._done_nodes:
                continue
            if len(nd.qargs) <= 1 or not isinstance(nd.op, Gate):
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
            else:
                commuting_op_set = self._commutes_with(nd)
                used_qarg = {qarg for node in commuting_op_set for qarg in node.qargs}
                used_carg = {carg for node in commuting_op_set for carg in node.cargs}
                new_dag.apply_operation_back(
                    Commuting2qBlock(commuting_op_set),
                    [qbit for qbit in dag.qubits if qbit in used_qarg],
                    [cbit for cbit in dag.clbits if cbit in used_carg],
                )
        return new_dag

    def _commutes_with(self, node):
        commutation_set = set([node])
        for other_element in self.commutation_set_by_node[node]:
            if other_element not in self._done_nodes:  # This op was already included somehow else
                self._done_nodes.add(other_element)
                commutation_set.update(self._commutes_with(other_element))
        return commutation_set

    @property
    def commutation_set_by_node(self):
        """like commutation_set, but the key is a node to which the value commutes with"""
        return self._commutation_set_by_node or self._create_commutation_set_by_node()

    def _create_commutation_set_by_node(self):
        global_commutation_sets = []
        for qbit, sets in self.property_set["commutation_set"].items():
            if not isinstance(qbit, Qubit):
                continue
            for commutation_set in sets:
                for node in commutation_set:
                    if not isinstance(node, DAGOpNode):
                        break
                else:
                    global_commutation_sets.append(commutation_set)
        op_nodes = {}
        for cset in global_commutation_sets:
            for node in cset:
                Commuting2qGateGrouper._search_commutation_sets(
                    op_nodes, global_commutation_sets, node
                )
        self._commutation_set_by_node = op_nodes
        return self._commutation_set_by_node

    @staticmethod
    def _update_op_nodes(op_nodes, node, cset):
        if node in op_nodes:
            op_nodes[node].update(cset)
        else:
            op_nodes[node] = set(cset)

    @staticmethod
    def _search_commutation_sets(op_nodes, commutation_set_list, node=None):
        if not commutation_set_list:
            return
        for idx, cset in enumerate(commutation_set_list):
            if node in cset:
                Commuting2qGateGrouper._update_op_nodes(op_nodes, node, cset)
                for other_node in cset:
                    Commuting2qGateGrouper._search_commutation_sets(
                        op_nodes, commutation_set_list[idx + 1 :], other_node
                    )
                    Commuting2qGateGrouper._update_op_nodes(
                        op_nodes, node, op_nodes.get(other_node, set())
                    )
