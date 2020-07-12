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

"""DAGDependency class for representing non-commutativity in a circuit.
"""

import heapq
from collections import OrderedDict
import networkx as nx
import retworkx as rx
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
from qiskit.quantum_info.operators import Operator


class DAGDependency:
    """Object to represent a quantum circuit as a directed acyclic graph
    via operation dependencies (i.e. lack of commutation).

    The nodes in the graph are operations represented by quantum gates.
    The edges correspond to non-commutation between two operations
    (i.e. a dependency). A directed edge from node A to node B means that
    operation A does not commute with operation B.
    The object's methods allow circuits to be constructed.

    The nodes in the graph have the following attributes:
    'operation', 'successors', 'predecessors'.

    **Example:**

    Bell circuit with no measurement.

    .. parsed-literal::

              ┌───┐
        qr_0: ┤ H ├──■──
              └───┘┌─┴─┐
        qr_1: ─────┤ X ├
                   └───┘

    The dependency DAG for the above circuit is represented by two nodes.
    The first one corresponds to Hadamard gate, the second one to the CNOT gate
    as the gates do not commute there is an edge between the two nodes.

    **Reference:**

    [1] Iten, R., Sutter, D. and Woerner, S., 2019.
    Efficient template matching in quantum circuits.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_
    """

    def __init__(self):
        """
        Create an empty DAGDependency.
        """
        # Circuit name
        self.name = None

        # Directed multigraph whose nodes are operations(gates) and edges
        # represent non-commutativity between two gates.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of all Qubit/Clbit wires.
        self.qubits = []
        self.clbits = []

    def to_networkx(self):
        """Returns a copy of the DAGDependency in networkx format."""
        # For backwards compatibility, return networkx structure from terra 0.12
        # where DAGNodes instances are used as indexes on the networkx graph.

        dag_networkx = nx.MultiDiGraph()

        for node in self.get_nodes():
            dag_networkx.add_node(node)
        for node in self.topological_nodes():
            for source_id, dest_id, edge in \
                    self.get_in_edges(node.node_id):
                dag_networkx.add_edge(self.get_node(source_id), self.get_node(dest_id), **edge)
        return dag_networkx

    def to_retworkx(self):
        """ Returns the DAGDependency in retworkx format."""
        return self._multi_graph

    def size(self):
        """ Returns the number of gates in the circuit"""
        return len(self._multi_graph)

    def depth(self):
        """Return the circuit depth.
        Returns:
            int: the circuit depth
        """
        depth = rx.dag_longest_path_length(self._multi_graph)
        return depth if depth >= 0 else 0

    def add_qreg(self, qreg):
        """Add qubits in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGDependencyError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGDependencyError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg
        for j in range(qreg.size):
            self.qubits.append(qreg[j])

    def add_creg(self, creg):
        """Add clbits in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGDependencyError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGDependencyError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        for j in range(creg.size):
            self.clbits.append(creg[j])

    def _add_multi_graph_node(self, node):
        """
        Args:
            node (DAGDepNode): considered node.

        Returns:
            node_id(int): corresponding label to the added node.
        """
        node_id = self._multi_graph.add_node(node)
        node.node_id = node_id
        return node_id

    def get_nodes(self):
        """
        Returns:
            generator(dict): iterator over all the nodes.
        """
        return iter(self._multi_graph.nodes())

    def get_node(self, node_id):
        """
        Args:
            node_id (int): label of considered node.

        Returns:
            node: corresponding to the label.
        """
        return self._multi_graph.get_node_data(node_id)

    def _add_multi_graph_edge(self, src_id, dest_id, data):
        """
        Function to add an edge from given data (dict) between two nodes.

        Args:
            src_id (int): label of the first node.
            dest_id (int): label of the second node.
            data (dict): data contained on the edge.

        """
        self._multi_graph.add_edge(src_id, dest_id, data)

    def get_edges(self, src_id, dest_id):
        """
        Edge enumeration between two nodes through method get_all_edge_data.

        Args:
            src_id (int): label of the first node.
            dest_id (int): label of the second node.

        Returns:
            List: corresponding to all edges between the two nodes.
        """
        return self._multi_graph.get_all_edge_data(src_id, dest_id)

    def get_all_edges(self):
        """
        Enumaration of all edges.

        Returns:
            List: corresponding to the label.
        """

        return [(src, dest, data)
                for src_node in self._multi_graph.nodes()
                for (src, dest, data)
                in self._multi_graph.out_edges(src_node.node_id)]

    def get_in_edges(self, node_id):
        """
        Enumeration of all incoming edges for a given node.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: corresponding incoming edges data.
        """
        return self._multi_graph.in_edges(node_id)

    def get_out_edges(self, node_id):
        """
        Enumeration of all outgoing edges for a given node.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: corresponding outgoing edges data.
        """
        return self._multi_graph.out_edges(node_id)

    def direct_successors(self, node_id):
        """
        Direct successors id of a given node as sorted list.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: direct successors id as a sorted list
        """
        return sorted(list(self._multi_graph.adj_direction(node_id, False).keys()))

    def direct_predecessors(self, node_id):
        """
        Direct predecessors id of a given node as sorted list.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: direct predecessors id as a sorted list
        """
        return sorted(list(self._multi_graph.adj_direction(node_id, True).keys()))

    def successors(self, node_id):
        """
        Successors id of a given node as sorted list.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: all successors id as a sorted list
        """
        return self._multi_graph.get_node_data(node_id).successors

    def predecessors(self, node_id):
        """
        Predecessors id of a given node as sorted list.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: all predecessors id as a sorted list
        """
        return self._multi_graph.get_node_data(node_id).predecessors

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order.
        """

        def _key(x):
            return x.sort_key

        return iter(rx.lexicographical_topological_sort(
            self._multi_graph,
            key=_key))

    def add_op_node(self, operation, qargs, cargs):
        """Add a DAGDepNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Instruction): operation as a quantum gate.
            qargs (list[Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to.
        """
        directives = ['measure', 'barrier', 'snapshot']
        if operation.name not in directives:
            qindices_list = []
            for elem in qargs:
                qindices_list.append(self.qubits.index(elem))
            if operation.condition:
                for clbit in self.clbits:
                    if clbit.register == operation.condition[0]:
                        initial = self.clbits.index(clbit)
                        final = self.clbits.index(clbit) + clbit.register.size
                        cindices_list = range(initial, final)
                        break
            else:
                cindices_list = []
        else:
            qindices_list = []
            cindices_list = []

        new_node = DAGDepNode(type="op", op=operation, name=operation.name, qargs=qargs,
                              cargs=cargs, condition=operation.condition, successors=[],
                              predecessors=[], qindices=qindices_list, cindices=cindices_list)
        self._add_multi_graph_node(new_node)
        self._update_edges()

    def _gather_pred(self, node_id, direct_pred):
        """Function set an attribute predecessors and gather multiple lists
        of direct predecessors into a single one.

        Args:
            node_id (int): label of the considered node in the DAG
            direct_pred (list): list of direct successors for the given node

        Returns:
            DAGDependency: A multigraph with update of the attribute ['predecessors']
            the lists of direct successors are put into a single one
        """
        gather = self._multi_graph
        gather.get_node_data(node_id).predecessors = []
        for d_pred in direct_pred:
            gather.get_node_data(node_id).predecessors.append([d_pred])
            pred = self._multi_graph.get_node_data(d_pred).predecessors
            gather.get_node_data(node_id).predecessors.append(pred)
        return gather

    def _gather_succ(self, node_id, direct_succ):
        """
        Function set an attribute successors and gather multiple lists
        of direct successors into a single one.

        Args:
            node_id (int): label of the considered node in the DAG
            direct_succ (list): list of direct successors for the given node

        Returns:
            MultiDiGraph: with update of the attribute ['predecessors']
            the lists of direct successors are put into a single one
        """
        gather = self._multi_graph
        for d_succ in direct_succ:
            gather.get_node_data(node_id).successors.append([d_succ])
            succ = gather.get_node_data(d_succ).successors
            gather.get_node_data(node_id).successors.append(succ)
        return gather

    def _list_pred(self, node_id):
        """
        Use _gather_pred function and merge_no_duplicates to construct
        the list of predecessors for a given node.

        Args:
            node_id (int): label of the considered node
        """
        direct_pred = self.direct_predecessors(node_id)
        self._multi_graph = self._gather_pred(node_id, direct_pred)
        self._multi_graph.get_node_data(node_id).predecessors = list(
            merge_no_duplicates(*(self._multi_graph.get_node_data(node_id).predecessors)))

    def _update_edges(self):
        """
        Function to verify the commutation relation and reachability
        for predecessors, the nodes do not commute and
        if the predecessor is reachable. Update the DAGDependency by
        introducing edges and predecessors(attribute)
        """
        max_node_id = len(self._multi_graph) - 1
        max_node = self._multi_graph.get_node_data(max_node_id)

        for current_node_id in range(0, max_node_id):
            self._multi_graph.get_node_data(current_node_id).reachable = True
        # Check the commutation relation with reachable node, it adds edges if it does not commute
        for prev_node_id in range(max_node_id - 1, -1, -1):
            if self._multi_graph.get_node_data(prev_node_id).reachable and not _commute(
                    self._multi_graph.get_node_data(prev_node_id), max_node):
                self._multi_graph.add_edge(prev_node_id, max_node_id, {'commute': False})
                self._list_pred(max_node_id)
                list_predecessors = self._multi_graph.get_node_data(max_node_id).predecessors
                for pred_id in list_predecessors:
                    self._multi_graph.get_node_data(pred_id).reachable = False

    def _add_successors(self):
        """
        Use _gather_succ and merge_no_duplicates to create the list of successors
        for each node. Update DAGDependency 'successors' attribute. It has to
        be used when the DAGDependency() object is complete (i.e. converters).
        """
        for node_id in range(len(self._multi_graph) - 1, -1, -1):
            direct_successors = self.direct_successors(node_id)

            self._multi_graph = self._gather_succ(node_id, direct_successors)

            self._multi_graph.get_node_data(node_id).successors = list(
                merge_no_duplicates(*self._multi_graph.get_node_data(node_id).successors))

    def copy(self):
        """
        Function to copy a DAGDependency object.
        Returns:
            DAGDependency: a copy of a DAGDependency object.
        """

        dag = DAGDependency()
        dag.name = self.name
        dag.cregs = self.cregs.copy()
        dag.qregs = self.qregs.copy()

        for node in self.get_nodes():
            dag._multi_graph.add_node(node.copy())
        for edges in self.get_all_edges():
            dag._multi_graph.add_edge(edges[0], edges[1], edges[2])
        return dag

    def draw(self, scale=0.7, filename=None, style='color'):
        """
        Draws the DAGDependency graph.

        This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs
        Graphviz <https://www.graphviz.org/>` to be installed.

        Args:
            scale (float): scaling factor
            filename (str): file path to save image to (format inferred from name)
            style (str): 'plain': B&W graph
                         'color' (default): color input/output/op nodes

        Returns:
            Ipython.display.Image: if in Jupyter notebook and not saving to file,
                otherwise None.
        """
        from qiskit.visualization.dag_visualization import dag_drawer
        return dag_drawer(dag=self, scale=scale, filename=filename,
                          style=style, category='dependency')


def merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging

    Args:
        *iterables: A list of k sorted lists

    Yields:
        Iterator: List from the merging of the k ones (without duplicates
    """
    last = object()
    for val in heapq.merge(*iterables):
        if val != last:
            last = val
            yield val


def _commute(node1, node2):
    """Function to verify commutation relation between two nodes in the DAG

    Args:
        node1 (DAGnode): first node operation (attribute ['operation'] in the DAG)
        node2 (DAGnode): second node operation

    Return:
        bool: True if the gates commute and false if it is not the case.
    """

    # Create set of qubits on which the operation acts
    qarg1 = [node1.qargs[i].index for i in range(0, len(node1.qargs))]
    qarg2 = [node2.qargs[i].index for i in range(0, len(node2.qargs))]

    # Create set of cbits on which the operation acts
    carg1 = [node1.qargs[i].index for i in range(0, len(node1.cargs))]
    carg2 = [node2.qargs[i].index for i in range(0, len(node2.cargs))]

    # Commutation for classical conditional gates
    if node1.condition or node2.condition:
        intersection = set(qarg1).intersection(set(qarg2))
        if intersection or carg1 or carg2:
            commute_condition = False
        else:
            commute_condition = True
        return commute_condition

    # Commutation for measurement
    if node1.name == 'measure' or node2.name == 'measure':
        intersection_q = set(qarg1).intersection(set(qarg2))
        intersection_c = set(carg1).intersection(set(carg2))
        if intersection_q or intersection_c:
            commute_measurement = False
        else:
            commute_measurement = True
        return commute_measurement

    # Commutation for barrier-like directives
    directives = ['barrier', 'snapshot']
    if node1.name in directives or node2.name in directives:
        intersection = set(qarg1).intersection(set(qarg2))
        if intersection:
            commute_directive = False
        else:
            commute_directive = True
        return commute_directive

    # List of non commuting gates (TO DO: add more elements)
    non_commute_list = [set(['x', 'y']), set(['x', 'z'])]

    if qarg1 == qarg2 and (set([node1.name, node2.name]) in non_commute_list):
        return False

    # Create matrices to check commutation relation if no other criteria are matched
    qarg = list(set(node1.qargs + node2.qargs))
    qbit_num = len(qarg)

    qarg1 = [qarg.index(q) for q in node1.qargs]
    qarg2 = [qarg.index(q) for q in node2.qargs]

    id_op = Operator(np.eye(2 ** qbit_num))

    op12 = id_op.compose(node1.op, qargs=qarg1).compose(node2.op, qargs=qarg2)
    op21 = id_op.compose(node2.op, qargs=qarg2).compose(node1.op, qargs=qarg1)

    if_commute = (op12 == op21)

    return if_commute
