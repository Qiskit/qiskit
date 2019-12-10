# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Object to represent a quantum circuit as a directed acyclic graph in the canonical form.

The nodes in the graph are operation represented by quantum gates.
The edges correspond to non-commutation between two operations. A directed edge
from node A to node B means that operation A does not commute with operation B.
The object's methods allow circuits to be constructed.

e.g. Bell circuit with no measurement

         ┌───┐
qr_0: |0>┤ H ├──■──
         └───┘┌─┴─┐
qr_1: |0>─────┤ X ├
              └───┘

In the dag canonical form the circuit is representede by two nodes (1 and 2):
the first one corresponds to Hamdamard gate, the second one to the CNot gate
as the gates do not commute there is an edge between the wo noodes.

The attributes are 'label' 'operation', 'successors', 'predecessors'
In the Bell circuit, the network takes the following form:

[(1, {'label': 1, 'operation': <qiskit.dagcircuit.dagnode.DAGNode object at 0x12207bad0>,
 'successors': [2], 'predecessors': [], 'reachable': False}),
(2, {'label': 2, 'operation': <qiskit.dagcircuit.dagnode.DAGNode object at 0x12207bdd0>,
'successors': [], 'predecessors': [1]})]

The reference paper is https://arxiv.org/abs/1909.05270v1

"""

from collections import OrderedDict
import copy
import heapq
import networkx as nx
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.quantum_info.operators import Operator
from .exceptions import DAGCircuitError
from .dagnode import DAGNode


class DAGcanonical:
    """
    Object to represent a quantum circuit as a DAG in the canonical form.
    """
    def __init__(self):
        """
        Create an empty directed acyclis graph (canonical form)
        """
        # Circuit name
        self.name = None

        # Directed multigraph whose nodes are operations(gates) and edges
        # represent non-commutativity between two gates.
        self._multi_graph = nx.MultiDiGraph()

        # Map of qreg name to QuantumRegister object
        self.qregs = OrderedDict()

        # Map of creg name to ClassicalRegister object
        self.cregs = OrderedDict()
        # Index of the last node added
        self._max_node_id = 0

        # Intern list of nodes
        self._id_to_node = {}

    def to_networkx(self):
        """Returns a copy of the DAGCircuit in networkx format."""
        return copy.deepcopy(self._multi_graph)

    def qubits(self):
        """Return a list of qubits (as a list of Qubit instances)."""
        return [qubit for qreg in self.qregs.values() for qubit in qreg]

    def node_counter(self):
        """ Returns the number of nodes in the dag """
        return len(self._multi_graph)

    def add_qreg(self, qreg):
        """Add qubits in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGCircuitError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGCircuitError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg

    def add_creg(self, creg):
        """Add all wires in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGCircuitError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGCircuitError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg

    def add_node(self, operation, qargs, cargs):
        """Add a DAGnode to the graph.

        Args:
            operation (Instruction): operation as a quantum gate.
            qargs (list[Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to.
        """
        node_properties = {
            "type": "op",
            "op": operation,
            "name": operation.name,
            "qargs": qargs,
            "cargs": cargs,
            "condition": operation.condition
        }

        # Add a new operation node to the graph
        self._max_node_id += 1
        new_node = DAGNode(data_dict=node_properties, nid=self._max_node_id)
        self._multi_graph.add_node(self._max_node_id, label=self._max_node_id, operation=new_node,
                                   successors=[], predecessors=[])
        self._id_to_node[self._max_node_id] = new_node

    def _gather_pred(self, node_id, direct_pred):
        """Function set an attribute predecessors and gather multiple lists
        of direct predecessors into a single one.

        Args:
            node_id (int): label of the considered node in the DAG
            direct_pred (list): list of direct successors for the given node

        Returns:
            DAGcanonical: A multigraph with update of the attribute ['predecessors']
            the lists of direct successors are put into a single one
        """
        gather = self._multi_graph
        gather.nodes[node_id]['predecessors'] = []
        for d_pred in direct_pred:
            gather.nodes[node_id]['predecessors'].append([d_pred])
            pred = self._multi_graph.nodes[d_pred]['predecessors']
            gather.nodes[node_id]['predecessors'].append(pred)
        return gather

    def _gather_succ(self, node_id, direct_succ):
        """
        Function set an attribute successors and gather multiple lists
        of direct successors into a single one.

        Args:
            node_id (int): label of the considered node in the DAG
            direct_succ (lis): list of direct successors for the given node

        Returns:
            MultiDiGraph: with update of the attribute ['predecessors']
            the lists of direct successors are put into a single one
        """
        gather = self._multi_graph
        for d_succ in direct_succ:
            gather.nodes[node_id]['successors'].append([d_succ])
            succ = gather.nodes[d_succ]['successors']
            gather.nodes[node_id]['successors'].append(succ)
        return gather

    def _list_pred(self, node_id):
        """
        Use _gather_pred function and merge_no_duplicates to construct
        the list of predecessors for a given node.

        Args:
            node_id (int): label of the considered node
        """
        direct_pred = sorted(list(self._multi_graph.predecessors(node_id)))
        self._multi_graph = self._gather_pred(node_id, direct_pred)
        self._multi_graph.nodes[node_id]['predecessors'] = list(
            merge_no_duplicates(*(self._multi_graph.nodes[node_id]['predecessors'])))

    def add_edge(self):
        """
        Function to verify the commutation relation and reachability
        for predecessors, the nodes do not commute and
        if the predecessor is reachable. Update the DAGcanonical by
        introducing edges and predecessors(attribute)
        """
        node = self._id_to_node[self._max_node_id]
        max_id = self._max_node_id
        for current_node in range(1, max_id):
            self._multi_graph.nodes[current_node]['reachable'] = True
        # Check the commutation relation with reachable node, it adds edges if it does not commute
        for prev_node in range(max_id - 1, 0, -1):
            if self._multi_graph.nodes[prev_node]['reachable'] and not _commute(
                    self._multi_graph.nodes[prev_node]['operation'], node):
                self._multi_graph.add_edge(prev_node, max_id)
                self._list_pred(max_id)
                list_predecessors = self._multi_graph.nodes[max_id]['predecessors']
                for pred in list_predecessors:
                    self._multi_graph.nodes[pred]['reachable'] = False

    def add_successors(self):
        """
        Use _gather_succ and merge_no_duplicates to create the list of successors
        for each node. Update DAGcanonical with attributes successors.
        """
        for node_id in range(len(self._multi_graph), 0, -1):

            direct_successors = sorted(list(self._multi_graph.successors(node_id)))

            self._multi_graph = self._gather_succ(node_id, direct_successors)

            self._multi_graph.nodes[node_id]['successors'] = list(
                merge_no_duplicates(*(self._multi_graph.nodes[node_id]['successors'])))

    def node(self, node_id):
        """
        Args:
            node_id (int): label of considered node.

        Returns:
            Node: corresponding to the label.
        """
        return self._multi_graph.nodes[node_id]

    def nodes(self):
        """Function to return all nodes

        Yields:
            Iterator: generate all nodes (label, DAGnode).
        """
        for node in self._multi_graph.nodes(data='operation'):
            yield node

    def edges(self):
        """Function to yield all edges.

        Yields:
            Iterator: generate all edges.
        """
        for edge in self._multi_graph.edges(data=True):
            yield edge

    def in_edge(self, node_id):
        """ Get the list of incoming nodes for a given node_id.

        Args:
            node_id (int): id of the corresponding node

        Returns:
            list[In_edges()]: List of all incoming edges.
        """
        return self._multi_graph.in_edges(node_id)

    def out_edge(self, node_id):
        """List of all outgoing edges for the given node id.

        Args:
            node_id (int): id of the corresponding node.

        Returns:
            list[out_edges()]: List of all incoming edges.
        """
        return self._multi_graph.out_edges(node_id)

    def direct_successors(self, node_id):
        """Returns label of direct successors of a node as sorted list """
        return sorted(list(self.to_networkx().successors(node_id)))

    def direct_predecessors(self, node_id):
        """Returns label of direct  predecessors of a node as sorted list """
        return sorted(list(self.to_networkx().predecessors(node_id)))

    def successors(self, node_id):
        """Returns set of the ancestors of a node as DAGNodes."""
        return self.to_networkx().nodes[node_id]['successors']

    def predecessors(self, node_id):
        """Returns set of the descendants of a node as DAGNodes."""
        return self.to_networkx().nodes[node_id]['predecessors']

    def draw(self, scale=0.7, filename=None, style='color', category='canonical'):
        """
        Draws the DAG canonical circuit.

        This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs
        Graphviz <https://www.graphviz.org/>` to be installed.

        Args:
            scale (float): scaling factor
            filename (str): file path to save image to (format inferred from name)
            style (str): 'plain': B&W graph
                         'color' (default): color input/output/op nodes
            category(str): 'canonical' Other type of DAG

        Returns:
            Ipython.display.Image: if in Jupyter notebook and not saving to file,
                otherwise None.
        """
        from qiskit.visualization.dag_visualization import dag_drawer
        return dag_drawer(dag=self, scale=scale, filename=filename, style=style, category=category)


def merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging

    Args:
        *iterables: A list of k sorted lists

    Yields:
        Iteraor: List from the merging of the k ones (without duplicates
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
        intersection = set(qarg1).intersection(set(qarg2))
        if intersection:
            commute_measurement = False
        else:
            commute_measurement = True
        return commute_measurement

    # Commutation for barrier
    if node1.name == 'barrier' or node2.name == 'barrier':
        intersection = set(qarg1).intersection(set(qarg2))
        if intersection:
            commute_barrier = False
        else:
            commute_barrier = True
        return commute_barrier

    # Commutation for snapshot
    if node1.name == 'snapshot' or node2.name == 'snapshot':
        return False

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
