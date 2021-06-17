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

import math
import heapq
from collections import OrderedDict, defaultdict
import numpy as np
import retworkx as rx

from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
from qiskit.quantum_info.operators import Operator
from qiskit.exceptions import MissingOptionalLibraryError


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

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

    """

    def __init__(self):
        """
        Create an empty DAGDependency.
        """
        # Circuit name
        self.name = None

        # Circuit metadata
        self.metadata = None

        # Directed multigraph whose nodes are operations(gates) and edges
        # represent non-commutativity between two gates.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of all Qubit/Clbit wires.
        self.qubits = []
        self.clbits = []

        self._global_phase = 0
        self._calibrations = defaultdict(dict)

        self.duration = None
        self.unit = "dt"

    @property
    def global_phase(self):
        """Return the global phase of the circuit."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        """Set the global phase of the circuit.

        Args:
            angle (float, ParameterExpression)
        """
        from qiskit.circuit.parameterexpression import ParameterExpression  # needed?

        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * math.pi)

    @property
    def calibrations(self):
        """Return calibration dictionary.

        The custom pulse definition of a given gate is of the form
            {'gate_name': {(qubits, params): schedule}}
        """
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations):
        """Set the circuit calibration data from a dictionary of calibration definition.

        Args:
            calibrations (dict): A dictionary of input in the format
                {'gate_name': {(qubits, gate_params): schedule}}
        """
        self._calibrations = defaultdict(dict, calibrations)

    def to_networkx(self):
        """Returns a copy of the DAGDependency in networkx format."""
        # For backwards compatibility, return networkx structure from terra 0.12
        # where DAGNodes instances are used as indexes on the networkx graph.
        try:
            import networkx as nx
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Networkx",
                name="DAG dependency",
                pip_install="pip install networkx",
            ) from ex
        dag_networkx = nx.MultiDiGraph()

        for node in self.get_nodes():
            dag_networkx.add_node(node)
        for node in self.topological_nodes():
            for source_id, dest_id, edge in self.get_in_edges(node.node_id):
                dag_networkx.add_edge(self.get_node(source_id), self.get_node(dest_id), **edge)
        return dag_networkx

    def to_retworkx(self):
        """Returns the DAGDependency in retworkx format."""
        return self._multi_graph

    def size(self):
        """Returns the number of gates in the circuit"""
        return len(self._multi_graph)

    def depth(self):
        """Return the circuit depth.
        Returns:
            int: the circuit depth
        """
        depth = rx.dag_longest_path_length(self._multi_graph)
        return depth if depth >= 0 else 0

    def add_qubits(self, qubits):
        """Add individual qubit wires."""
        if any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise DAGDependencyError("not a Qubit instance.")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGDependencyError("duplicate qubits %s" % duplicate_qubits)

        self.qubits.extend(qubits)

    def add_clbits(self, clbits):
        """Add individual clbit wires."""
        if any(not isinstance(clbit, Clbit) for clbit in clbits):
            raise DAGDependencyError("not a Clbit instance.")

        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGDependencyError("duplicate clbits %s" % duplicate_clbits)

        self.clbits.extend(clbits)

    def add_qreg(self, qreg):
        """Add qubits in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGDependencyError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGDependencyError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])

    def add_creg(self, creg):
        """Add clbits in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGDependencyError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGDependencyError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] not in existing_clbits:
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
        Enumeration of all edges.

        Returns:
            List: corresponding to the label.
        """

        return [
            (src, dest, data)
            for src_node in self._multi_graph.nodes()
            for (src, dest, data) in self._multi_graph.out_edges(src_node.node_id)
        ]

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

        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))

    def add_op_node(self, operation, qargs, cargs):
        """Add a DAGDepNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Instruction): operation as a quantum gate.
            qargs (list[Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to.
        """
        directives = ["measure"]
        if not operation._directive and operation.name not in directives:
            qindices_list = []
            for elem in qargs:
                qindices_list.append(self.qubits.index(elem))
            if operation.condition:
                for clbit in self.clbits:
                    if clbit in operation.condition[0]:
                        initial = self.clbits.index(clbit)
                        final = self.clbits.index(clbit) + operation.condition[0].size
                        cindices_list = range(initial, final)
                        break
            else:
                cindices_list = []
        else:
            qindices_list = []
            cindices_list = []

        new_node = DAGDepNode(
            type="op",
            op=operation,
            name=operation.name,
            qargs=qargs,
            cargs=cargs,
            successors=[],
            predecessors=[],
            qindices=qindices_list,
            cindices=cindices_list,
        )
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
            merge_no_duplicates(*(self._multi_graph.get_node_data(node_id).predecessors))
        )

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
            if self._multi_graph.get_node_data(prev_node_id).reachable and not _does_commute(
                self._multi_graph.get_node_data(prev_node_id), max_node
            ):
                self._multi_graph.add_edge(prev_node_id, max_node_id, {"commute": False})
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
                merge_no_duplicates(*self._multi_graph.get_node_data(node_id).successors)
            )

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

    def draw(self, scale=0.7, filename=None, style="color"):
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

        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)


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


def _does_commute(node1, node2):
    """Function to verify commutation relation between two nodes in the DAG.

    Args:
        node1 (DAGnode): first node operation
        node2 (DAGnode): second node operation

    Return:
        bool: True if the nodes commute and false if it is not the case.
    """

    # Create set of qubits on which the operation acts
    qarg1 = [node1.qargs[i] for i in range(0, len(node1.qargs))]
    qarg2 = [node2.qargs[i] for i in range(0, len(node2.qargs))]

    # Create set of cbits on which the operation acts
    carg1 = [node1.cargs[i] for i in range(0, len(node1.cargs))]
    carg2 = [node2.cargs[i] for i in range(0, len(node2.cargs))]

    # Commutation for classical conditional gates
    # if and only if the qubits are different.
    # TODO: qubits can be the same if conditions are identical and
    # the non-conditional gates commute.
    if node1.type == "op" and node2.type == "op":
        if node1.op.condition or node2.op.condition:
            intersection = set(qarg1).intersection(set(qarg2))
            return not intersection

    # Commutation for non-unitary or parameterized or opaque ops
    # (e.g. measure, reset, directives or pulse gates)
    # if and only if the qubits and clbits are different.
    non_unitaries = ["measure", "reset", "initialize", "delay"]

    def _unknown_commutator(n):
        return n.op._directive or n.name in non_unitaries or n.op.is_parameterized()

    if _unknown_commutator(node1) or _unknown_commutator(node2):
        intersection_q = set(qarg1).intersection(set(qarg2))
        intersection_c = set(carg1).intersection(set(carg2))
        return not (intersection_q or intersection_c)

    # Known non-commuting gates (TODO: add more).
    non_commute_gates = [{"x", "y"}, {"x", "z"}]
    if qarg1 == qarg2 and ({node1.name, node2.name} in non_commute_gates):
        return False

    # Create matrices to check commutation relation if no other criteria are matched
    qarg = list(set(node1.qargs + node2.qargs))
    qbit_num = len(qarg)

    qarg1 = [qarg.index(q) for q in node1.qargs]
    qarg2 = [qarg.index(q) for q in node2.qargs]

    dim = 2 ** qbit_num
    id_op = np.reshape(np.eye(dim), (2, 2) * qbit_num)

    op1 = np.reshape(node1.op.to_matrix(), (2, 2) * len(qarg1))
    op2 = np.reshape(node2.op.to_matrix(), (2, 2) * len(qarg2))

    op = Operator._einsum_matmul(id_op, op1, qarg1)
    op12 = Operator._einsum_matmul(op, op2, qarg2, right_mul=False)
    op21 = Operator._einsum_matmul(op, op2, qarg2, shift=qbit_num, right_mul=True)

    return np.allclose(op12, op21)
