# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Object to represent a quantum circuit as a directed acyclic graph (DAG).

The nodes in the graph are either input/output nodes or operation nodes.
The edges correspond to qubits or bits in the circuit. A directed edge
from node A to node B means that the (qu)bit passes from the output of A
to the input of B. The object's methods allow circuits to be constructed,
composed, and modified. Some natural properties like depth can be computed
directly from the graph.
"""
import os
import warnings
from collections import OrderedDict
import copy
import itertools
import networkx as nx
import retworkx as rx

from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.gate import Gate
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode


# During retworkx transition, transition between 'nx' and 'rx' graph libraries
# depending on self._USE_RX.
_gls = {  # pylint: disable=invalid-name
    'nx': nx,
    'rx': rx,
}


class DAGCircuit:
    """
    Quantum circuit as a directed acyclic graph.

    There are 3 types of nodes in the graph: inputs, outputs, and operations.
    The nodes are connected by directed edges that correspond to qubits and
    bits.
    """

    # pylint: disable=invalid-name

    def __new__(cls):
        if os.environ.get('USE_RETWORKX', 'Y').lower() == 'y':
            from .retworkx_dagcircuit import RetworkxDAGCircuit  # pylint: disable=cyclic-import
            return super().__new__(RetworkxDAGCircuit)
        else:
            from .networkx_dagcircuit import NetworkxDAGCircuit  # pylint: disable=cyclic-import
            return super().__new__(NetworkxDAGCircuit)

    def __init__(self):
        """Create an empty circuit."""

        # Circuit name.  Generally, this corresponds to the name
        # of the QuantumCircuit from which the DAG was generated.
        self.name = None

        # Set of wires (Register,idx) in the dag
        self._wires = set()

        # Map from wire (Register,idx) to input nodes of the graph
        self.input_map = OrderedDict()

        # Map from wire (Register,idx) to output nodes of the graph
        self.output_map = OrderedDict()

        # Stores the max id of a node added to the DAG
        self._max_node_id = -1

        # Directed multigraph whose nodes are inputs, outputs, or operations.
        # Operation nodes have equal in- and out-degrees and carry
        # additional data about the operation, including the argument order
        # and parameter values.
        # Input nodes have out-degree 1 and output nodes have in-degree 1.
        # Edges carry wire labels (reg,idx) and each operation has
        # corresponding in- and out-edges with the same wire labels.

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of Qubit/Clbit wires that the DAG acts on.
        class DummyCallableList(list):
            """Dummy class so we can deprecate dag.qubits() and do
            dag.qubits as property.
            """
            def __call__(self):
                warnings.warn('dag.qubits() and dag.clbits() are no longer methods. Use '
                              'dag.qubits and dag.clbits properties instead.', DeprecationWarning)
                return self
        self._qubits = DummyCallableList()  # TODO: make these a regular empty list [] after the
        self._clbits = DummyCallableList()  # DeprecationWarning period, and remove name underscore.

        self._id_to_node = {}

        self._multi_graph = None
        self._gx = None
        self._USE_RX = None

    # Multigraph methods where retworkx API differs syntactically from networkx.
    def _add_multi_graph_node(self, node):
        # nx: requires manual node id handling.
        # rx: provides defined ids for added nodes.
        raise NotImplementedError()

    def _get_multi_graph_nodes(self):
        raise NotImplementedError()

    def _add_multi_graph_edge(self, src_id, dest_id, data):
        # nx: accepts edge data as kwargs.
        # rx: accepts edge data as a dict arg.
        raise NotImplementedError()

    def _get_all_multi_graph_edges(self, src_id, dest_id):
        # nx: edge enumeration through indexing multigraph
        # rx: edge enumeration through method get_all_edge_data
        raise NotImplementedError()

    def _get_multi_graph_edges(self):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        raise NotImplementedError()

    def _get_multi_graph_in_edges(self, node_id):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        raise NotImplementedError()

    def _get_multi_graph_out_edges(self, node_id):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        raise NotImplementedError()

    def to_networkx(self):
        """Returns a copy of the DAGCircuit in networkx format."""
        # For backwards compatibility, return networkx structure from terra 0.12
        # where DAGNodes instances are used as indexes on the networkx graph.

        G = nx.MultiDiGraph()
        for node in self._get_multi_graph_nodes():
            G.add_node(node)
        for node in self.topological_nodes():
            for source_id, dest_id, edge in self._get_multi_graph_in_edges(node._node_id):
                G.add_edge(self._id_to_node[source_id], self._id_to_node[dest_id],
                           **edge)

        return G

    @classmethod
    def from_networkx(cls, graph):
        """Take a networkx MultiDigraph and create a new DAGCircuit.

        Args:
            graph (networkx.MultiDiGraph): The graph to create a DAGCircuit
                object from. The format of this MultiDiGraph format must be
                in the same format as returned by to_networkx.

        Returns:
            DAGCircuit: The dagcircuit object created from the networkx
                MultiDiGraph.
        """

        dag = DAGCircuit()
        for node in nx.topological_sort(graph):
            if node.type == 'out':
                continue
            if node.type == 'in':
                dag._add_wire(node.wire)
            elif node.type == 'op':
                dag.apply_operation_back(node.op.copy(), node.qargs,
                                         node.cargs, node.condition)
        return dag

    @property
    def qubits(self):
        """Return a list of qubits (as a list of Qubit instances)."""
        # TODO: remove this property after DeprecationWarning period (~9/2020)
        return self._qubits

    @property
    def clbits(self):
        """Return a list of classical bits (as a list of Clbit instances)."""
        # TODO: remove this property after DeprecationWarning period (~9/2020)
        return self._clbits

    @property
    def wires(self):
        """Return a list of the wires in order."""
        out_list = [bit for reg in self.qregs.values() for bit in reg]
        out_list += [bit for reg in self.cregs.values() for bit in reg]
        return out_list

    @property
    def node_counter(self):
        """
        Returns the number of nodes in the dag.
        """
        return len(self._multi_graph)

    def remove_all_ops_named(self, opname):
        """Remove all operation nodes with the given name."""
        for n in self.named_nodes(opname):
            self.remove_op_node(n)

    def add_qreg(self, qreg):
        """Add all wires in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGCircuitError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGCircuitError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg
        for j in range(qreg.size):
            self.qubits.append(qreg[j])
            self._add_wire(qreg[j])

    def add_creg(self, creg):
        """Add all wires in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGCircuitError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGCircuitError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        for j in range(creg.size):
            self.clbits.append(creg[j])
            self._add_wire(creg[j])

    def _add_wire(self, wire):
        """Add a qubit or bit to the circuit.

        Args:
            wire (Bit): the wire to be added
            This adds a pair of in and out nodes connected by an edge.

        Raises:
            DAGCircuitError: if trying to add duplicate wire
        """
        if wire not in self._wires:
            self._wires.add(wire)

            wire_name = "%s[%s]" % (wire.register.name, wire.index)

            inp_node = DAGNode(type='in', name=wire_name, wire=wire)
            outp_node = DAGNode(type='out', name=wire_name, wire=wire)

            inp_node_id = self._add_multi_graph_node(inp_node)
            outp_node_id = self._add_multi_graph_node(outp_node)

            self.input_map[wire] = inp_node
            self.output_map[wire] = outp_node

            self._add_multi_graph_edge(inp_node._node_id, outp_node._node_id,
                                       {'name': wire_name, 'wire': wire})

        else:
            raise DAGCircuitError("duplicate wire %s" % (wire,))

    def _check_condition(self, name, condition):
        """Verify that the condition is valid.

        Args:
            name (string): used for error reporting
            condition (tuple or None): a condition tuple (ClassicalRegister,int)

        Raises:
            DAGCircuitError: if conditioning on an invalid register
        """
        # Verify creg exists
        if condition is not None and condition[0].name not in self.cregs:
            raise DAGCircuitError("invalid creg in condition for %s" % name)

    def _check_bits(self, args, amap):
        """Check the values of a list of (qu)bit arguments.

        For each element of args, check that amap contains it.

        Args:
            args (list[Bit]): the elements to be checked
            amap (dict): a dictionary keyed on Qubits/Clbits

        Raises:
            DAGCircuitError: if a qubit is not contained in amap
        """
        # Check for each wire
        for wire in args:
            if wire not in amap:
                raise DAGCircuitError("(qu)bit %s[%d] not found" %
                                      (wire.register.name, wire.index))

    def _bits_in_condition(self, cond):
        """Return a list of bits in the given condition.

        Args:
            cond (tuple or None): optional condition (ClassicalRegister, int)

        Returns:
            list[Clbit]: list of classical bits
        """
        return [] if cond is None else list(cond[0])

    def _add_op_node(self, op, qargs, cargs, condition=None):
        """Add a new operation node to the graph and assign properties.

        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): list of quantum wires to attach to.
            cargs (list[Clbit]): list of classical wires to attach to.
            condition (tuple or None): optional condition (ClassicalRegister, int)
        Returns:
            DAGNode: The node for the new op on the DAG
        """
        # Add a new operation node to the graph
        new_node = DAGNode(type="op", op=op, name=op.name, qargs=qargs,
                           cargs=cargs, condition=condition)
        self._add_multi_graph_node(new_node)
        return new_node

    def apply_operation_back(self, op, qargs=None, cargs=None, condition=None):
        """Apply an operation to the output of the circuit.

        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): qubits that op will be applied to
            cargs (list[Clbit]): cbits that op will be applied to
            condition (tuple or None): optional condition (ClassicalRegister, int)

        Returns:
            DAGNode: the current max node

        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs

        """
        qargs = qargs or []
        cargs = cargs or []

        all_cbits = self._bits_in_condition(condition)
        all_cbits = set(all_cbits).union(cargs)

        self._check_condition(op.name, condition)
        self._check_bits(qargs, self.output_map)
        self._check_bits(all_cbits, self.output_map)

        node = self._add_op_node(op, qargs, cargs, condition)

        # Add new in-edges from predecessors of the output nodes to the
        # operation node while deleting the old in-edges of the output nodes
        # and adding new edges from the operation node to each output node
        al = [qargs, all_cbits]
        for q in itertools.chain(*al):
            ie = list(self.predecessors(self.output_map[q]))

            if len(ie) != 1:
                raise DAGCircuitError("output node has multiple in-edges")

            self._add_multi_graph_edge(ie[0]._node_id, node._node_id,
                                       {'name': "%s[%s]" % (q.register.name, q.index), 'wire': q})
            self._multi_graph.remove_edge(ie[0]._node_id, self.output_map[q]._node_id)
            self._add_multi_graph_edge(node._node_id, self.output_map[q]._node_id,
                                       {'name': "%s[%s]" % (q.register.name, q.index), 'wire': q})

        return node

    def apply_operation_front(self, op, qargs, cargs, condition=None):
        """Apply an operation to the input of the circuit.

        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): qubits that op will be applied to
            cargs (list[Clbit]): cbits that op will be applied to
            condition (tuple or None): optional condition (ClassicalRegister, value)

        Returns:
            DAGNode: the current max node

        Raises:
            DAGCircuitError: if initial nodes connected to multiple out edges
        """
        all_cbits = self._bits_in_condition(condition)
        all_cbits.extend(cargs)

        self._check_condition(op.name, condition)
        self._check_bits(qargs, self.input_map)
        self._check_bits(all_cbits, self.input_map)
        node = self._add_op_node(op, qargs, cargs, condition)
        # Add new out-edges to successors of the input nodes from the
        # operation node while deleting the old out-edges of the input nodes
        # and adding new edges to the operation node from each input node
        al = [qargs, all_cbits]
        for q in itertools.chain(*al):
            ie = list(self.successors(self.input_map[q]))
            if len(ie) != 1:
                raise DAGCircuitError("input node has multiple out-edges")
            self._add_multi_graph_edge(node._node_id, ie[0]._node_id,
                                       {'name': "%s[%s]" % (q.register.name, q.index), 'wire': q})
            self._multi_graph.remove_edge(self.input_map[q]._node_id, ie[0]._node_id)
            self._add_multi_graph_edge(self.input_map[q]._node_id, node._node_id,
                                       {'name': "%s[%s]" % (q.register.name, q.index), 'wire': q})

        return node

    def _check_edgemap_registers(self, edge_map, keyregs, valregs, valreg=True):
        """Check that wiremap neither fragments nor leaves duplicate registers.

        1. There are no fragmented registers. A register in keyregs
        is fragmented if not all of its (qu)bits are renamed by edge_map.
        2. There are no duplicate registers. A register is duplicate if
        it appears in both self and keyregs but not in edge_map.

        Args:
            edge_map (dict): map from Bit in keyregs to Bit in valregs
            keyregs (dict): a map from register names to Register objects
            valregs (dict): a map from register names to Register objects
            valreg (bool): if False the method ignores valregs and does not
                add regs for bits in the edge_map image that don't appear in valregs

        Returns:
            set(Register): the set of regs to add to self

        Raises:
            DAGCircuitError: if the wiremap fragments, or duplicates exist
        """
        # FIXME: some mixing of objects and strings here are awkward (due to
        # self.qregs/self.cregs still keying on string.
        add_regs = set()
        reg_frag_chk = {}
        for v in keyregs.values():
            reg_frag_chk[v] = {j: False for j in range(len(v))}
        for k in edge_map.keys():
            if k.register.name in keyregs:
                reg_frag_chk[k.register][k.index] = True
        for k, v in reg_frag_chk.items():
            s = set(v.values())
            if len(s) == 2:
                raise DAGCircuitError("edge_map fragments reg %s" % k)
            if s == {False}:
                if k in self.qregs.values() or k in self.cregs.values():
                    raise DAGCircuitError("unmapped duplicate reg %s" % k)
                # Add registers that appear only in keyregs
                add_regs.add(k)
            else:
                if valreg:
                    # If mapping to a register not in valregs, add it.
                    # (k,0) exists in edge_map because edge_map doesn't
                    # fragment k
                    if not edge_map[k[0]].register.name in valregs:
                        size = max(map(lambda x: x.index,
                                       filter(lambda x: x.register == edge_map[k[0]].register,
                                              edge_map.values())))
                        qreg = QuantumRegister(size + 1, edge_map[k[0]].register.name)
                        add_regs.add(qreg)
        return add_regs

    def _check_wiremap_validity(self, wire_map, keymap, valmap):
        """Check that the wiremap is consistent.

        Check that the wiremap refers to valid wires and that
        those wires have consistent types.

        Args:
            wire_map (dict): map from Bit in keymap to Bit in valmap
            keymap (dict): a map whose keys are wire_map keys
            valmap (dict): a map whose keys are wire_map values

        Raises:
            DAGCircuitError: if wire_map not valid
        """
        for k, v in wire_map.items():
            kname = "%s[%d]" % (k.register.name, k.index)
            vname = "%s[%d]" % (v.register.name, v.index)
            if k not in keymap:
                raise DAGCircuitError("invalid wire mapping key %s" % kname)
            if v not in valmap:
                raise DAGCircuitError("invalid wire mapping value %s" % vname)
            if type(k) is not type(v):
                raise DAGCircuitError("inconsistent wire_map at (%s,%s)" %
                                      (kname, vname))

    def _map_condition(self, wire_map, condition):
        """Use the wire_map dict to change the condition tuple's creg name.

        Args:
            wire_map (dict): a map from wires to wires
            condition (tuple or None): (ClassicalRegister,int)
        Returns:
            tuple(ClassicalRegister,int): new condition
        """
        if condition is None:
            new_condition = None
        else:
            # Map the register name, using fact that registers must not be
            # fragmented by the wire_map (this must have been checked
            # elsewhere)
            bit0 = condition[0][0]
            new_condition = (wire_map.get(bit0, bit0).register, condition[1])
        return new_condition

    def extend_back(self, dag, edge_map=None):
        """DEPRECATED: Add `dag` at the end of `self`, using `edge_map`.
        """
        warnings.warn("dag.extend_back is deprecated, please use dag.compose.",
                      DeprecationWarning, stacklevel=2)
        edge_map = edge_map or {}
        for qreg in dag.qregs.values():
            if qreg.name not in self.qregs:
                self.add_qreg(QuantumRegister(qreg.size, qreg.name))
            edge_map.update([(qbit, qbit) for qbit in qreg if qbit not in edge_map])

        for creg in dag.cregs.values():
            if creg.name not in self.cregs:
                self.add_creg(ClassicalRegister(creg.size, creg.name))
            edge_map.update([(cbit, cbit) for cbit in creg if cbit not in edge_map])

        self.compose_back(dag, edge_map)

    def compose_back(self, input_circuit, edge_map=None):
        """DEPRECATED: use DAGCircuit.compose() instead.
        """
        warnings.warn("dag.compose_back is deprecated, please use dag.compose.",
                      DeprecationWarning, stacklevel=2)
        self.compose(input_circuit, edge_map)

    def compose(self, other, edge_map=None, qubits=None, clbits=None, front=False, inplace=True):
        """Compose the ``other`` circuit onto the output of this circuit.

        A subset of input wires of ``other`` are mapped
        to a subset of output wires of this circuit.

        ``other`` can be narrower or of equal width to ``self``.

        Args:
            other (DAGCircuit): circuit to compose with self
            edge_map (dict): DEPRECATED - a {Bit: Bit} map from input wires of other
                to output wires of self (i.e. rhs->lhs).
                The key, value pairs can be either Qubit or Clbit mappings.
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If True, front composition will be performed (not implemented yet)
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Returns:
            DAGCircuit: the composed dag (returns None if inplace==True).

        Raises:
            DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.
        """
        if front:
            raise DAGCircuitError("Front composition not supported yet.")

        if len(other.qubits) > len(self.qubits) or \
           len(other.clbits) > len(self.clbits):
            raise DAGCircuitError("Trying to compose with another DAGCircuit "
                                  "which has more 'in' edges.")

        if edge_map is not None:
            warnings.warn("edge_map arg as a dictionary is deprecated. "
                          "Use qubits and clbits args to specify a list of "
                          "self edges to compose onto.", DeprecationWarning,
                          stacklevel=2)
        if qubits is None:
            qubits = []
        if clbits is None:
            clbits = []
        qubit_map = {other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
                     for i, q in enumerate(qubits)}
        clbit_map = {other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
                     for i, c in enumerate(clbits)}
        edge_map = edge_map or {**qubit_map, **clbit_map} or None
        # if no edge_map, try to do a 1-1 mapping in order
        if edge_map is None:
            identity_qubit_map = dict(zip(other.qubits, self.qubits))
            identity_clbit_map = dict(zip(other.clbits, self.clbits))
            edge_map = {**identity_qubit_map, **identity_clbit_map}

        # Check the edge_map for duplicate values
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError("duplicates in wire_map")

        # Compose
        if inplace:
            dag = self
        else:
            dag = copy.deepcopy(self)

        for nd in other.topological_nodes():
            if nd.type == "in":
                # if in edge_map, get new name, else use existing name
                m_wire = edge_map.get(nd.wire, nd.wire)
                # the mapped wire should already exist
                if m_wire not in dag.output_map:
                    raise DAGCircuitError("wire %s[%d] not in self" % (
                        m_wire.register.name, m_wire.index))
                if nd.wire not in other._wires:
                    raise DAGCircuitError("inconsistent wire type for %s[%d] in other"
                                          % (nd.register.name, nd.wire.index))
            elif nd.type == "out":
                # ignore output nodes
                pass
            elif nd.type == "op":
                condition = dag._map_condition(edge_map, nd.condition)
                dag._check_condition(nd.name, condition)
                m_qargs = list(map(lambda x: edge_map.get(x, x), nd.qargs))
                m_cargs = list(map(lambda x: edge_map.get(x, x), nd.cargs))
                dag.apply_operation_back(nd.op, m_qargs, m_cargs, condition)
            else:
                raise DAGCircuitError("bad node type %s" % nd.type)

        if not inplace:
            return dag
        else:
            return None

    def idle_wires(self, ignore=None):
        """Return idle wires.

        Args:
            ignore (list(str)): List of node names to ignore. Default: []

        Yields:
            Bit: Bit in idle wire.
        """
        if ignore is None:
            ignore = []
        for wire in self._wires:
            nodes = [node for node in self.nodes_on_wire(wire, only_ops=False)
                     if node.name not in ignore]
            if len(nodes) == 2:
                yield wire

    def size(self):
        """Return the number of operations."""
        return len(self._multi_graph) - 2 * len(self._wires)

    def depth(self):
        """Return the circuit depth.
        Returns:
            int: the circuit depth
        Raises:
            DAGCircuitError: if not a directed acyclic graph
        """
        if not _gls[self._gx].is_directed_acyclic_graph(self._multi_graph):
            raise DAGCircuitError("not a DAG")

        depth = _gls[self._gx].dag_longest_path_length(self._multi_graph) - 1
        return depth if depth >= 0 else 0

    def width(self):
        """Return the total number of qubits + clbits used by the circuit.
           This function formerly returned the number of qubits by the calculation
           return len(self._wires) - self.num_clbits()
           but was changed by issue #2564 to return number of qubits + clbits
           with the new function DAGCircuit.num_qubits replacing the former
           semantic of DAGCircuit.width().
        """
        return len(self._wires)

    def num_qubits(self):
        """Return the total number of qubits used by the circuit.
           num_qubits() replaces former use of width().
           DAGCircuit.width() now returns qubits + clbits for
           consistency with Circuit.width() [qiskit-terra #2564].
        """
        return len(self._wires) - self.num_clbits()

    def num_clbits(self):
        """Return the total number of classical bits used by the circuit."""
        return sum(creg.size for creg in self.cregs.values())

    def num_tensor_factors(self):
        """Compute how many components the circuit can decompose into."""
        return _gls[self._gx].number_weakly_connected_components(self._multi_graph)

    def _check_wires_list(self, wires, node):
        """Check that a list of wires is compatible with a node to be replaced.

        - no duplicate names
        - correct length for operation
        Raise an exception otherwise.

        Args:
            wires (list[Bit]): gives an order for (qu)bits
                in the input circuit that is replacing the node.
            node (DAGNode): a node in the dag

        Raises:
            DAGCircuitError: if check doesn't pass.
        """
        if len(set(wires)) != len(wires):
            raise DAGCircuitError("duplicate wires")

        wire_tot = len(node.qargs) + len(node.cargs)
        if node.condition is not None:
            wire_tot += node.condition[0].size

        if len(wires) != wire_tot:
            raise DAGCircuitError("expected %d wires, got %d"
                                  % (wire_tot, len(wires)))

    def _make_pred_succ_maps(self, node):
        """Return predecessor and successor dictionaries.

        Args:
            node (DAGNode): reference to multi_graph node

        Returns:
            tuple(dict): tuple(predecessor_map, successor_map)
                These map from wire (Register, int) to predecessor (successor)
                nodes of n.
        """

        pred_map = {e[2]['wire']: self._id_to_node[e[0]] for e in
                    self._get_multi_graph_in_edges(node._node_id)}
        succ_map = {e[2]['wire']: self._id_to_node[e[1]] for e in
                    self._get_multi_graph_out_edges(node._node_id)}
        return pred_map, succ_map

    def _full_pred_succ_maps(self, pred_map, succ_map, input_circuit,
                             wire_map):
        """Map all wires of the input circuit.

        Map all wires of the input circuit to predecessor and
        successor nodes in self, keyed on wires in self.

        Args:
            pred_map (dict): comes from _make_pred_succ_maps
            succ_map (dict): comes from _make_pred_succ_maps
            input_circuit (DAGCircuit): the input circuit
            wire_map (dict): the map from wires of input_circuit to wires of self

        Returns:
            tuple: full_pred_map, full_succ_map (dict, dict)

        Raises:
            DAGCircuitError: if more than one predecessor for output nodes
        """
        full_pred_map = {}
        full_succ_map = {}
        for w in input_circuit.input_map:
            # If w is wire mapped, find the corresponding predecessor
            # of the node
            if w in wire_map:
                full_pred_map[wire_map[w]] = pred_map[wire_map[w]]
                full_succ_map[wire_map[w]] = succ_map[wire_map[w]]
            else:
                # Otherwise, use the corresponding output nodes of self
                # and compute the predecessor.
                full_succ_map[w] = self.output_map[w]
                full_pred_map[w] = self.predecessors(self.output_map[w])[0]
                if len(list(self.predecessors(self.output_map[w]))) != 1:
                    raise DAGCircuitError("too many predecessors for %s[%d] "
                                          "output node" % (w.register, w.index))

        return full_pred_map, full_succ_map

    def __eq__(self, other):
        raise NotImplementedError()

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order
        """
        raise NotImplementedError()

    def topological_op_nodes(self):
        """
        Yield op nodes in topological order.

        Returns:
            generator(DAGNode): op node in topological order
        """
        return (nd for nd in self.topological_nodes() if nd.type == 'op')

    def substitute_node_with_dag(self, node, input_dag, wires=None):
        """Replace one node with dag.

        Args:
            node (DAGNode): node to substitute
            input_dag (DAGCircuit): circuit that will substitute the node
            wires (list[Bit]): gives an order for (qu)bits
                in the input circuit. This order gets matched to the node wires
                by qargs first, then cargs, then conditions.

        Raises:
            DAGCircuitError: if met with unexpected predecessor/successors
        """
        condition = node.condition
        # the dag must be amended if used in a
        # conditional context. delete the op nodes and replay
        # them with the condition.
        if condition:
            input_dag.add_creg(condition[0])
            to_replay = []
            for sorted_node in input_dag.topological_nodes():
                if sorted_node.type == "op":
                    sorted_node.op.condition = condition
                    to_replay.append(sorted_node)
            for input_node in input_dag.op_nodes():
                input_dag.remove_op_node(input_node)
            for replay_node in to_replay:
                input_dag.apply_operation_back(replay_node.op, replay_node.qargs,
                                               replay_node.cargs, condition=condition)

        if wires is None:
            wires = input_dag.wires

        self._check_wires_list(wires, node)

        # Create a proxy wire_map to identify fragments and duplicates
        # and determine what registers need to be added to self
        proxy_map = {w: QuantumRegister(1, 'proxy') for w in wires}
        add_qregs = self._check_edgemap_registers(proxy_map,
                                                  input_dag.qregs,
                                                  {}, False)
        for qreg in add_qregs:
            self.add_qreg(qreg)

        add_cregs = self._check_edgemap_registers(proxy_map,
                                                  input_dag.cregs,
                                                  {}, False)
        for creg in add_cregs:
            self.add_creg(creg)

        # Replace the node by iterating through the input_circuit.
        # Constructing and checking the validity of the wire_map.
        # If a gate is conditioned, we expect the replacement subcircuit
        # to depend on those condition bits as well.
        if node.type != "op":
            raise DAGCircuitError("expected node type \"op\", got %s"
                                  % node.type)

        condition_bit_list = self._bits_in_condition(node.condition)

        wire_map = dict(zip(wires, list(node.qargs) + list(node.cargs) + list(condition_bit_list)))
        self._check_wiremap_validity(wire_map, wires, self.input_map)
        pred_map, succ_map = self._make_pred_succ_maps(node)
        full_pred_map, full_succ_map = self._full_pred_succ_maps(pred_map, succ_map,
                                                                 input_dag, wire_map)

        if condition_bit_list:
            # If we are replacing a conditional node, map input dag through
            # wire_map to verify that it will not modify any of the conditioning
            # bits.
            condition_bits = set(condition_bit_list)

            for op_node in input_dag.op_nodes():
                mapped_cargs = {wire_map[carg] for carg in op_node.cargs}

                if condition_bits & mapped_cargs:
                    raise DAGCircuitError('Mapped DAG would alter clbits '
                                          'on which it would be conditioned.')

        # Now that we know the connections, delete node
        self._multi_graph.remove_node(node._node_id)

        # Iterate over nodes of input_circuit
        for sorted_node in input_dag.topological_op_nodes():
            # Insert a new node
            condition = self._map_condition(wire_map, sorted_node.condition)
            m_qargs = list(map(lambda x: wire_map.get(x, x),
                               sorted_node.qargs))
            m_cargs = list(map(lambda x: wire_map.get(x, x),
                               sorted_node.cargs))
            node = self._add_op_node(sorted_node.op, m_qargs, m_cargs, condition)
            # Add edges from predecessor nodes to new node
            # and update predecessor nodes that change
            all_cbits = self._bits_in_condition(condition)
            all_cbits.extend(m_cargs)
            al = [m_qargs, all_cbits]
            for q in itertools.chain(*al):
                self._add_multi_graph_edge(full_pred_map[q]._node_id,
                                           node._node_id,
                                           dict(name="%s[%s]" % (q.register.name, q.index),
                                                wire=q))
                full_pred_map[q] = node

        # Connect all predecessors and successors, and remove
        # residual edges between input and output nodes
        for w in full_pred_map:
            self._add_multi_graph_edge(full_pred_map[w]._node_id,
                                       full_succ_map[w]._node_id,
                                       dict(name="%s[%s]" % (w.register.name, w.index),
                                            wire=w))
            o_pred = list(self.predecessors(self.output_map[w]))
            if len(o_pred) > 1:
                if len(o_pred) != 2:
                    raise DAGCircuitError("expected 2 predecessors here")

                p = [x for x in o_pred if x != full_pred_map[w]]
                if len(p) != 1:
                    raise DAGCircuitError("expected 1 predecessor to pass filter")

                self._multi_graph.remove_edge(p[0], self.output_map[w])

    def substitute_node(self, node, op, inplace=False):
        """Replace a DAGNode with a single instruction. qargs, cargs and
        conditions for the new instruction will be inferred from the node to be
        replaced. The new instruction will be checked to match the shape of the
        replaced instruction.

        Args:
            node (DAGNode): Node to be replaced
            op (qiskit.circuit.Instruction): The :class:`qiskit.circuit.Instruction`
                instance to be added to the DAG
            inplace (bool): Optional, default False. If True, existing DAG node
                will be modified to include op. Otherwise, a new DAG node will
                be used.

        Returns:
            DAGNode: the new node containing the added instruction.

        Raises:
            DAGCircuitError: If replacement instruction was incompatible with
            location of target node.
        """

        if node.type != 'op':
            raise DAGCircuitError('Only DAGNodes of type "op" can be replaced.')

        if (
                node.op.num_qubits != op.num_qubits
                or node.op.num_clbits != op.num_clbits
        ):
            raise DAGCircuitError(
                'Cannot replace node of width ({} qubits, {} clbits) with '
                'instruction of mismatched width ({} qubits, {} clbits).'.format(
                    node.op.num_qubits, node.op.num_clbits,
                    op.num_qubits, op.num_clbits))

        if inplace:
            node.op = op
            node.name = op.name
            return node

        new_node = copy.copy(node)
        new_node.op = op
        new_node.name = op.name

        node_index = self._add_multi_graph_node(new_node)

        in_edges = self._get_multi_graph_in_edges(node._node_id)
        out_edges = self._get_multi_graph_out_edges(node._node_id)

        for src_id, _, data in in_edges:
            self._add_multi_graph_edge(src_id, node_index, data)
        for _, dest_id, data in out_edges:
            self._add_multi_graph_edge(node_index, dest_id, data)

        self._multi_graph.remove_node(node._node_id)

        return new_node

    def node(self, node_id):
        """Get the node in the dag.

        Args:
            node_id(int): Node identifier.

        Returns:
            node: the node.
        """
        return self._id_to_node[node_id]

    def nodes(self):
        """Iterator for node values.

        Yield:
            node: the node.
        """
        for node in self._get_multi_graph_nodes():
            yield node

    def edges(self, nodes=None):
        """Iterator for node values.

        Yield:
            node: the node.
        """
        if nodes is None:
            nodes = self._get_multi_graph_nodes()
        elif isinstance(nodes, DAGNode):
            nodes = [nodes]

        for node in nodes:
            raw_nodes = self._get_multi_graph_out_edges(node._node_id)
            for source, dest, edge in raw_nodes:
                yield (self._id_to_node[source],
                       self._id_to_node[dest],
                       edge)

    def op_nodes(self, op=None, include_directives=True):
        """Get the list of "op" nodes in the dag.

        Args:
            op (qiskit.circuit.Instruction): op nodes to return.
                If None, return all op nodes.
            include_directives (bool): include `barrier`, `snapshot` etc.

        Returns:
            list[DAGNode]: the list of node ids containing the given op.
        """
        nodes = []
        for node in self._get_multi_graph_nodes():
            if node.type == "op":
                if not include_directives and node.name in ['snapshot', 'barrier']:
                    continue
                if op is None or isinstance(node.op, op):
                    nodes.append(node)
        return nodes

    def gate_nodes(self):
        """Get the list of gate nodes in the dag.

        Returns:
            list[DAGNode]: the list of DAGNodes that represent gates.
        """
        nodes = []
        for node in self.op_nodes():
            if isinstance(node.op, Gate):
                nodes.append(node)
        return nodes

    def named_nodes(self, *names):
        """Get the set of "op" nodes with the given name."""
        named_nodes = []
        for node in self._get_multi_graph_nodes():
            if node.type == 'op' and node.op.name in names:
                named_nodes.append(node)
        return named_nodes

    def twoQ_gates(self):
        """Get list of 2-qubit gates. Ignore snapshot, barriers, and the like."""
        warnings.warn('deprecated function, use dag.two_qubit_ops(). '
                      'filter output by isinstance(op, Gate) to only get unitary Gates.',
                      DeprecationWarning, stacklevel=2)
        two_q_gates = []
        for node in self.gate_nodes():
            if len(node.qargs) == 2:
                two_q_gates.append(node)
        return two_q_gates

    def threeQ_or_more_gates(self):
        """Get list of 3-or-more-qubit gates: (id, data)."""
        warnings.warn('deprecated function, use dag.multi_qubit_ops(). '
                      'filter output by isinstance(op, Gate) to only get unitary Gates.',
                      DeprecationWarning, stacklevel=2)
        three_q_gates = []
        for node in self.gate_nodes():
            if len(node.qargs) >= 3:
                three_q_gates.append(node)
        return three_q_gates

    def two_qubit_ops(self):
        """Get list of 2 qubit operations. Ignore directives like snapshot and barrier."""
        ops = []
        for node in self.op_nodes(include_directives=False):
            if len(node.qargs) == 2:
                ops.append(node)
        return ops

    def multi_qubit_ops(self):
        """Get list of 3+ qubit operations. Ignore directives like snapshot and barrier."""
        ops = []
        for node in self.op_nodes(include_directives=False):
            if len(node.qargs) >= 3:
                ops.append(node)
        return ops

    def longest_path(self):
        """Returns the longest path in the dag as a list of DAGNodes."""
        return [self._id_to_node[idx]
                for idx in _gls[self._gx].dag_longest_path(self._multi_graph)]

    def successors(self, node):
        """Returns iterator of the successors of a node as DAGNodes."""
        raise NotImplementedError()

    def predecessors(self, node):
        """Returns iterator of the predecessors of a node as DAGNodes."""
        raise NotImplementedError()

    def quantum_predecessors(self, node):
        """Returns iterator of the predecessors of a node that are
        connected by a quantum edge as DAGNodes."""
        for predecessor in self.predecessors(node):
            if any(isinstance(x['wire'], Qubit) for x in
                   self._get_all_multi_graph_edges(predecessor._node_id, node._node_id)):
                yield predecessor

    def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGNodes."""
        return set(self._id_to_node[idx]
                   for idx in _gls[self._gx].ancestors(self._multi_graph, node._node_id))

    def descendants(self, node):
        """Returns set of the descendants of a node as DAGNodes."""
        return set(self._id_to_node[idx]
                   for idx in _gls[self._gx].descendants(self._multi_graph, node._node_id))

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
        and [DAGNode] is its successors in  BFS order.
        """
        raise NotImplementedError()

    def quantum_successors(self, node):
        """Returns iterator of the successors of a node that are
        connected by a quantum edge as DAGNodes."""
        for successor in self.successors(node):
            if any(isinstance(x['wire'], Qubit)
                   for x in
                   self._get_all_multi_graph_edges(
                       node._node_id, successor._node_id)):
                yield successor

    def remove_op_node(self, node):
        """Remove an operation node n.

        Add edges from predecessors to successors.
        """
        if node.type != 'op':
            raise DAGCircuitError('The method remove_op_node only works on op node types. An "%s" '
                                  'node type was wrongly provided.' % node.type)

        pred_map, succ_map = self._make_pred_succ_maps(node)

        # remove from graph and map
        self._multi_graph.remove_node(node._node_id)

        for w in pred_map.keys():
            self._add_multi_graph_edge(pred_map[w]._node_id, succ_map[w]._node_id,
                                       {'name': "%s[%s]" % (w.register.name, w.index), 'wire': w})

    def remove_ancestors_of(self, node):
        """Remove all of the ancestor operation nodes of node."""
        anc = _gls[self._gx].ancestors(self._multi_graph, node)
        # TODO: probably better to do all at once using
        # multi_graph.remove_nodes_from; same for related functions ...
        for anc_node in anc:
            if anc_node.type == "op":
                self.remove_op_node(anc_node)

    def remove_descendants_of(self, node):
        """Remove all of the descendant operation nodes of node."""
        desc = _gls[self._gx].descendants(self._multi_graph, node)
        for desc_node in desc:
            if desc_node.type == "op":
                self.remove_op_node(desc_node)

    def remove_nonancestors_of(self, node):
        """Remove all of the non-ancestors operation nodes of node."""
        anc = _gls[self._gx].ancestors(self._multi_graph, node)
        comp = list(set(self._get_multi_graph_nodes()) - set(anc))
        for n in comp:
            if n.type == "op":
                self.remove_op_node(n)

    def remove_nondescendants_of(self, node):
        """Remove all of the non-descendants operation nodes of node."""
        dec = _gls[self._gx].descendants(self._multi_graph, node)
        comp = list(set(self._get_multi_graph_nodes()) - set(dec))
        for n in comp:
            if n.type == "op":
                self.remove_op_node(n)

    def layers(self):
        """Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.

        A layer is a circuit whose gates act on disjoint qubits, i.e.,
        a layer has depth 1. The total number of layers equals the
        circuit depth d. The layers are indexed from 0 to d-1 with the
        earliest layer at index 0. The layers are constructed using a
        greedy algorithm. Each returned layer is a dict containing
        {"graph": circuit graph, "partition": list of qubit lists}.

        New but semantically equivalent DAGNodes will be included in the returned layers,
        NOT the DAGNodes from the original DAG. The original vs. new nodes can be compared using
        DAGNode.semantic_eq(node1, node2).

        TODO: Gates that use the same cbits will end up in different
        layers as this is currently implemented. This may not be
        the desired behavior.
        """
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)  # Remove input nodes
        except StopIteration:
            return

        for graph_layer in graph_layers:

            # Get the op nodes from the layer, removing any input and output nodes.
            op_nodes = [node for node in graph_layer if node.type == "op"]

            # Sort to make sure they are in the order they were added to the original DAG
            # It has to be done by node_id as graph_layer is just a list of nodes
            # with no implied topology
            # Drawing tools that rely on _node_id to infer order of node creation
            # so we need this to be preserved by layers()
            op_nodes.sort(key=lambda nd: nd._node_id)

            # Stop yielding once there are no more op_nodes in a layer.
            if not op_nodes:
                return

            # Construct a shallow copy of self
            new_layer = DAGCircuit()
            new_layer.name = self.name

            # add in the registers - this adds the input/output nodes
            for creg in self.cregs.values():
                new_layer.add_creg(creg)
            for qreg in self.qregs.values():
                new_layer.add_qreg(qreg)

            for node in op_nodes:
                # this creates new DAGNodes in the new_layer
                new_layer.apply_operation_back(node.op,
                                               node.qargs,
                                               node.cargs,
                                               node.condition)

            # The quantum registers that have an operation in this layer.
            support_list = [
                op_node.qargs
                for op_node in new_layer.op_nodes()
                if op_node.name not in {"barrier", "snapshot", "save", "load", "noise"}
            ]

            yield {"graph": new_layer, "partition": support_list}

    def serial_layers(self):
        """Yield a layer for all gates of this circuit.

        A serial layer is a circuit with one gate. The layers have the
        same structure as in layers().
        """
        for next_node in self.topological_op_nodes():
            new_layer = DAGCircuit()
            for qreg in self.qregs.values():
                new_layer.add_qreg(qreg)
            for creg in self.cregs.values():
                new_layer.add_creg(creg)
            # Save the support of the operation we add to the layer
            support_list = []
            # Operation data
            op = copy.copy(next_node.op)
            qa = copy.copy(next_node.qargs)
            ca = copy.copy(next_node.cargs)
            co = copy.copy(next_node.condition)
            _ = self._bits_in_condition(co)

            # Add node to new_layer
            new_layer.apply_operation_back(op, qa, ca, co)
            # Add operation to partition
            if next_node.name not in ["barrier",
                                      "snapshot", "save", "load", "noise"]:
                support_list.append(list(qa))
            l_dict = {"graph": new_layer, "partition": support_list}
            yield l_dict

    def multigraph_layers(self):
        """Yield layers of the multigraph."""
        raise NotImplementedError()

    def collect_runs(self, namelist):
        """Return a set of non-conditional runs of "op" nodes with the given names.

        For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."
        would produce the tuple of cx nodes as an element of the set returned
        from a call to collect_runs(["cx"]). If instead the cx nodes were
        "cx q[0],q[1]; cx q[1],q[0];", the method would still return the
        pair in a tuple. The namelist can contain names that are not
        in the circuit's basis.

        Nodes must have only one successor to continue the run.
        """
        group_list = []

        # Iterate through the nodes of self in topological order
        # and form tuples containing sequences of gates
        # on the same qubit(s).
        topo_ops = list(self.topological_op_nodes())
        nodes_seen = dict(zip(topo_ops, [False] * len(topo_ops)))
        for node in topo_ops:
            if node.name in namelist and node.condition is None \
                    and not nodes_seen[node]:
                group = [node]
                nodes_seen[node] = True
                s = list(self.successors(node))
                while len(s) == 1 and \
                        s[0].type == "op" and \
                        s[0].name in namelist and \
                        s[0].condition is None:
                    group.append(s[0])
                    nodes_seen[s[0]] = True
                    s = list(self.successors(s[0]))
                if len(group) >= 1:
                    group_list.append(tuple(group))
        return set(group_list)

    def nodes_on_wire(self, wire, only_ops=False):
        """
        Iterator for nodes that affect a given wire.

        Args:
            wire (Bit): the wire to be looked at.
            only_ops (bool): True if only the ops nodes are wanted;
                        otherwise, all nodes are returned.
        Yield:
             DAGNode: the successive ops on the given wire

        Raises:
            DAGCircuitError: if the given wire doesn't exist in the DAG
        """
        current_node = self.input_map.get(wire, None)

        if not current_node:
            raise DAGCircuitError('The given wire %s is not present in the circuit'
                                  % str(wire))

        more_nodes = True
        while more_nodes:
            more_nodes = False
            # allow user to just get ops on the wire - not the input/output nodes
            if current_node.type == 'op' or not only_ops:
                yield current_node

            # find the adjacent node that takes the wire being looked at as input
            # TODO(mtreinish): Add function in retworkx that does this nested api
            for _, node_index, __ in self._get_multi_graph_out_edges(current_node._node_id):
                node = self._id_to_node[node_index]
                if self._multi_graph.has_edge(current_node._node_id,
                                              node_index):
                    edge_data = self._get_all_multi_graph_edges(
                        current_node._node_id, node_index)
                else:
                    edge_data = self._get_all_multi_graph_edges(
                        node_index, current_node._node_id)
                if any(wire == edge['wire'] for edge in edge_data):
                    current_node = node
                    more_nodes = True
                    break

    def count_ops(self):
        """Count the occurrences of operation names.

        Returns a dictionary of counts keyed on the operation name.
        """
        op_dict = {}
        for node in self.topological_op_nodes():
            name = node.name
            if name not in op_dict:
                op_dict[name] = 1
            else:
                op_dict[name] += 1
        return op_dict

    def count_ops_longest_path(self):
        """Count the occurrences of operation names on the longest path.

        Returns a dictionary of counts keyed on the operation name.
        """
        op_dict = {}
        path = self.longest_path()
        path = path[1:-1]     # remove qubits at beginning and end of path
        for node in path:
            name = node.name
            if name not in op_dict:
                op_dict[name] = 1
            else:
                op_dict[name] += 1
        return op_dict

    def properties(self):
        """Return a dictionary of circuit properties."""
        summary = {"size": self.size(),
                   "depth": self.depth(),
                   "width": self.width(),
                   "qubits": self.num_qubits(),
                   "bits": self.num_clbits(),
                   "factors": self.num_tensor_factors(),
                   "operations": self.count_ops()}
        return summary

    def draw(self, scale=0.7, filename=None, style='color'):
        """
        Draws the dag circuit.

        This function needs `pydot <https://github.com/erocarrera/pydot>`_, which in turn needs
        `Graphviz <https://www.graphviz.org/>`_ to be installed.

        Args:
            scale (float): scaling factor
            filename (str): file path to save image to (format inferred from name)
            style (str):
                'plain': B&W graph;
                'color' (default): color input/output/op nodes

        Returns:
            Ipython.display.Image: if in Jupyter notebook and not saving to file,
            otherwise None.
        """
        from qiskit.visualization.dag_visualization import dag_drawer
        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)
