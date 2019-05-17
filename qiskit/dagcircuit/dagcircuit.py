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
Object to represent a quantum circuit as a directed acyclic graph.

The nodes in the graph are either input/output nodes or operation nodes.
The edges correspond to qubits or bits in the circuit. A directed edge
from node A to node B means that the (qu)bit passes from the output of A
to the input of B. The object's methods allow circuits to be constructed,
composed, and modified. Some natural properties like depth can be computed
directly from the graph.
"""
import re
from collections import OrderedDict
import copy
import itertools
import warnings
import networkx as nx

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.gate import Gate
from .exceptions import DAGCircuitError
from .dagnode import DAGNode


class DAGCircuit:
    """
    Quantum circuit as a directed acyclic graph.

    There are 3 types of nodes in the graph: inputs, outputs, and operations.
    The nodes are connected by directed edges that correspond to qubits and
    bits.
    """

    # pylint: disable=invalid-name

    def __init__(self):
        """Create an empty circuit."""

        # Circuit name.  Generally, this corresponds to the name
        # of the QuantumCircuit from which the DAG was generated.
        self.name = None

        # Set of wires (Register,idx) in the dag
        self.wires = []

        # Map from wire (Register,idx) to input nodes of the graph
        self.input_map = OrderedDict()

        # Map from wire (Register,idx) to output nodes of the graph
        self.output_map = OrderedDict()

        # Stores the max id of a node added to the DAG
        self._max_node_id = 0

        # Directed multigraph whose nodes are inputs, outputs, or operations.
        # Operation nodes have equal in- and out-degrees and carry
        # additional data about the operation, including the argument order
        # and parameter values.
        # Input nodes have out-degree 1 and output nodes have in-degree 1.
        # Edges carry wire labels (reg,idx) and each operation has
        # corresponding in- and out-edges with the same wire labels.
        self._multi_graph = nx.MultiDiGraph()

        # Map of qreg name to QuantumRegister object
        self.qregs = OrderedDict()

        # Map of creg name to ClassicalRegister object
        self.cregs = OrderedDict()

        # TO REMOVE WHEN NODE IS HAVE BEEN REMOVED FULLY
        self._id_to_node = {}

    @property
    def multi_graph(self):
        """Deprecated. Returns internal multi_graph."""
        warnings.warn('DAGCircuit.multi_graph access has been deprecated ' +
                      'in favor of access through the DAGCircuit API.', DeprecationWarning)
        return self._multi_graph

    @multi_graph.setter
    def multi_graph(self, multi_graph):
        """Deprecated. Sets internal multi_graph."""
        warnings.warn('DAGCircuit.multi_graph access has been deprecated ' +
                      'in favor of access through the DAGCircuit API. ', DeprecationWarning)
        self._multi_graph = multi_graph

    def to_networkx(self):
        """Returns a copy of the DAGCircuit in networkx format."""
        return copy.deepcopy(self._multi_graph)

    def get_qubits(self):
        """Deprecated. Use qubits()."""
        warnings.warn('The method get_qubits() is being replaced by qubits()',
                      DeprecationWarning, 2)
        return self.qubits()

    def qubits(self):
        """Return a list of qubits as (QuantumRegister, index) pairs."""
        return [(v, i) for k, v in self.qregs.items() for i in range(v.size)]

    def get_bits(self):
        """Deprecated. Use clbits()."""
        warnings.warn('The method get_bits() is being replaced by clbits()',
                      DeprecationWarning, 2)
        return self.clbits()

    def clbits(self):
        """Return a list of bits as (ClassicalRegister, index) pairs."""
        return [(v, i) for k, v in self.cregs.items() for i in range(v.size)]

    @property
    def node_counter(self):
        """Deprecated usage to return max node id, now returns size of DAG"""
        warnings.warn('Usage of node_counter to return the maximum node id is deprecated,'
                      ' it now returns the number of nodes in the current DAG',
                      DeprecationWarning, 2)
        return len(self._multi_graph)

    # TODO: unused function. is it needed?
    def rename_register(self, regname, newname):
        """Rename a classical or quantum register throughout the circuit.

        regname = existing register name string
        newname = replacement register name string
        """
        if regname == newname:
            return
        if newname in self.qregs or newname in self.cregs:
            raise DAGCircuitError("duplicate register name %s" % newname)
        if regname not in self.qregs and regname not in self.cregs:
            raise DAGCircuitError("no register named %s" % regname)
        if regname in self.qregs:
            reg = self.qregs[regname]
            reg.name = newname
            self.qregs[newname] = reg
            self.qregs.pop(regname, None)
        if regname in self.cregs:
            reg = self.cregs[regname]
            reg.name = newname
            self.qregs[newname] = reg
            self.qregs.pop(regname, None)

        for node in self._multi_graph.nodes():
            if node.type == "in" or node.type == "out":
                if node.name and regname in node.name:
                    node.name = newname
            elif node.type == "op":
                qa = []
                for a in node.qargs:
                    if a[0] == regname:
                        a = (newname, a[1])
                    qa.append(a)
                node.qargs = qa
                ca = []
                for a in node.cargs:
                    if a[0] == regname:
                        a = (newname, a[1])
                    ca.append(a)
                node.cargs = ca
                if node.condition is not None:
                    if node.condition[0] == regname:
                        node.condition = (newname, node.condition[1])
        # eX = edge, d= data
        for _, _, edge_data in self._multi_graph.edges(data=True):
            if regname in edge_data['name']:
                edge_data['name'] = re.sub(regname, newname, edge_data['name'])

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
            self._add_wire((qreg, j))

    def add_creg(self, creg):
        """Add all wires in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGCircuitError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGCircuitError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        for j in range(creg.size):
            self._add_wire((creg, j))

    def _add_wire(self, wire):
        """Add a qubit or bit to the circuit.

        Args:
            wire (tuple): (Register,int) containing a register instance and index
            This adds a pair of in and out nodes connected by an edge.

        Raises:
            DAGCircuitError: if trying to add duplicate wire
        """
        if wire not in self.wires:
            self.wires.append(wire)
            self._max_node_id += 1
            input_map_wire = self.input_map[wire] = self._max_node_id

            self._max_node_id += 1
            output_map_wire = self._max_node_id

            wire_name = "%s[%s]" % (wire[0].name, wire[1])

            inp_node = DAGNode(data_dict={'type': 'in', 'name': wire_name, 'wire': wire},
                               nid=input_map_wire)
            outp_node = DAGNode(data_dict={'type': 'out', 'name': wire_name, 'wire': wire},
                                nid=output_map_wire)
            self._id_to_node[input_map_wire] = inp_node
            self._id_to_node[output_map_wire] = outp_node

            self.input_map[wire] = inp_node
            self.output_map[wire] = outp_node

            self._multi_graph.add_node(inp_node)
            self._multi_graph.add_node(outp_node)

            self._multi_graph.add_edge(inp_node,
                                       outp_node)

            self._multi_graph.adj[inp_node][outp_node][0]["name"] \
                = "%s[%s]" % (wire[0].name, wire[1])
            self._multi_graph.adj[inp_node][outp_node][0]["wire"] \
                = wire
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
            args (list): (register,idx) tuples
            amap (dict): a dictionary keyed on (register,idx) tuples

        Raises:
            DAGCircuitError: if a qubit is not contained in amap
        """
        # Check for each wire
        for wire in args:
            if wire not in amap:
                raise DAGCircuitError("(qu)bit %s[%d] not found" %
                                      (wire[0].name, wire[1]))

    def _bits_in_condition(self, cond):
        """Return a list of bits in the given condition.

        Args:
            cond (tuple or None): optional condition (ClassicalRegister, int)

        Returns:
            list[(ClassicalRegister, idx)]: list of bits
        """
        all_bits = []
        if cond is not None:
            all_bits.extend([(cond[0], j) for j in range(self.cregs[cond[0].name].size)])
        return all_bits

    def _add_op_node(self, op, qargs, cargs, condition=None):
        """Add a new operation node to the graph and assign properties.

        Args:
            op (Instruction): the operation associated with the DAG node
            qargs (list): list of quantum wires to attach to.
            cargs (list): list of classical wires to attach to.
            condition (tuple or None): optional condition (ClassicalRegister, int)
        """
        node_properties = {
            "type": "op",
            "op": op,
            "name": op.name,
            "qargs": qargs,
            "cargs": cargs,
            "condition": condition
        }

        # Add a new operation node to the graph
        self._max_node_id += 1
        new_node = DAGNode(data_dict=node_properties, nid=self._max_node_id)
        self._multi_graph.add_node(new_node)
        self._id_to_node[self._max_node_id] = new_node

    def apply_operation_back(self, op, qargs=None, cargs=None, condition=None):
        """Apply an operation to the output of the circuit.

        Args:
            op (Instruction): the operation associated with the DAG node
            qargs (list[tuple]): qubits that op will be applied to
            cargs (list[tuple]): cbits that op will be applied to
            condition (tuple or None): optional condition (ClassicalRegister, int)

        Returns:
            DAGNode: the current max node

        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs

        """
        qargs = qargs or []
        cargs = cargs or []

        all_cbits = self._bits_in_condition(condition)
        all_cbits.extend(cargs)

        self._check_condition(op.name, condition)
        self._check_bits(qargs, self.output_map)
        self._check_bits(all_cbits, self.output_map)

        self._add_op_node(op, qargs, cargs, condition)

        # Add new in-edges from predecessors of the output nodes to the
        # operation node while deleting the old in-edges of the output nodes
        # and adding new edges from the operation node to each output node
        al = [qargs, all_cbits]
        for q in itertools.chain(*al):
            ie = list(self._multi_graph.predecessors(self.output_map[q]))

            if len(ie) != 1:
                raise DAGCircuitError("output node has multiple in-edges")

            self._multi_graph.add_edge(ie[0], self._id_to_node[self._max_node_id],
                                       name="%s[%s]" % (q[0].name, q[1]), wire=q)
            self._multi_graph.remove_edge(ie[0], self.output_map[q])
            self._multi_graph.add_edge(self._id_to_node[self._max_node_id], self.output_map[q],
                                       name="%s[%s]" % (q[0].name, q[1]), wire=q)

        return self._id_to_node[self._max_node_id]

    def apply_operation_front(self, op, qargs, cargs, condition=None):
        """Apply an operation to the input of the circuit.

        Args:
            op (Instruction): the operation associated with the DAG node
            qargs (list[tuple]): qubits that op will be applied to
            cargs (list[tuple]): cbits that op will be applied to
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
        self._add_op_node(op, qargs, cargs, condition)
        # Add new out-edges to successors of the input nodes from the
        # operation node while deleting the old out-edges of the input nodes
        # and adding new edges to the operation node from each input node
        al = [qargs, all_cbits]
        for q in itertools.chain(*al):
            ie = list(self._multi_graph.successors(self.input_map[q]))
            if len(ie) != 1:
                raise DAGCircuitError("input node has multiple out-edges")
            self._multi_graph.add_edge(self._id_to_node[self._max_node_id], ie[0],
                                       name="%s[%s]" % (q[0].name, q[1]), wire=q)
            self._multi_graph.remove_edge(self.input_map[q], ie[0])
            self._multi_graph.add_edge(self.input_map[q], self._id_to_node[self._max_node_id],
                                       name="%s[%s]" % (q[0].name, q[1]), wire=q)

        return self._id_to_node[self._max_node_id]

    def _check_edgemap_registers(self, edge_map, keyregs, valregs, valreg=True):
        """Check that wiremap neither fragments nor leaves duplicate registers.

        1. There are no fragmented registers. A register in keyregs
        is fragmented if not all of its (qu)bits are renamed by edge_map.
        2. There are no duplicate registers. A register is duplicate if
        it appears in both self and keyregs but not in edge_map.

        Args:
            edge_map (dict): map from (reg,idx) in keyregs to (reg,idx) in valregs
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
            if k[0].name in keyregs:
                reg_frag_chk[k[0]][k[1]] = True
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
                    if not edge_map[(k, 0)][0].name in valregs:
                        size = max(map(lambda x: x[1],
                                       filter(lambda x: x[0] == edge_map[(k, 0)][0],
                                              edge_map.values())))
                        qreg = QuantumRegister(size + 1, edge_map[(k, 0)][0].name)
                        add_regs.add(qreg)
        return add_regs

    def _check_wiremap_validity(self, wire_map, keymap, valmap):
        """Check that the wiremap is consistent.

        Check that the wiremap refers to valid wires and that
        those wires have consistent types.

        Args:
            wire_map (dict): map from (register,idx) in keymap to
                (register,idx) in valmap
            keymap (dict): a map whose keys are wire_map keys
            valmap (dict): a map whose keys are wire_map values

        Raises:
            DAGCircuitError: if wire_map not valid
        """
        for k, v in wire_map.items():
            kname = "%s[%d]" % (k[0].name, k[1])
            vname = "%s[%d]" % (v[0].name, v[1])
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
            condition (tuple): (ClassicalRegister,int)
        Returns:
            tuple(ClassicalRegister,int): new condition
        """
        if condition is None:
            new_condition = None
        else:
            # Map the register name, using fact that registers must not be
            # fragmented by the wire_map (this must have been checked
            # elsewhere)
            bit0 = (condition[0], 0)
            new_condition = (wire_map.get(bit0, bit0)[0], condition[1])
        return new_condition

    def extend_back(self, dag, edge_map=None):
        """Add `dag` at the end of `self`, using `edge_map`.
        """
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
        """Apply the input circuit to the output of this circuit.

        The two bases must be "compatible" or an exception occurs.
        A subset of input qubits of the input circuit are mapped
        to a subset of output qubits of this circuit.

        Args:
            input_circuit (DAGCircuit): circuit to append
            edge_map (dict): map {(Register, int): (Register, int)}
                from the output wires of input_circuit to input wires
                of self.

        Raises:
            DAGCircuitError: if missing, duplicate or inconsistent wire
        """
        edge_map = edge_map or {}

        # Check the wire map for duplicate values
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError("duplicates in wire_map")

        add_qregs = self._check_edgemap_registers(edge_map,
                                                  input_circuit.qregs,
                                                  self.qregs)
        for qreg in add_qregs:
            self.add_qreg(qreg)

        add_cregs = self._check_edgemap_registers(edge_map,
                                                  input_circuit.cregs,
                                                  self.cregs)
        for creg in add_cregs:
            self.add_creg(creg)

        self._check_wiremap_validity(edge_map, input_circuit.input_map,
                                     self.output_map)

        # Compose
        for nd in input_circuit.topological_nodes():
            if nd.type == "in":
                # if in wire_map, get new name, else use existing name
                m_wire = edge_map.get(nd.wire, nd.wire)
                # the mapped wire should already exist
                if m_wire not in self.output_map:
                    raise DAGCircuitError("wire %s[%d] not in self" % (m_wire[0].name, m_wire[1]))

                if nd.wire not in input_circuit.wires:
                    raise DAGCircuitError("inconsistent wire type for %s[%d] in input_circuit"
                                          % (nd.wire[0].name, nd.wire[1]))

            elif nd.type == "out":
                # ignore output nodes
                pass
            elif nd.type == "op":
                condition = self._map_condition(edge_map, nd.condition)
                self._check_condition(nd.name, condition)
                m_qargs = list(map(lambda x: edge_map.get(x, x), nd.qargs))
                m_cargs = list(map(lambda x: edge_map.get(x, x), nd.cargs))
                self.apply_operation_back(nd.op, m_qargs, m_cargs, condition)
            else:
                raise DAGCircuitError("bad node type %s" % nd.type)

    # FIXME: this does not work as expected. it is also not used anywhere
    def compose_front(self, input_circuit, edge_map=None):
        """Apply the input circuit to the input of this circuit.

        The two bases must be "compatible" or an exception occurs.
        A subset of output qubits of the input circuit are mapped
        to a subset of input qubits of this circuit.

        Args:
            input_circuit (DAGCircuit): circuit to append
            edge_map (dict): map {(Register, int): (Register, int)}
                from the output wires of input_circuit to input wires
                of self.

        Raises:
            DAGCircuitError: missing, duplicate or inconsistent wire
        """
        edge_map = edge_map or {}

        # Check the wire map
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError("duplicates in edge_map")

        add_qregs = self._check_edgemap_registers(edge_map,
                                                  input_circuit.qregs,
                                                  self.qregs)
        for qreg in add_qregs:
            self.add_qreg(qreg)

        add_cregs = self._check_edgemap_registers(edge_map,
                                                  input_circuit.cregs,
                                                  self.cregs)
        for creg in add_cregs:
            self.add_creg(creg)

        self._check_wiremap_validity(edge_map, input_circuit.output_map,
                                     self.input_map)

        # Compose
        for nd in reversed(list(input_circuit.topological_nodes())):
            if nd.type == "out":
                # if in edge_map, get new name, else use existing name
                m_name = edge_map.get(nd.wire, nd.wire)
                # the mapped wire should already exist
                if m_name not in self.input_map:
                    raise DAGCircuitError("wire %s[%d] not in self" % (m_name[0].name, m_name[1]))

                if nd.wire not in input_circuit.wires:
                    raise DAGCircuitError(
                        "inconsistent wire for %s[%d] in input_circuit"
                        % (nd.wire[0].name, nd.wire[1]))

            elif nd.type == "in":
                # ignore input nodes
                pass
            elif nd.type == "op":
                condition = self._map_condition(edge_map, nd["condition"])
                self._check_condition(nd.name, condition)
                m_qargs = list(map(lambda x: edge_map.get(x, x), nd.qargs))
                m_cargs = list(map(lambda x: edge_map.get(x, x), nd.cargs))
                self.apply_operation_front(nd.op, m_qargs, m_cargs, condition)
            else:
                raise DAGCircuitError("bad node type %s" % nd.type)

    def size(self):
        """Return the number of operations."""
        return self._multi_graph.order() - 2 * len(self.wires)

    def depth(self):
        """Return the circuit depth.
        Returns:
            int: the circuit depth
        Raises:
            DAGCircuitError: if not a directed acyclic graph
        """
        if not nx.is_directed_acyclic_graph(self._multi_graph):
            raise DAGCircuitError("not a DAG")

        depth = nx.dag_longest_path_length(self._multi_graph) - 1
        return depth if depth != -1 else 0

    def width(self):
        """Return the total number of qubits used by the circuit."""
        return len(self.wires) - self.num_cbits()

    def num_cbits(self):
        """Return the total number of bits used by the circuit."""
        return sum(creg.size for creg in self.cregs.values())

    def num_tensor_factors(self):
        """Compute how many components the circuit can decompose into."""
        return nx.number_weakly_connected_components(self._multi_graph)

    def qasm(self):
        """Deprecated. use qiskit.converters.dag_to_circuit() then call
        qasm() on the obtained QuantumCircuit instance.
        """
        warnings.warn('printing qasm() from DAGCircuit is deprecated. '
                      'use qiskit.converters.dag_to_circuit() then call '
                      'qasm() on the obtained QuantumCircuit instance.',
                      DeprecationWarning, 2)

    def _check_wires_list(self, wires, node):
        """Check that a list of wires is compatible with a node to be replaced.

        - no duplicate names
        - correct length for operation
        Raise an exception otherwise.

        Args:
            wires (list[register, index]): gives an order for (qu)bits
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

        pred_map = {e[2]['wire']: e[0] for e in
                    self._multi_graph.in_edges(nbunch=node, data=True)}
        succ_map = {e[2]['wire']: e[1] for e in
                    self._multi_graph.out_edges(nbunch=node, data=True)}
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
                full_pred_map[w] = self._multi_graph.predecessors(
                    self.output_map[w])[0]
                if len(list(self._multi_graph.predecessors(self.output_map[w]))) != 1:
                    raise DAGCircuitError("too many predecessors for %s[%d] "
                                          "output node" % (w[0], w[1]))

        return full_pred_map, full_succ_map

    def __eq__(self, other):
        # TODO this works but is a horrible way to do this
        slf = copy.deepcopy(self._multi_graph)
        oth = copy.deepcopy(other._multi_graph)

        for node in slf.nodes:
            slf.nodes[node]['node'] = node
        for node in oth.nodes:
            oth.nodes[node]['node'] = node

        return nx.is_isomorphic(slf, oth,
                                node_match=lambda x, y: DAGNode.semantic_eq(x['node'], y['node']))

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order
        """
        return nx.lexicographical_topological_sort(self._multi_graph,
                                                   key=lambda x: str(x.qargs))

    def topological_op_nodes(self):
        """
        Yield op nodes in topological order.

        Returns:
            generator(DAGnode): op node in topological order
        """
        return (nd for nd in self.topological_nodes() if nd.type == 'op')

    def substitute_node_with_dag(self, node, input_dag, wires=None):
        """Replace one node with dag.

        Args:
            node (DAGNode): node to substitute
            input_dag (DAGCircuit): circuit that will substitute the node
            wires (list[(Register, index)]): gives an order for (qu)bits
                in the input circuit. This order gets matched to the node wires
                by qargs first, then cargs, then conditions.

        Raises:
            DAGCircuitError: if met with unexpected predecessor/successors
        """
        if isinstance(node, int):
            warnings.warn('Calling substitute_node_with_dag() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)

            node = self._id_to_node[node]

        condition = node.condition
        # the dag must be amended if used in a
        # conditional context. delete the op nodes and replay
        # them with the condition.
        if condition:
            input_dag.add_creg(condition[0])
            to_replay = []
            for sorted_node in input_dag.topological_nodes():
                if sorted_node.type == "op":
                    sorted_node.op.control = condition
                    to_replay.append(sorted_node)
            for input_node in input_dag.op_nodes():
                input_dag.remove_op_node(input_node)
            for replay_node in to_replay:
                input_dag.apply_operation_back(replay_node.op, replay_node.qargs,
                                               replay_node.cargs, condition=condition)

        if wires is None:
            qwires = [w for w in input_dag.wires if isinstance(w[0], QuantumRegister)]
            cwires = [w for w in input_dag.wires if isinstance(w[0], ClassicalRegister)]
            wires = qwires + cwires

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
        # to depend on those control bits as well.
        if node.type != "op":
            raise DAGCircuitError("expected node type \"op\", got %s"
                                  % node.type)

        condition_bit_list = self._bits_in_condition(node.condition)

        wire_map = {k: v for k, v in zip(wires,
                                         [i for s in [node.qargs,
                                                      node.cargs,
                                                      condition_bit_list]
                                          for i in s])}
        self._check_wiremap_validity(wire_map, wires, self.input_map)
        pred_map, succ_map = self._make_pred_succ_maps(node)
        full_pred_map, full_succ_map = self._full_pred_succ_maps(pred_map, succ_map,
                                                                 input_dag, wire_map)
        # Now that we know the connections, delete node
        self._multi_graph.remove_node(node)

        # Iterate over nodes of input_circuit
        for sorted_node in input_dag.topological_op_nodes():
            # Insert a new node
            condition = self._map_condition(wire_map, sorted_node.condition)
            m_qargs = list(map(lambda x: wire_map.get(x, x),
                               sorted_node.qargs))
            m_cargs = list(map(lambda x: wire_map.get(x, x),
                               sorted_node.cargs))
            self._add_op_node(sorted_node.op, m_qargs, m_cargs, condition)
            # Add edges from predecessor nodes to new node
            # and update predecessor nodes that change
            all_cbits = self._bits_in_condition(condition)
            all_cbits.extend(m_cargs)
            al = [m_qargs, all_cbits]
            for q in itertools.chain(*al):
                self._multi_graph.add_edge(full_pred_map[q],
                                           self._id_to_node[self._max_node_id],
                                           name="%s[%s]" % (q[0].name, q[1]),
                                           wire=q)
                full_pred_map[q] = self._id_to_node[self._max_node_id]

        # Connect all predecessors and successors, and remove
        # residual edges between input and output nodes
        for w in full_pred_map:
            self._multi_graph.add_edge(full_pred_map[w],
                                       full_succ_map[w],
                                       name="%s[%s]" % (w[0].name, w[1]),
                                       wire=w)
            o_pred = list(self._multi_graph.predecessors(self.output_map[w]))
            if len(o_pred) > 1:
                if len(o_pred) != 2:
                    raise DAGCircuitError("expected 2 predecessors here")

                p = [x for x in o_pred if x != full_pred_map[w]]
                if len(p) != 1:
                    raise DAGCircuitError("expected 1 predecessor to pass filter")

                self._multi_graph.remove_edge(p[0], self.output_map[w])

    def node(self, node_id):
        """Get the node in the dag.

        Args:
            node_id(int): Node identifier.

        Returns:
            node: the node.
        """
        return self._multi_graph.nodes[node_id]

    def nodes(self):
        """Iterator for node values.

        Yield:
            node: the node.
        """
        for node in self._multi_graph.nodes:
            yield node

    def edges(self, nodes=None):
        """Iterator for node values.

        Yield:
            node: the node.
        """
        for source_node, dest_node, edge_data in self._multi_graph.edges(nodes, data=True):
            yield source_node, dest_node, edge_data

    def get_op_nodes(self, op=None, data=False):

        """Deprecated. Use op_nodes()."""
        warnings.warn('The method get_op_nodes() is being replaced by op_nodes().'
                      'Returning a list of node_ids/(node_id, data) tuples is '
                      'also deprecated, op_nodes() returns a list of DAGNodes ',
                      DeprecationWarning, 2)
        if data:
            warnings.warn('The parameter data is deprecated, op_nodes() returns DAGNodes'
                          ' which always contain the data',
                          DeprecationWarning, 2)
        nodes = []
        for node in self._multi_graph.nodes():
            if node.type == "op":
                if op is None or isinstance(node.op, op):
                    nodes.append((node._node_id, node.data_dict))
        if not data:
            nodes = [n[0] for n in nodes]
        return nodes

    def op_nodes(self, op=None):
        """Get the list of "op" nodes in the dag.

        Args:
            op (Type): Instruction subclass op nodes to return. if op=None, return
                all op nodes.
        Returns:
            list[DAGNode]: the list of node ids containing the given op.
        """
        nodes = []
        for node in self._multi_graph.nodes():
            if node.type == "op":
                if op is None or isinstance(node.op, op):
                    nodes.append(node)
        return nodes

    def get_gate_nodes(self, data=False):
        """Deprecated. Use gate_nodes()."""
        warnings.warn('The method get_gate_nodes() is being replaced by gate_nodes().'
                      'Returning a list of node_ids/(node_id, data) tuples is also '
                      'deprecated, gate_nodes() returns a list of DAGNodes ',
                      DeprecationWarning, 2)
        if data:
            warnings.warn('The parameter data is deprecated, '
                          'get_gate_nodes() now returns DAGNodes '
                          'which always contain the data',
                          DeprecationWarning, 2)

        nodes = []
        for node in self.op_nodes():
            if isinstance(node.op, Gate):
                nodes.append((node._node_id, node))
        if not data:
            nodes = [n[0] for n in nodes]
        return nodes

    def gate_nodes(self):
        """Get the list of gate nodes in the dag.

        Returns:
            list: the list of node ids that represent gates.
        """
        nodes = []
        for node in self.op_nodes():
            if isinstance(node.op, Gate):
                nodes.append(node)
        return nodes

    def get_named_nodes(self, *names):
        """Deprecated. Use named_nodes()."""
        warnings.warn('The method get_named_nodes() is being replaced by named_nodes()',
                      'Returning a list of node_ids is also deprecated, named_nodes() '
                      'returns a list of DAGNodes ',
                      DeprecationWarning, 2)

        named_nodes = []
        for node in self._multi_graph.nodes():
            if node.type == 'op' and node.op.name in names:
                named_nodes.append(node._node_id)
        return named_nodes

    def named_nodes(self, *names):
        """Get the set of "op" nodes with the given name."""
        named_nodes = []
        for node in self._multi_graph.nodes():
            if node.type == 'op' and node.op.name in names:
                named_nodes.append(node)
        return named_nodes

    def get_2q_nodes(self):
        """Deprecated. Use twoQ_gates()."""
        warnings.warn('The method get_2q_nodes() is being replaced by twoQ_gates()',
                      'Returning a list of data_dicts is also deprecated, twoQ_gates() '
                      'returns a list of DAGNodes.',
                      DeprecationWarning, 2)

        two_q_nodes = []
        for node in self._multi_graph.nodes():
            if node.type == 'op' and len(node.qargs) == 2:
                two_q_nodes.append(node.data_dict)

        return two_q_nodes

    def twoQ_gates(self):
        """Get list of 2-qubit gates. Ignore snapshot, barriers, and the like."""
        two_q_gates = []
        for node in self.gate_nodes():
            if len(node.qargs) == 2:
                two_q_gates.append(node)
        return two_q_gates

    def get_3q_or_more_nodes(self):
        """Deprecated. Use threeQ_or_more_gates()."""
        warnings.warn('The method get_3q_or_more_nodes() is being replaced by'
                      ' threeQ_or_more_gates()',
                      'Returning a list of (node_id, data) tuples is also deprecated, '
                      'threeQ_or_more_gates() returns a list of DAGNodes.',
                      DeprecationWarning, 2)

        three_q_nodes = []
        for node in self._multi_graph.nodes():
            if node.type == 'op' and len(node.qargs) >= 3:
                three_q_nodes.append((node._node_id, node.data_dict))
        return three_q_nodes

    def threeQ_or_more_gates(self):
        """Get list of 3-or-more-qubit gates: (id, data)."""
        three_q_gates = []
        for node in self.gate_nodes():
            if len(node.qargs) >= 3:
                three_q_gates.append(node)
        return three_q_gates

    def successors(self, node):
        """Returns list of the successors of a node as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling successors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return self._multi_graph.successors(node)

    def predecessors(self, node):
        """Returns list of the predecessors of a node as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling predecessors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return self._multi_graph.predecessors(node)

    def quantum_predecessors(self, node):
        """Returns list of the predecessors of a node that are
        connected by a quantum edge as DAGNodes."""

        predecessors = []
        for predecessor in self.predecessors(node):
            if isinstance(self._multi_graph.get_edge_data(predecessor, node, key=0)['wire'][0],
                          QuantumRegister):
                predecessors.append(predecessor)
        return predecessors

    def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling ancestors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return nx.ancestors(self._multi_graph, node)

    def descendants(self, node):
        """Returns set of the descendants of a node as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling descendants() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return nx.descendants(self._multi_graph, node)

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
        and [DAGNode] is its successors in  BFS order.
        """
        if isinstance(node, int):
            warnings.warn('Calling bfs_successors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return nx.bfs_successors(self._multi_graph, node)

    def quantum_successors(self, node):
        """Returns list of the successors of a node that are
        connected by a quantum edge as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling quantum_successors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        successors = []
        for successor in self.successors(node):
            if isinstance(self._multi_graph.get_edge_data(
                    node, successor, key=0)['wire'][0],
                          QuantumRegister):
                successors.append(successor)
        return successors

    def remove_op_node(self, node):
        """Remove an operation node n.

        Add edges from predecessors to successors.
        """
        if isinstance(node, int):
            warnings.warn('Calling remove_op_node() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        if node.type != 'op':
            raise DAGCircuitError('The method remove_op_node only works on op node types. An "%s" '
                                  'node type was wrongly provided.' % node.type)

        pred_map, succ_map = self._make_pred_succ_maps(node)

        # remove from graph and map
        self._multi_graph.remove_node(node)

        for w in pred_map.keys():
            self._multi_graph.add_edge(pred_map[w], succ_map[w],
                                       name="%s[%s]" % (w[0].name, w[1]), wire=w)

    def remove_ancestors_of(self, node):
        """Remove all of the ancestor operation nodes of node."""
        if isinstance(node, int):
            warnings.warn('Calling remove_ancestors_of() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        anc = nx.ancestors(self._multi_graph, node)
        # TODO: probably better to do all at once using
        # multi_graph.remove_nodes_from; same for related functions ...
        for anc_node in anc:
            if anc_node.type == "op":
                self.remove_op_node(anc_node)

    def remove_descendants_of(self, node):
        """Remove all of the descendant operation nodes of node."""
        if isinstance(node, int):
            warnings.warn('Calling remove_descendants_of() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        desc = nx.descendants(self._multi_graph, node)
        for desc_node in desc:
            if desc_node.type == "op":
                self.remove_op_node(desc_node)

    def remove_nonancestors_of(self, node):
        """Remove all of the non-ancestors operation nodes of node."""
        if isinstance(node, int):
            warnings.warn('Calling remove_nonancestors_of() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        anc = nx.ancestors(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(anc))
        for n in comp:
            if n.type == "op":
                self.remove_op_node(n)

    def remove_nondescendants_of(self, node):
        """Remove all of the non-descendants operation nodes of node."""
        if isinstance(node, int):
            warnings.warn('Calling remove_nondescendants_of() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        dec = nx.descendants(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(dec))
        for n in comp:
            if n.type == "op":
                self.remove_op_node(n)

    def layers(self):
        """Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.

        A layer is a circuit whose gates act on disjoint qubits, i.e.
        a layer has depth 1. The total number of layers equals the
        circuit depth d. The layers are indexed from 0 to d-1 with the
        earliest layer at index 0. The layers are constructed using a
        greedy algorithm. Each returned layer is a dict containing
        {"graph": circuit graph, "partition": list of qubit lists}.

        TODO: Gates that use the same cbits will end up in different
        layers as this is currently implemented. This may not be
        the desired behavior.
        """
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)  # Remove input nodes
        except StopIteration:
            return

        def add_nodes_from(layer, nodes):
            """ Convert DAGNodes into a format that can be added to a
             multigraph and then add to graph"""
            layer._multi_graph.add_nodes_from(nodes)

        for graph_layer in graph_layers:

            # Get the op nodes from the layer, removing any input and output nodes.
            op_nodes = [node for node in graph_layer if node.type == "op"]

            # Stop yielding once there are no more op_nodes in a layer.
            if not op_nodes:
                return

            # Construct a shallow copy of self
            new_layer = DAGCircuit()
            new_layer.name = self.name

            for creg in self.cregs.values():
                new_layer.add_creg(creg)
            for qreg in self.qregs.values():
                new_layer.add_qreg(qreg)

            add_nodes_from(new_layer, self.input_map.values())
            add_nodes_from(new_layer, self.output_map.values())
            add_nodes_from(new_layer, op_nodes)

            # The quantum registers that have an operation in this layer.
            support_list = [
                op_node.qargs
                for op_node in op_nodes
                if op_node.name not in {"barrier", "snapshot", "save", "load", "noise"}
            ]

            # Now add the edges to the multi_graph
            # By default we just wire inputs to the outputs.
            wires = {self.input_map[wire]: self.output_map[wire]
                     for wire in self.wires}
            # Wire inputs to op nodes, and op nodes to outputs.
            for op_node in op_nodes:
                args = self._bits_in_condition(op_node.condition) \
                       + op_node.cargs + op_node.qargs
                arg_ids = (self.input_map[(arg[0], arg[1])] for arg in args)
                for arg_id in arg_ids:
                    wires[arg_id], wires[op_node] = op_node, wires[arg_id]

            # Add wiring to/from the operations and between unused inputs & outputs.
            new_layer._multi_graph.add_edges_from(wires.items())
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
        predecessor_count = dict()  # Dict[node, predecessors not visited]
        cur_layer = [node for node in self.input_map.values()]
        yield cur_layer
        next_layer = []
        while cur_layer:
            for node in cur_layer:
                # Count multiedges with multiplicity.
                for successor in self._multi_graph.successors(node):
                    multiplicity = self._multi_graph.number_of_edges(node, successor)
                    if successor in predecessor_count:
                        predecessor_count[successor] -= multiplicity
                    else:
                        predecessor_count[successor] = \
                            self._multi_graph.in_degree(successor) - multiplicity

                    if predecessor_count[successor] == 0:
                        next_layer.append(successor)
                        del predecessor_count[successor]

            yield next_layer
            cur_layer = next_layer
            next_layer = []

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
                s = list(self._multi_graph.successors(node))
                while len(s) == 1 and \
                        s[0].type == "op" and \
                        s[0].name in namelist:
                    group.append(s[0])
                    nodes_seen[s[0]] = True
                    s = list(self._multi_graph.successors(s[0]))
                if len(group) >= 1:
                    group_list.append(tuple(group))
        return set(group_list)

    def nodes_on_wire(self, wire, only_ops=False):
        """
        Iterator for nodes that affect a given wire

        Args:
            wire (tuple(Register, index)): the wire to be looked at.
            only_ops (bool): True if only the ops nodes are wanted
                        otherwise all nodes are returned.
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
            for node, edges in self._multi_graph.adj[current_node].items():
                if any(wire == edge['wire'] for edge in edges.values()):
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

    def properties(self):
        """Return a dictionary of circuit properties."""
        summary = {"size": self.size(),
                   "depth": self.depth(),
                   "width": self.width(),
                   "bits": self.num_cbits(),
                   "factors": self.num_tensor_factors(),
                   "operations": self.count_ops()}
        return summary
