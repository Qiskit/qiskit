# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
        self.multi_graph = nx.MultiDiGraph()

        # Map of qreg name to QuantumRegister object
        self.qregs = OrderedDict()

        # Map of creg name to ClassicalRegister object
        self.cregs = OrderedDict()

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
        return len(self.multi_graph)

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
        # n node d = data
        for _, d in self.multi_graph.nodes(data=True):
            if d["type"] == "in" or d["type"] == "out":
                if regname in d["name"]:
                    d["name"] = re.sub(regname, newname, d["name"])
            elif d["type"] == "op":
                qa = []
                for a in d["qargs"]:
                    if a[0] == regname:
                        a = (newname, a[1])
                    qa.append(a)
                d["qargs"] = qa
                ca = []
                for a in d["cargs"]:
                    if a[0] == regname:
                        a = (newname, a[1])
                    ca.append(a)
                d["cargs"] = ca
                if d["condition"] is not None:
                    if d["condition"][0] == regname:
                        d["condition"] = (newname, d["condition"][1])
        # eX = edge, d= data
        for _, _, d in self.multi_graph.edges(data=True):
            if regname in d["name"]:
                d["name"] = re.sub(regname, newname, d["name"])

    def remove_all_ops_named(self, opname):
        """Remove all operation nodes with the given name."""
        for n in self.named_nodes(opname):
            self._remove_op_node(n)

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
            output_map_wire = self.output_map[wire] = self._max_node_id

            self.multi_graph.add_edge(input_map_wire,
                                      output_map_wire)

            wire_name = "%s[%s]" % (wire[0].name, wire[1])

            self.multi_graph.add_nodes_from([(input_map_wire, {'type': 'in'}),
                                             (output_map_wire, {'type': 'out'})
                                             ],
                                            name=wire_name,
                                            wire=wire,
                                            )
            self.multi_graph.adj[input_map_wire][output_map_wire][0]["name"] \
                = "%s[%s]" % (wire[0].name, wire[1])
            self.multi_graph.adj[input_map_wire][output_map_wire][0]["wire"] \
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
                raise DAGCircuitError("(qu)bit %s[%d] not found" % (wire[0].name, wire[1]))

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
        # Add a new operation node to the graph
        self._max_node_id += 1
        self.multi_graph.add_node(self._max_node_id)
        # Update the operation itself. TODO: remove after qargs not connected to op
        op.qargs = qargs
        op.cargs = cargs
        # Update that operation node's data
        self.multi_graph.node[self._max_node_id]["type"] = "op"
        self.multi_graph.node[self._max_node_id]["op"] = op
        self.multi_graph.node[self._max_node_id]["name"] = op.name
        self.multi_graph.node[self._max_node_id]["qargs"] = qargs
        self.multi_graph.node[self._max_node_id]["cargs"] = cargs
        self.multi_graph.node[self._max_node_id]["condition"] = condition

    def apply_operation_back(self, op, qargs=None, cargs=None, condition=None):
        """Apply an operation to the output of the circuit.
        TODO: make `qargs` and `cargs` mandatory, when they are dropped from op.

        Args:
            op (Instruction): the operation associated with the DAG node
            qargs (list[tuple]): qubits that op will be applied to
            cargs (list[tuple]): cbits that op will be applied to
            condition (tuple or None): optional condition (ClassicalRegister, int)

        Returns:
            int: the current max node id

        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs

        """
        qargs = qargs or op.qargs
        cargs = cargs or op.cargs
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
            ie = list(self.multi_graph.predecessors(self.output_map[q]))
            if len(ie) != 1:
                raise DAGCircuitError("output node has multiple in-edges")

            self.multi_graph.add_edge(ie[0], self._max_node_id,
                                      name="%s[%s]" % (q[0].name, q[1]), wire=q)
            self.multi_graph.remove_edge(ie[0], self.output_map[q])
            self.multi_graph.add_edge(self._max_node_id, self.output_map[q],
                                      name="%s[%s]" % (q[0].name, q[1]), wire=q)

        return self._max_node_id

    def apply_operation_front(self, op, qargs=None, cargs=None, condition=None):
        """Apply an operation to the input of the circuit.
        TODO: make `qargs` and `cargs` mandatory, when they are dropped from op.

        Args:
            op (Instruction): the operation associated with the DAG node
            qargs (list[tuple]): qubits that op will be applied to
            cargs (list[tuple]): cbits that op will be applied to
            condition (tuple or None): optional condition (ClassicalRegister, value)

        Returns:
            int: the current max node id

        Raises:
            DAGCircuitError: if initial nodes connected to multiple out edges
        """
        qargs = qargs or op.qargs
        cargs = cargs or op.cargs
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
            ie = list(self.multi_graph.successors(self.input_map[q]))
            if len(ie) != 1:
                raise DAGCircuitError("input node has multiple out-edges")
            self.multi_graph.add_edge(self._max_node_id, ie[0],
                                      name="%s[%s]" % (q[0].name, q[1]), wire=q)
            self.multi_graph.remove_edge(self.input_map[q], ie[0])
            self.multi_graph.add_edge(self.input_map[q], self._max_node_id,
                                      name="%s[%s]" % (q[0].name, q[1]), wire=q)

        return self._max_node_id

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
            elif s == set([False]):
                if k in self.qregs.values() or k in self.cregs.values():
                    raise DAGCircuitError("unmapped duplicate reg %s" % k)
                else:
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
            DAGCircuitError: if missing, duplicate or incosistent wire
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
        for node in nx.topological_sort(input_circuit.multi_graph):
            nd = input_circuit.multi_graph.node[node]
            if nd["type"] == "in":
                # if in wire_map, get new name, else use existing name
                m_wire = edge_map.get(nd["wire"], nd["wire"])
                # the mapped wire should already exist
                if m_wire not in self.output_map:
                    raise DAGCircuitError("wire %s[%d] not in self" % (m_wire[0].name, m_wire[1]))

                if nd["wire"] not in input_circuit.wires:
                    raise DAGCircuitError("inconsistent wire type for %s[%d] in input_circuit"
                                          % (nd["wire"][0].name, nd["wire"][1]))

            elif nd["type"] == "out":
                # ignore output nodes
                pass
            elif nd["type"] == "op":
                condition = self._map_condition(edge_map, nd["condition"])
                self._check_condition(nd["name"], condition)
                m_qargs = list(map(lambda x: edge_map.get(x, x), nd["qargs"]))
                m_cargs = list(map(lambda x: edge_map.get(x, x), nd["cargs"]))
                self.apply_operation_back(nd["op"], m_qargs, m_cargs, condition)
            else:
                raise DAGCircuitError("bad node type %s" % nd["type"])

    # FIXME: this does not work as expected. it is also not used anywhere
    def compose_front(self, input_circuit, wire_map=None):
        """Apply the input circuit to the input of this circuit.

        The two bases must be "compatible" or an exception occurs.
        A subset of output qubits of the input circuit are mapped
        to a subset of input qubits of this circuit.

        Args:
            input_circuit (DAGCircuit): circuit to append
            wire_map (dict): map {(Register, int): (Register, int)}
                from the output wires of input_circuit to input wires
                of self.

        Raises:
            DAGCircuitError: missing, duplicate or inconsistent wire
        """
        wire_map = wire_map or {}

        # Check the wire map
        if len(set(wire_map.values())) != len(wire_map):
            raise DAGCircuitError("duplicates in wire_map")

        add_qregs = self._check_edgemap_registers(wire_map,
                                                  input_circuit.qregs,
                                                  self.qregs)
        for qreg in add_qregs:
            self.add_qreg(qreg)

        add_cregs = self._check_edgemap_registers(wire_map,
                                                  input_circuit.cregs,
                                                  self.cregs)
        for creg in add_cregs:
            self.add_creg(creg)

        self._check_wiremap_validity(wire_map, input_circuit.output_map,
                                     self.input_map)

        # Compose
        for n in reversed(list(nx.topological_sort(input_circuit.multi_graph))):
            nd = input_circuit.multi_graph.node[n]
            if nd["type"] == "out":
                # if in wire_map, get new name, else use existing name
                m_name = wire_map.get(nd["wire"], nd["wire"])
                # the mapped wire should already exist
                if m_name not in self.input_map:
                    raise DAGCircuitError("wire %s[%d] not in self" % (m_name[0].name, m_name[1]))

                if nd["wire"] not in input_circuit.wires:
                    raise DAGCircuitError(
                        "inconsistent wire for %s[%d] in input_circuit"
                        % (nd["wire"][0].name, nd["wire"][1]))

            elif nd["type"] == "in":
                # ignore input nodes
                pass
            elif nd["type"] == "op":
                condition = self._map_condition(wire_map, nd["condition"])
                self._check_condition(nd["name"], condition)
                self.apply_operation_front(nd["op"], condition)
            else:
                raise DAGCircuitError("bad node type %s" % nd["type"])

    def size(self):
        """Return the number of operations."""
        return self.multi_graph.order() - 2 * len(self.wires)

    def depth(self):
        """Return the circuit depth.

        Returns:
            int: the circuit depth

        Raises:
            DAGCircuitError: if not a directed acyclic graph
        """
        if not nx.is_directed_acyclic_graph(self.multi_graph):
            raise DAGCircuitError("not a DAG")

        return nx.dag_longest_path_length(self.multi_graph) - 1

    def width(self):
        """Return the total number of qubits used by the circuit."""
        return len(self.wires) - self.num_cbits()

    def num_cbits(self):
        """Return the total number of bits used by the circuit."""
        return sum(creg.size for creg in self.cregs.values())

    def num_tensor_factors(self):
        """Compute how many components the circuit can decompose into."""
        return nx.number_weakly_connected_components(self.multi_graph)

    def qasm(self):
        """Deprecated. use qiskit.converters.dag_to_circuit() then call
        qasm() on the obtained QuantumCircuit instance.
        """
        warnings.warn('printing qasm() from DAGCircuit is deprecated. '
                      'use qiskit.converters.dag_to_circuit() then call '
                      'qasm() on the obtained QuantumCircuit instance.',
                      DeprecationWarning, 2)

    def _check_wires_list(self, wires, op, input_circuit, condition=None):
        """Check that a list of wires satisfies some conditions.

        - no duplicate names
        - correct length for operation
        - elements are wires of input_circuit
        Raise an exception otherwise.

        Args:
            wires (list[register, index]): gives an order for (qu)bits
                in the input_circuit that is replacing the operation.
            op (Instruction): operation
            input_circuit (DAGCircuit): replacement circuit for operation
            condition (tuple or None): if this instance of the
                operation is classically controlled by a (ClassicalRegister, int)

        Raises:
            DAGCircuitError: if check doesn't pass.
        """
        if len(set(wires)) != len(wires):
            raise DAGCircuitError("duplicate wires")

        wire_tot = len(op.qargs) + len(op.cargs)
        if condition is not None:
            wire_tot += condition[0].size

        if len(wires) != wire_tot:
            raise DAGCircuitError("expected %d wires, got %d"
                                  % (wire_tot, len(wires)))

        for w in wires:
            if w not in input_circuit.wires:
                raise DAGCircuitError("wire (%s,%d) not in input circuit"
                                      % (w[0], w[1]))

    def _make_pred_succ_maps(self, n):
        """Return predecessor and successor dictionaries.

        Args:
            n (int): reference to self.multi_graph node id

        Returns:
            tuple(dict): tuple(predecessor_map, successor_map)
                These map from wire (Register, int) to predecessor (successor)
                nodes of n.
        """
        pred_map = {e[2]['wire']: e[0] for e in
                    self.multi_graph.in_edges(nbunch=n, data=True)}
        succ_map = {e[2]['wire']: e[1] for e in
                    self.multi_graph.out_edges(nbunch=n, data=True)}
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
                full_pred_map[w] = self.multi_graph.predecessors(
                    self.output_map[w])[0]
                if len(list(self.multi_graph.predecessors(self.output_map[w]))) != 1:
                    raise DAGCircuitError("too many predecessors for %s[%d] "
                                          "output node" % (w[0], w[1]))
        return full_pred_map, full_succ_map

    @staticmethod
    def _match_dag_nodes(node1, node2):
        """
        Check if DAG nodes are considered equivalent, e.g. as a node_match for nx.is_isomorphic.
        Args:
            node1 (dict): A node to compare.
            node2 (dict): The other node to compare.

        Returns:
            Bool: If node1 == node2
        """
        copy_node1 = {k: v for (k, v) in node1.items()}
        copy_node2 = {k: v for (k, v) in node2.items()}

        # For barriers, qarg order is not significant so compare as sets
        if 'barrier' == copy_node1['name'] == copy_node2['name']:
            node1_qargs = set(copy_node1.pop('qargs', []))
            node2_qargs = set(copy_node2.pop('qargs', []))

            if node1_qargs != node2_qargs:
                return False

        return copy_node1 == copy_node2

    def __eq__(self, other):
        return nx.is_isomorphic(self.multi_graph, other.multi_graph,
                                node_match=DAGCircuit._match_dag_nodes)

    def node_nums_in_topological_order(self):
        """
        Returns the nodes (their ids) in topological order.

        Returns:
            list: The list of node numbers in topological order
        """
        return nx.lexicographical_topological_sort(self.multi_graph)

    def substitute_circuit_all(self, op, input_circuit, wires=None):
        """Replace every occurrence of operation op with input_circuit.

        Args:
            op (Instruction): operation type to substitute across the dag.
            input_circuit (DAGCircuit): what to replace with
            wires (list[register, index]): gives an order for (qu)bits
                in the input_circuit that is replacing the operation.

        Raises:
            DAGCircuitError: if met with unexpected predecessor/successors
        """
        # TODO: rewrite this method to call substitute_node_with_dag
        wires = wires or []

        self._check_wires_list(wires, op, input_circuit)

        # Create a proxy wire_map to identify fragments and duplicates
        # and determine what registers need to be added to self
        proxy_map = {w: (QuantumRegister(1, 'proxy'), 0) for w in wires}
        add_qregs = self._check_edgemap_registers(proxy_map,
                                                  input_circuit.qregs,
                                                  {}, False)
        for qreg in add_qregs:
            self.add_qreg(qreg)

        add_cregs = self._check_edgemap_registers(proxy_map,
                                                  input_circuit.cregs,
                                                  {}, False)
        for creg in add_cregs:
            self.add_creg(creg)

        # Iterate through the nodes of self and replace the selected nodes
        # by iterating through the input_circuit, constructing and
        # checking the validity of the wire_map for each replacement
        # NOTE: We do not replace conditioned gates. One way to implement
        #       this later is to add or update the conditions of each gate
        #       that we add from the input_circuit.
        for n in self.node_nums_in_topological_order():
            nd = self.multi_graph.node[n]
            if nd["type"] == "op" and nd["op"] == op:
                if nd["condition"] is None:
                    wire_map = {k: v for k, v in zip(wires,
                                                     [i for s in [nd["qargs"], nd["cargs"]]
                                                      for i in s])}
                    self._check_wiremap_validity(wire_map, wires,
                                                 self.input_map)
                    pred_map, succ_map = self._make_pred_succ_maps(n)
                    full_pred_map, full_succ_map = \
                        self._full_pred_succ_maps(pred_map, succ_map,
                                                  input_circuit, wire_map)
                    # Now that we know the connections, delete node
                    self.multi_graph.remove_node(n)
                    # Iterate over nodes of input_circuit
                    for m in nx.topological_sort(input_circuit.multi_graph):
                        md = input_circuit.multi_graph.node[m]
                        if md["type"] == "op":
                            # Insert a new node
                            condition = self._map_condition(wire_map,
                                                            md["condition"])
                            m_qargs = [wire_map.get(x, x) for x in md["qargs0"]]
                            m_cargs = [wire_map.get(x, x) for x in md["cargs0"]]
                            self._add_op_node(md["op"], m_qargs, m_cargs, condition)
                            # Add edges from predecessor nodes to new node
                            # and update predecessor nodes that change
                            all_cbits = self._bits_in_condition(condition)
                            all_cbits.extend(m_cargs)
                            al = [m_qargs, all_cbits]
                            for q in itertools.chain(*al):
                                self.multi_graph.add_edge(full_pred_map[q],
                                                          self._max_node_id,
                                                          name="%s[%s]" % (q[0].name, q[1]),
                                                          wire=q)
                                full_pred_map[q] = copy.copy(self._max_node_id)
                    # Connect all predecessors and successors, and remove
                    # residual edges between input and output nodes
                    for w in full_pred_map:
                        self.multi_graph.add_edge(full_pred_map[w],
                                                  full_succ_map[w],
                                                  name="%s[%s]" % (w[0].name, w[1]),
                                                  wire=w)
                        o_pred = list(self.multi_graph.predecessors(
                            self.output_map[w]))
                        if len(o_pred) > 1:
                            if len(o_pred) != 2:
                                raise DAGCircuitError("expected 2 predecessors here")

                            p = [x for x in o_pred if x != full_pred_map[w]]
                            if len(p) != 1:
                                raise DAGCircuitError("expected 1 predecessor to pass filter")

                            self.multi_graph.remove_edge(
                                p[0], self.output_map[w])

    def substitute_node_with_dag(self, node, input_dag, wires=None):
        """Replace one node with dag.

        Args:
            node (int): node of self.multi_graph (of type "op") to substitute
            input_dag (DAGCircuit): circuit that will substitute the node
            wires (list[(Register, index)]): gives an order for (qu)bits
                in the input circuit. This order gets matched to the node wires
                by qargs first, then cargs, then conditions.

        Raises:
            DAGCircuitError: if met with unexpected predecessor/successors
        """
        nd = self.multi_graph.node[node]

        condition = nd["condition"]
        # the decomposition rule must be amended if used in a
        # conditional context. delete the op nodes and replay
        # them with the condition.
        if condition:
            input_dag.add_creg(condition[0])
            to_replay = []
            for n_it in nx.topological_sort(input_dag.multi_graph):
                n = input_dag.multi_graph.nodes[n_it]
                if n["type"] == "op":
                    n["op"].control = condition
                    to_replay.append(n)
            for n in input_dag.op_nodes():
                input_dag._remove_op_node(n)
            for n in to_replay:
                input_dag.apply_operation_back(n["op"], condition=condition)

        if wires is None:
            qwires = [w for w in input_dag.wires if isinstance(w[0], QuantumRegister)]
            cwires = [w for w in input_dag.wires if isinstance(w[0], ClassicalRegister)]
            wires = qwires + cwires

        self._check_wires_list(wires, nd["op"], input_dag, nd["condition"])

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
        if nd["type"] != "op":
            raise DAGCircuitError("expected node type \"op\", got %s"
                                  % nd["type"])

        condition_bit_list = self._bits_in_condition(nd["condition"])

        wire_map = {k: v for k, v in zip(wires,
                                         [i for s in [nd["qargs"],
                                                      nd["cargs"],
                                                      condition_bit_list]
                                          for i in s])}
        self._check_wiremap_validity(wire_map, wires, self.input_map)
        pred_map, succ_map = self._make_pred_succ_maps(node)
        full_pred_map, full_succ_map = self._full_pred_succ_maps(pred_map, succ_map,
                                                                 input_dag, wire_map)
        # Now that we know the connections, delete node
        self.multi_graph.remove_node(node)
        # Iterate over nodes of input_circuit
        for m in nx.topological_sort(input_dag.multi_graph):
            md = input_dag.multi_graph.node[m]
            if md["type"] == "op":
                # Insert a new node
                condition = self._map_condition(wire_map, md["condition"])
                m_qargs = list(map(lambda x: wire_map.get(x, x),
                                   md["qargs"]))
                m_cargs = list(map(lambda x: wire_map.get(x, x),
                                   md["cargs"]))
                self._add_op_node(md["op"], m_qargs, m_cargs, condition)
                # Add edges from predecessor nodes to new node
                # and update predecessor nodes that change
                all_cbits = self._bits_in_condition(condition)
                all_cbits.extend(m_cargs)
                al = [m_qargs, all_cbits]
                for q in itertools.chain(*al):
                    self.multi_graph.add_edge(full_pred_map[q],
                                              self._max_node_id,
                                              name="%s[%s]" % (q[0].name, q[1]),
                                              wire=q)
                    full_pred_map[q] = copy.copy(self._max_node_id)
        # Connect all predecessors and successors, and remove
        # residual edges between input and output nodes
        for w in full_pred_map:
            self.multi_graph.add_edge(full_pred_map[w],
                                      full_succ_map[w],
                                      name="%s[%s]" % (w[0].name, w[1]),
                                      wire=w)
            o_pred = list(self.multi_graph.predecessors(self.output_map[w]))
            if len(o_pred) > 1:
                if len(o_pred) != 2:
                    raise DAGCircuitError("expected 2 predecessors here")

                p = [x for x in o_pred if x != full_pred_map[w]]
                if len(p) != 1:
                    raise DAGCircuitError("expected 1 predecessor to pass filter")

                self.multi_graph.remove_edge(p[0], self.output_map[w])

    def get_op_nodes(self, op=None, data=False):
        """Deprecated. Use op_nodes()."""
        warnings.warn('The method get_op_nodes() is being replaced by op_nodes()',
                      DeprecationWarning, 2)
        return self.op_nodes(op, data)

    def op_nodes(self, op=None, data=False):
        """Get the list of "op" nodes in the dag.

        Args:
            op (Type): Instruction subclass op nodes to return. if op=None, return
                all op nodes.
            data (bool): Default: False. If True, return a list of tuple
                (node_id, node_data). If False, return a list of int (node_id)

        Returns:
            list: the list of node ids containing the given op.
        """
        nodes = []
        for node_id, node_data in self.multi_graph.nodes(data=True):
            if node_data["type"] == "op":
                if op is None or isinstance(node_data["op"], op):
                    nodes.append((node_id, node_data))
        if not data:
            nodes = [n[0] for n in nodes]
        return nodes

    def get_gate_nodes(self, data=False):
        """Deprecated. Use gate_nodes()."""
        warnings.warn('The method get_gate_nodes() is being replaced by gate_nodes()',
                      DeprecationWarning, 2)
        return self.gate_nodes(data)

    def gate_nodes(self, data=False):
        """Get the list of gate nodes in the dag.

        Args:
            data (bool): Default: False. If True, return a list of tuple
                (node_id, node_data). If False, return a list of int (node_id)

        Returns:
            list: the list of node ids that represent gates.
        """
        nodes = []
        for node_id, node_data in self.op_nodes(data=True):
            if isinstance(node_data['op'], Gate):
                nodes.append((node_id, node_data))
        if not data:
            nodes = [n[0] for n in nodes]
        return nodes

    def get_named_nodes(self, *names):
        """Deprecated. Use named_nodes()."""
        warnings.warn('The method get_named_nodes() is being replaced by named_nodes()',
                      DeprecationWarning, 2)
        return self.named_nodes(*names)

    def named_nodes(self, *names):
        """Get the set of "op" nodes with the given name."""
        named_nodes = []
        for node_id, node_data in self.multi_graph.nodes(data=True):
            if node_data['type'] == 'op' and node_data['op'].name in names:
                named_nodes.append(node_id)
        return named_nodes

    def get_2q_nodes(self):
        """Deprecated. Use twoQ_nodes()."""
        warnings.warn('The method get_2q_nodes() is being replaced by twoQ_nodes()',
                      DeprecationWarning, 2)
        return self.twoQ_nodes()

    def twoQ_nodes(self):
        """Get list of 2-qubit nodes."""
        two_q_nodes = []
        for node_id, node_data in self.multi_graph.nodes(data=True):
            if node_data['type'] == 'op' and len(node_data['qargs']) == 2:
                two_q_nodes.append(self.multi_graph.node[node_id])
        return two_q_nodes

    def get_3q_or_more_nodes(self):
        """Deprecated. Use threeQ_or_more_nodes()."""
        warnings.warn('The method get_3q_or_more_nodes() is being replaced by'
                      ' threeQ_or_more_nodes()', DeprecationWarning, 2)
        return self.threeQ_or_more_nodes()

    def threeQ_or_more_nodes(self):
        """Get list of 3-or-more-qubit nodes: (id, data)."""
        three_q_nodes = []
        for node_id, node_data in self.multi_graph.nodes(data=True):
            if node_data['type'] == 'op' and len(node_data['qargs']) >= 3:
                three_q_nodes.append((node_id, self.multi_graph.node[node_id]))
        return three_q_nodes

    def successors(self, node):
        """Returns the successors of a node."""
        return self.multi_graph.successors(node)

    def ancestors(self, node):
        """Returns the ancestors of a node."""
        return nx.ancestors(self.multi_graph, node)

    def descendants(self, node):
        """Returns the descendants of a node."""
        return nx.descendants(self.multi_graph, node)

    def bfs_successors(self, node):
        """Returns successors of a node in BFS order"""
        return nx.bfs_successors(self.multi_graph, node)

    def quantum_successors(self, node):
        """Returns the successors of a node that are connected by a quantum edge"""
        successors = []
        for successor in self.successors(node):
            if isinstance(self.multi_graph.get_edge_data(node, successor, key=0)['wire'][0],
                          QuantumRegister):
                successors.append(successor)
        return successors

    def _remove_op_node(self, n):
        """Remove an operation node n.

        Add edges from predecessors to successors.
        """
        pred_map, succ_map = self._make_pred_succ_maps(n)
        self.multi_graph.remove_node(n)
        for w in pred_map.keys():
            self.multi_graph.add_edge(pred_map[w], succ_map[w],
                                      name="%s[%s]" % (w[0].name, w[1]), wire=w)

    def remove_ancestors_of(self, node):
        """Remove all of the ancestor operation nodes of node."""
        anc = nx.ancestors(self.multi_graph, node)
        # TODO: probably better to do all at once using
        # multi_graph.remove_nodes_from; same for related functions ...
        for n in anc:
            nd = self.multi_graph.node[n]
            if nd["type"] == "op":
                self._remove_op_node(n)

    def remove_descendants_of(self, node):
        """Remove all of the descendant operation nodes of node."""
        dec = nx.descendants(self.multi_graph, node)
        for n in dec:
            nd = self.multi_graph.node[n]
            if nd["type"] == "op":
                self._remove_op_node(n)

    def remove_nonancestors_of(self, node):
        """Remove all of the non-ancestors operation nodes of node."""
        anc = nx.ancestors(self.multi_graph, node)
        comp = list(set(self.multi_graph.nodes()) - set(anc))
        for n in comp:
            nd = self.multi_graph.node[n]
            if nd["type"] == "op":
                self._remove_op_node(n)

    def remove_nondescendants_of(self, node):
        """Remove all of the non-descendants operation nodes of node."""
        dec = nx.descendants(self.multi_graph, node)
        comp = list(set(self.multi_graph.nodes()) - set(dec))
        for n in comp:
            nd = self.multi_graph.node[n]
            if nd["type"] == "op":
                self._remove_op_node(n)

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

        def nodes_data(nodes):
            """Construct full nodes from just node ids."""
            return ((node_id, self.multi_graph.nodes[node_id]) for node_id in nodes)

        for graph_layer in graph_layers:
            # Get the op nodes from the layer, removing any input and output nodes.
            op_nodes = [node for node in nodes_data(graph_layer) if node[1]["type"] == "op"]

            # Stop yielding once there are no more op_nodes in a layer.
            if not op_nodes:
                return

            # Construct a shallow copy of self
            new_layer = copy.copy(self)
            new_layer.multi_graph = nx.MultiDiGraph()

            new_layer.multi_graph.add_nodes_from(nodes_data(self.input_map.values()))
            new_layer.multi_graph.add_nodes_from(nodes_data(self.output_map.values()))

            # The quantum registers that have an operation in this layer.
            support_list = [
                op_node[1]["qargs"]
                for op_node in op_nodes
                if op_node[1]["op"].name not in {"barrier", "snapshot", "save", "load", "noise"}
            ]
            new_layer.multi_graph.add_nodes_from(op_nodes)

            # Now add the edges to the multi_graph
            # By default we just wire inputs to the outputs.
            wires = {self.input_map[wire]: self.output_map[wire]
                     for wire in self.wires}
            # Wire inputs to op nodes, and op nodes to outputs.
            for op_node in op_nodes:
                args = self._bits_in_condition(op_node[1]["condition"]) \
                       + op_node[1]["cargs"] + op_node[1]["qargs"]
                arg_ids = (self.input_map[(arg[0], arg[1])] for arg in args)
                for arg_id in arg_ids:
                    wires[arg_id], wires[op_node[0]] = op_node[0], wires[arg_id]

            # Add wiring to/from the operations and between unused inputs & outputs.
            new_layer.multi_graph.add_edges_from(wires.items())
            yield {"graph": new_layer, "partition": support_list}

    def serial_layers(self):
        """Yield a layer for all gates of this circuit.

        A serial layer is a circuit with one gate. The layers have the
        same structure as in layers().
        """
        for n in self.node_nums_in_topological_order():
            nxt_nd = self.multi_graph.node[n]
            if nxt_nd["type"] == "op":
                new_layer = DAGCircuit()
                for qreg in self.qregs.values():
                    new_layer.add_qreg(qreg)
                for creg in self.cregs.values():
                    new_layer.add_creg(creg)
                # Save the support of the operation we add to the layer
                support_list = []
                # Operation data
                op = copy.copy(nxt_nd["op"])
                qa = copy.copy(nxt_nd["qargs"])
                ca = copy.copy(nxt_nd["cargs"])
                co = copy.copy(nxt_nd["condition"])
                _ = self._bits_in_condition(co)

                # Add node to new_layer
                new_layer.apply_operation_back(op, qa, ca, co)
                # Add operation to partition
                if nxt_nd["name"] not in ["barrier",
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
                for successor in self.multi_graph.successors(node):
                    multiplicity = self.multi_graph.number_of_edges(node, successor)
                    if successor in predecessor_count:
                        predecessor_count[successor] -= multiplicity
                    else:
                        predecessor_count[successor] = \
                            self.multi_graph.in_degree(successor) - multiplicity

                    if predecessor_count[successor] == 0:
                        next_layer.append(successor)
                        del predecessor_count[successor]

            yield next_layer
            cur_layer = next_layer
            next_layer = []

    def collect_runs(self, namelist):
        """Return a set of runs of "op" nodes with the given names.

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
        tops_node = list(self.node_nums_in_topological_order())
        nodes_seen = dict(zip(tops_node, [False] * len(tops_node)))
        for node in tops_node:
            nd = self.multi_graph.node[node]
            if nd["type"] == "op" and nd["name"] in namelist \
                    and not nodes_seen[node]:
                group = [node]
                nodes_seen[node] = True
                s = list(self.multi_graph.successors(node))
                while len(s) == 1 and \
                        self.multi_graph.node[s[0]]["type"] == "op" and \
                        self.multi_graph.node[s[0]]["name"] in namelist:
                    group.append(s[0])
                    nodes_seen[s[0]] = True
                    s = list(self.multi_graph.successors(s[0]))
                if len(group) > 1:
                    group_list.append(tuple(group))
        return set(group_list)

    def count_ops(self):
        """Count the occurrences of operation names.

        Returns a dictionary of counts keyed on the operation name.
        """
        op_dict = {}
        for node in self.node_nums_in_topological_order():
            nd = self.multi_graph.node[node]
            name = nd["name"]
            if nd["type"] == "op":
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
