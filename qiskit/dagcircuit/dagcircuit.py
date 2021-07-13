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
from collections import OrderedDict, defaultdict
import copy
import itertools
import warnings
import math

import numpy as np
import retworkx as rx

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode
from qiskit.exceptions import MissingOptionalLibraryError


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

        # Circuit metadata
        self.metadata = None

        # Set of wires (Register,idx) in the dag
        self._wires = set()

        # Map from wire (Register,idx) to input nodes of the graph
        self.input_map = OrderedDict()

        # Map from wire (Register,idx) to output nodes of the graph
        self.output_map = OrderedDict()

        # Directed multigraph whose nodes are inputs, outputs, or operations.
        # Operation nodes have equal in- and out-degrees and carry
        # additional data about the operation, including the argument order
        # and parameter values.
        # Input nodes have out-degree 1 and output nodes have in-degree 1.
        # Edges carry wire labels (reg,idx) and each operation has
        # corresponding in- and out-edges with the same wire labels.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of Qubit/Clbit wires that the DAG acts on.
        self.qubits = []
        self.clbits = []

        self._global_phase = 0
        self._calibrations = defaultdict(dict)

        self.duration = None
        self.unit = "dt"

    def to_networkx(self):
        """Returns a copy of the DAGCircuit in networkx format."""
        try:
            import networkx as nx
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Networkx",
                name="DAG converter to networkx",
                pip_install="pip install networkx",
            ) from ex
        G = nx.MultiDiGraph()
        for node in self._multi_graph.nodes():
            G.add_node(node)
        for node_id in rx.topological_sort(self._multi_graph):
            for source_id, dest_id, edge in self._multi_graph.in_edges(node_id):
                G.add_edge(self._multi_graph[source_id], self._multi_graph[dest_id], wire=edge)
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
        Raises:
            MissingOptionalLibraryError: If networkx is not installed
            DAGCircuitError: If input networkx graph is malformed
        """
        try:
            import networkx as nx
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Networkx",
                name="DAG converter from networkx",
                pip_install="pip install networkx",
            ) from ex
        dag = DAGCircuit()
        for node in nx.topological_sort(graph):
            if node.type == "out":
                continue
            if node.type == "in":
                if isinstance(node.wire, Qubit):
                    dag.add_qubits([node.wire])
                elif isinstance(node.wire, Clbit):
                    dag.add_clbits([node.wire])
                else:
                    raise DAGCircuitError(f"unknown node wire type: {node.wire}")
            elif node.type == "op":
                dag.apply_operation_back(node.op.copy(), node.qargs, node.cargs)
        return dag

    @property
    def wires(self):
        """Return a list of the wires in order."""
        return self.qubits + self.clbits

    @property
    def node_counter(self):
        """
        Returns the number of nodes in the dag.
        """
        return len(self._multi_graph)

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

    def add_calibration(self, gate, qubits, schedule, params=None):
        """Register a low-level, custom pulse definition for the given gate.

        Args:
            gate (Union[Gate, str]): Gate information.
            qubits (Union[int, Tuple[int]]): List of qubits to be measured.
            schedule (Schedule): Schedule information.
            params (Optional[List[Union[float, Parameter]]]): A list of parameters.

        Raises:
            Exception: if the gate is of type string and params is None.
        """
        if isinstance(gate, Gate):
            self._calibrations[gate.name][(tuple(qubits), tuple(gate.params))] = schedule
        else:
            self._calibrations[gate][(tuple(qubits), tuple(params or []))] = schedule

    def has_calibration_for(self, node):
        """Return True if the dag has a calibration defined for the node operation. In this
        case, the operation does not need to be translated to the device basis.
        """
        if not self.calibrations or node.name not in self.calibrations:
            return False
        qubits = tuple(self.qubits.index(qubit) for qubit in node.qargs)
        params = []
        for p in node.op.params:
            if isinstance(p, ParameterExpression) and not p.parameters:
                params.append(float(p))
            else:
                params.append(p)
        params = tuple(params)
        return (qubits, params) in self.calibrations[node.name]

    def remove_all_ops_named(self, opname):
        """Remove all operation nodes with the given name."""
        for n in self.named_nodes(opname):
            self.remove_op_node(n)

    def add_qubits(self, qubits):
        """Add individual qubit wires."""
        if any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise DAGCircuitError("not a Qubit instance.")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGCircuitError("duplicate qubits %s" % duplicate_qubits)

        self.qubits.extend(qubits)
        for qubit in qubits:
            self._add_wire(qubit)

    def add_clbits(self, clbits):
        """Add individual clbit wires."""
        if any(not isinstance(clbit, Clbit) for clbit in clbits):
            raise DAGCircuitError("not a Clbit instance.")

        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGCircuitError("duplicate clbits %s" % duplicate_clbits)

        self.clbits.extend(clbits)
        for clbit in clbits:
            self._add_wire(clbit)

    def add_qreg(self, qreg):
        """Add all wires in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGCircuitError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGCircuitError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])
                self._add_wire(qreg[j])

    def add_creg(self, creg):
        """Add all wires in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGCircuitError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGCircuitError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] not in existing_clbits:
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

            inp_node = DAGNode(type="in", wire=wire)
            outp_node = DAGNode(type="out", wire=wire)
            input_map_id, output_map_id = self._multi_graph.add_nodes_from([inp_node, outp_node])
            inp_node._node_id = input_map_id
            outp_node._node_id = output_map_id
            self.input_map[wire] = inp_node
            self.output_map[wire] = outp_node
            self._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, wire)
        else:
            raise DAGCircuitError(f"duplicate wire {wire}")

    def _check_condition(self, name, condition):
        """Verify that the condition is valid.

        Args:
            name (string): used for error reporting
            condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)

        Raises:
            DAGCircuitError: if conditioning on an invalid register
        """
        if (
            condition is not None
            and condition[0] not in self.clbits
            and condition[0].name not in self.cregs
        ):
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
                raise DAGCircuitError(f"(qu)bit {wire} not found in {amap}")

    def _bits_in_condition(self, cond):
        """Return a list of bits in the given condition.

        Args:
            cond (tuple or None): optional condition (ClassicalRegister, int) or (Clbit, bool)

        Returns:
            list[Clbit]: list of classical bits

        Raises:
            CircuitError: if cond[0] is not ClassicalRegister or Clbit
        """
        if cond is None:
            return []
        elif isinstance(cond[0], ClassicalRegister):
            # Returns a list of all the cbits in the given creg cond[0].
            return cond[0][:]
        elif isinstance(cond[0], Clbit):
            # Returns a singleton list of the conditional cbit.
            return [cond[0]]
        else:
            raise CircuitError("Condition must be used with ClassicalRegister or Clbit.")

    def _add_op_node(self, op, qargs, cargs):
        """Add a new operation node to the graph and assign properties.

        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): list of quantum wires to attach to.
            cargs (list[Clbit]): list of classical wires to attach to.
        Returns:
            int: The integer node index for the new op node on the DAG
        """
        # Add a new operation node to the graph
        new_node = DAGNode(type="op", op=op, qargs=qargs, cargs=cargs)
        node_index = self._multi_graph.add_node(new_node)
        new_node._node_id = node_index
        return node_index

    def _copy_circuit_metadata(self):
        """Return a copy of source_dag with metadata but empty."""
        target_dag = DAGCircuit()
        target_dag.name = self.name
        target_dag._global_phase = self._global_phase
        target_dag.duration = self.duration
        target_dag.unit = self.unit
        target_dag.metadata = self.metadata

        target_dag.add_qubits(self.qubits)
        target_dag.add_clbits(self.clbits)

        for qreg in self.qregs.values():
            target_dag.add_qreg(qreg)
        for creg in self.cregs.values():
            target_dag.add_creg(creg)

        return target_dag

    def apply_operation_back(self, op, qargs=None, cargs=None, condition=None):
        """Apply an operation to the output of the circuit.

        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): qubits that op will be applied to
            cargs (list[Clbit]): cbits that op will be applied to
            condition (tuple or None): DEPRECATED optional condition (ClassicalRegister, int)
        Returns:
            DAGNode: the current max node

        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs

        """
        if condition:
            warnings.warn(
                "Use of condition arg is deprecated, set condition in instruction",
                DeprecationWarning,
            )
        op.condition = condition if op.condition is None else op.condition

        qargs = qargs or []
        cargs = cargs or []

        all_cbits = self._bits_in_condition(op.condition)
        all_cbits = set(all_cbits).union(cargs)

        self._check_condition(op.name, op.condition)
        self._check_bits(qargs, self.output_map)
        self._check_bits(all_cbits, self.output_map)

        node_index = self._add_op_node(op, qargs, cargs)

        # Add new in-edges from predecessors of the output nodes to the
        # operation node while deleting the old in-edges of the output nodes
        # and adding new edges from the operation node to each output node

        al = [qargs, all_cbits]
        self._multi_graph.insert_node_on_in_edges_multiple(
            node_index, [self.output_map[q]._node_id for q in itertools.chain(*al)]
        )
        return self._multi_graph[node_index]

    def apply_operation_front(self, op, qargs, cargs, condition=None):
        """Apply an operation to the input of the circuit.

        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): qubits that op will be applied to
            cargs (list[Clbit]): cbits that op will be applied to
            condition (tuple or None): DEPRECATED optional condition (ClassicalRegister, int)
        Returns:
            DAGNode: the current max node

        Raises:
            DAGCircuitError: if initial nodes connected to multiple out edges
        """
        if condition:
            warnings.warn(
                "Use of condition arg is deprecated, set condition in instruction",
                DeprecationWarning,
            )

        op.condition = condition if op.condition is None else op.condition
        all_cbits = self._bits_in_condition(op.condition)
        all_cbits.extend(cargs)

        self._check_condition(op.name, op.condition)
        self._check_bits(qargs, self.input_map)
        self._check_bits(all_cbits, self.input_map)
        node_index = self._add_op_node(op, qargs, cargs)

        # Add new out-edges to successors of the input nodes from the
        # operation node while deleting the old out-edges of the input nodes
        # and adding new edges to the operation node from each input node
        al = [qargs, all_cbits]
        self._multi_graph.insert_node_on_out_edges_multiple(
            node_index, [self.input_map[q]._node_id for q in itertools.chain(*al)]
        )
        return self._multi_graph[node_index]

    def _check_edgemap_registers(self, inbound_wires, inbound_regs):
        """Check that wiremap neither fragments nor leaves duplicate registers.

        1. There are no fragmented registers. A register in keyregs
        is fragmented if not all of its (qu)bits are renamed by edge_map.
        2. There are no duplicate registers. A register is duplicate if
        it appears in both self and keyregs but not in edge_map.

        Args:
            inbound_wires (list): a list of wires being mapped from the inbound dag
            inbound_regs (list): a list from registers from the inbound dag

        Returns:
            set(Register): the set of regs to add to self

        Raises:
            DAGCircuitError: if the wiremap fragments, or duplicates exist
        """
        add_regs = set()
        reg_frag_chk = {}

        for inbound_reg in inbound_regs:
            reg_frag_chk[inbound_reg] = {reg_bit: False for reg_bit in inbound_reg}

        for inbound_bit in inbound_wires:
            for inbound_reg in inbound_regs:
                if inbound_bit in inbound_reg:
                    reg_frag_chk[inbound_reg][inbound_bit] = True
                    break

        for inbound_reg, v in reg_frag_chk.items():
            s = set(v.values())
            if len(s) == 2:
                raise DAGCircuitError("inbound_wires fragments reg %s" % inbound_reg)
            if s == {False}:
                if inbound_reg.name in self.qregs or inbound_reg.name in self.cregs:
                    raise DAGCircuitError("unmapped duplicate reg %s" % inbound_reg)

                # Add registers that appear only in inbound_regs
                add_regs.add(inbound_reg)

        return add_regs

    def _check_wiremap_validity(self, wire_map, keymap, valmap):
        """Check that the wiremap is consistent.

        Check that the wiremap refers to valid wires and that
        those wires have consistent types.

        Args:
            wire_map (dict): map from Bit in keymap to Bit in valmap
            keymap (list): a list of wire_map keys
            valmap (dict): a map whose keys are wire_map values

        Raises:
            DAGCircuitError: if wire_map not valid
        """
        for k, v in wire_map.items():

            if k not in keymap:
                raise DAGCircuitError("invalid wire mapping key %s" % k)
            if v not in valmap:
                raise DAGCircuitError("invalid wire mapping value %s" % v)
            # TODO Support mapping from AncillaQubit to Qubit, since AncillaQubits are mapped to
            # Qubits upon being converted to an Instruction. Until this translation is fixed
            # and Instructions have a concept of ancilla qubits, this fix is required.
            if not (isinstance(k, type(v)) or isinstance(v, type(k))):
                raise DAGCircuitError(f"inconsistent wire_map at ({k},{v})")

    @staticmethod
    def _map_condition(wire_map, condition, target_cregs):
        """Use the wire_map dict to change the condition tuple's creg name.

        Args:
            wire_map (dict): a map from source wires to destination wires
            condition (tuple or None): (ClassicalRegister,int)
            target_cregs (list[ClassicalRegister]): List of all cregs in the
              target circuit onto which the condition might possibly be mapped.
        Returns:
            tuple(ClassicalRegister,int): new condition
        Raises:
            DAGCircuitError: if condition register not in wire_map, or if
                wire_map maps condition onto more than one creg, or if the
                specified condition is not present in a classical register.
        """

        if condition is None:
            new_condition = None
        else:
            # if there is a condition, map the condition bits to the
            # composed cregs based on the wire_map
            is_reg = False
            if isinstance(condition[0], Clbit):
                cond_creg = [condition[0]]
            else:
                cond_creg = condition[0]
                is_reg = True
            cond_val = condition[1]
            new_cond_val = 0
            new_creg = None
            bits_in_condcreg = [bit for bit in wire_map if bit in cond_creg]
            for bit in bits_in_condcreg:
                if is_reg:
                    try:
                        candidate_creg = next(
                            creg for creg in target_cregs if wire_map[bit] in creg
                        )
                    except StopIteration as ex:
                        raise DAGCircuitError(
                            "Did not find creg containing " "mapped clbit in conditional."
                        ) from ex
                else:
                    # If cond is on a single Clbit then the candidate_creg is
                    # the target Clbit to which 'bit' is mapped to.
                    candidate_creg = wire_map[bit]

                if new_creg is None:
                    new_creg = candidate_creg
                elif new_creg != candidate_creg:
                    # Raise if wire_map maps condition creg on to more than one
                    # creg in target DAG.
                    raise DAGCircuitError(
                        "wire_map maps conditional " "register onto more than one creg."
                    )

                if not is_reg:
                    # If the cond is on a single Clbit then the new_cond_val is the
                    # same as the cond_val since the new_creg is also a single Clbit.
                    new_cond_val = cond_val
                elif 2 ** (cond_creg[:].index(bit)) & cond_val:
                    # If the conditional values of the Clbit 'bit' is 1 then the new_cond_val
                    # is updated such that the conditional value of the Clbit to which 'bit'
                    # is mapped to in new_creg is 1.
                    new_cond_val += 2 ** (new_creg[:].index(wire_map[bit]))
            if new_creg is None:
                raise DAGCircuitError("Condition registers not found in wire_map.")
            new_condition = (new_creg, new_cond_val)
        return new_condition

    def extend_back(self, dag, edge_map=None):
        """DEPRECATED: Add `dag` at the end of `self`, using `edge_map`."""
        warnings.warn(
            "dag.extend_back is deprecated, please use dag.compose.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """DEPRECATED: use DAGCircuit.compose() instead."""
        warnings.warn(
            "dag.compose_back is deprecated, please use dag.compose.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        if len(other.qubits) > len(self.qubits) or len(other.clbits) > len(self.clbits):
            raise DAGCircuitError(
                "Trying to compose with another DAGCircuit " "which has more 'in' edges."
            )

        if edge_map is not None:
            warnings.warn(
                "edge_map arg as a dictionary is deprecated. "
                "Use qubits and clbits args to specify a list of "
                "self edges to compose onto.",
                DeprecationWarning,
                stacklevel=2,
            )

        # number of qubits and clbits must match number in circuit or None
        identity_qubit_map = dict(zip(other.qubits, self.qubits))
        identity_clbit_map = dict(zip(other.clbits, self.clbits))
        if qubits is None:
            qubit_map = identity_qubit_map
        elif len(qubits) != len(other.qubits):
            raise DAGCircuitError(
                "Number of items in qubits parameter does not"
                " match number of qubits in the circuit."
            )
        else:
            qubit_map = {
                other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
                for i, q in enumerate(qubits)
            }
        if clbits is None:
            clbit_map = identity_clbit_map
        elif len(clbits) != len(other.clbits):
            raise DAGCircuitError(
                "Number of items in clbits parameter does not"
                " match number of clbits in the circuit."
            )
        else:
            clbit_map = {
                other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
                for i, c in enumerate(clbits)
            }
        edge_map = edge_map or {**qubit_map, **clbit_map} or None

        # if no edge_map, try to do a 1-1 mapping in order
        if edge_map is None:
            edge_map = {**identity_qubit_map, **identity_clbit_map}

        # Check the edge_map for duplicate values
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError("duplicates in wire_map")

        # Compose
        if inplace:
            dag = self
        else:
            dag = copy.deepcopy(self)
        dag.global_phase += other.global_phase

        for gate, cals in other.calibrations.items():
            dag._calibrations[gate].update(cals)

        for nd in other.topological_nodes():
            if nd.type == "in":
                # if in edge_map, get new name, else use existing name
                m_wire = edge_map.get(nd.wire, nd.wire)
                # the mapped wire should already exist
                if m_wire not in dag.output_map:
                    raise DAGCircuitError(
                        "wire %s[%d] not in self" % (m_wire.register.name, m_wire.index)
                    )
                if nd.wire not in other._wires:
                    raise DAGCircuitError(
                        "inconsistent wire type for %s[%d] in other"
                        % (nd.register.name, nd.wire.index)
                    )
            elif nd.type == "out":
                # ignore output nodes
                pass
            elif nd.type == "op":
                condition = dag._map_condition(edge_map, nd.op.condition, dag.cregs.values())
                dag._check_condition(nd.name, condition)
                m_qargs = list(map(lambda x: edge_map.get(x, x), nd.qargs))
                m_cargs = list(map(lambda x: edge_map.get(x, x), nd.cargs))
                op = nd.op.copy()
                op.condition = condition
                dag.apply_operation_back(op, m_qargs, m_cargs)
            else:
                raise DAGCircuitError("bad node type %s" % nd.type)

        if not inplace:
            return dag
        else:
            return None

    def reverse_ops(self):
        """Reverse the operations in the ``self`` circuit.

        Returns:
            DAGCircuit: the reversed dag.
        """
        # TODO: speed up
        # pylint: disable=cyclic-import
        from qiskit.converters import dag_to_circuit, circuit_to_dag

        qc = dag_to_circuit(self)
        reversed_qc = qc.reverse_ops()
        reversed_dag = circuit_to_dag(reversed_qc)
        return reversed_dag

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
            nodes = [
                node for node in self.nodes_on_wire(wire, only_ops=False) if node.name not in ignore
            ]
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
        try:
            depth = rx.dag_longest_path_length(self._multi_graph) - 1
        except rx.DAGHasCycle as ex:
            raise DAGCircuitError("not a DAG") from ex
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
        return len(self.qubits)

    def num_clbits(self):
        """Return the total number of classical bits used by the circuit."""
        return len(self.clbits)

    def num_tensor_factors(self):
        """Compute how many components the circuit can decompose into."""
        return rx.number_weakly_connected_components(self._multi_graph)

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
        if node.op.condition is not None:
            wire_tot += node.op.condition[0].size

        if len(wires) != wire_tot:
            raise DAGCircuitError("expected %d wires, got %d" % (wire_tot, len(wires)))

    def _make_pred_succ_maps(self, node):
        """Return predecessor and successor dictionaries.

        Args:
            node (DAGNode): reference to multi_graph node

        Returns:
            tuple(dict): tuple(predecessor_map, successor_map)
                These map from wire (Register, int) to the node ids for the
                predecessor (successor) nodes of the input node.
        """

        pred_map = {e[2]: e[0] for e in self._multi_graph.in_edges(node._node_id)}
        succ_map = {e[2]: e[1] for e in self._multi_graph.out_edges(node._node_id)}
        return pred_map, succ_map

    def _full_pred_succ_maps(self, pred_map, succ_map, input_circuit, wire_map):
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
                full_pred_map[w] = self._multi_graph.predecessors(self.output_map[w])[0]
                if len(self._multi_graph.predecessors(self.output_map[w])) != 1:
                    raise DAGCircuitError(
                        "too many predecessors for %s[%d] " "output node" % (w.register, w.index)
                    )

        return full_pred_map, full_succ_map

    def __eq__(self, other):
        # Try to convert to float, but in case of unbound ParameterExpressions
        # a TypeError will be raise, fallback to normal equality in those
        # cases

        try:
            self_phase = float(self.global_phase)
            other_phase = float(other.global_phase)
            if (
                abs((self_phase - other_phase + np.pi) % (2 * np.pi) - np.pi) > 1.0e-10
            ):  # TODO: atol?
                return False
        except TypeError:
            if self.global_phase != other.global_phase:
                return False
        if self.calibrations != other.calibrations:
            return False

        self_bit_indices = {bit: idx for idx, bit in enumerate(self.qubits + self.clbits)}
        other_bit_indices = {bit: idx for idx, bit in enumerate(other.qubits + other.clbits)}

        self_qreg_indices = [
            (regname, [self_bit_indices[bit] for bit in reg]) for regname, reg in self.qregs.items()
        ]
        self_creg_indices = [
            (regname, [self_bit_indices[bit] for bit in reg]) for regname, reg in self.cregs.items()
        ]

        other_qreg_indices = [
            (regname, [other_bit_indices[bit] for bit in reg])
            for regname, reg in other.qregs.items()
        ]
        other_creg_indices = [
            (regname, [other_bit_indices[bit] for bit in reg])
            for regname, reg in other.cregs.items()
        ]
        if self_qreg_indices != other_qreg_indices or self_creg_indices != other_creg_indices:
            return False

        def node_eq(node_self, node_other):
            return DAGNode.semantic_eq(node_self, node_other, self_bit_indices, other_bit_indices)

        return rx.is_isomorphic_node_match(self._multi_graph, other._multi_graph, node_eq)

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order
        """

        def _key(x):
            return x.sort_key

        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))

    def topological_op_nodes(self):
        """
        Yield op nodes in topological order.

        Returns:
            generator(DAGNode): op node in topological order
        """
        return (nd for nd in self.topological_nodes() if nd.type == "op")

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
        in_dag = input_dag
        condition = None if node.type != "op" else node.op.condition

        # the dag must be amended if used in a
        # conditional context. delete the op nodes and replay
        # them with the condition.
        if condition:
            in_dag = copy.deepcopy(input_dag)
            in_dag.add_creg(condition[0])
            to_replay = []
            for sorted_node in in_dag.topological_nodes():
                if sorted_node.type == "op":
                    sorted_node.op.condition = condition
                    to_replay.append(sorted_node)
            for input_node in in_dag.op_nodes():
                in_dag.remove_op_node(input_node)
            for replay_node in to_replay:
                in_dag.apply_operation_back(replay_node.op, replay_node.qargs, replay_node.cargs)

        if in_dag.global_phase:
            self.global_phase += in_dag.global_phase

        if wires is None:
            wires = in_dag.wires

        self._check_wires_list(wires, node)

        # Create a proxy wire_map to identify fragments and duplicates
        # and determine what registers need to be added to self
        add_qregs = self._check_edgemap_registers(wires, in_dag.qregs.values())
        for qreg in add_qregs:
            self.add_qreg(qreg)

        add_cregs = self._check_edgemap_registers(wires, in_dag.cregs.values())
        for creg in add_cregs:
            self.add_creg(creg)

        # Replace the node by iterating through the input_circuit.
        # Constructing and checking the validity of the wire_map.
        # If a gate is conditioned, we expect the replacement subcircuit
        # to depend on those condition bits as well.
        if node.type != "op":
            raise DAGCircuitError('expected node type "op", got %s' % node.type)

        condition_bit_list = self._bits_in_condition(condition)

        wire_map = dict(zip(wires, list(node.qargs) + list(node.cargs) + list(condition_bit_list)))
        self._check_wiremap_validity(wire_map, wires, self.input_map)
        pred_map, succ_map = self._make_pred_succ_maps(node)
        full_pred_map, full_succ_map = self._full_pred_succ_maps(
            pred_map, succ_map, in_dag, wire_map
        )

        if condition_bit_list:
            # If we are replacing a conditional node, map input dag through
            # wire_map to verify that it will not modify any of the conditioning
            # bits.
            condition_bits = set(condition_bit_list)

            for op_node in in_dag.op_nodes():
                mapped_cargs = {wire_map[carg] for carg in op_node.cargs}

                if condition_bits & mapped_cargs:
                    raise DAGCircuitError(
                        "Mapped DAG would alter clbits " "on which it would be conditioned."
                    )

        # Now that we know the connections, delete node
        self._multi_graph.remove_node(node._node_id)

        # Iterate over nodes of input_circuit
        for sorted_node in in_dag.topological_op_nodes():
            # Insert a new node
            condition = self._map_condition(wire_map, sorted_node.op.condition, self.cregs.values())
            m_qargs = list(map(lambda x: wire_map.get(x, x), sorted_node.qargs))
            m_cargs = list(map(lambda x: wire_map.get(x, x), sorted_node.cargs))
            node_index = self._add_op_node(sorted_node.op, m_qargs, m_cargs)

            # Add edges from predecessor nodes to new node
            # and update predecessor nodes that change
            all_cbits = self._bits_in_condition(condition)
            all_cbits.extend(m_cargs)
            al = [m_qargs, all_cbits]
            for q in itertools.chain(*al):
                self._multi_graph.add_edge(full_pred_map[q], node_index, q)
                full_pred_map[q] = node_index

        # Connect all predecessors and successors, and remove
        # residual edges between input and output nodes
        for w in full_pred_map:
            self._multi_graph.add_edge(full_pred_map[w], full_succ_map[w], w)
            o_pred = self._multi_graph.predecessors(self.output_map[w]._node_id)
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

        if node.type != "op":
            raise DAGCircuitError('Only DAGNodes of type "op" can be replaced.')

        if node.op.num_qubits != op.num_qubits or node.op.num_clbits != op.num_clbits:
            raise DAGCircuitError(
                "Cannot replace node of width ({} qubits, {} clbits) with "
                "instruction of mismatched width ({} qubits, {} clbits).".format(
                    node.op.num_qubits, node.op.num_clbits, op.num_qubits, op.num_clbits
                )
            )

        if inplace:
            save_condition = node.op.condition
            node.op = op
            node.name = op.name
            node.op.condition = save_condition
            return node

        new_node = copy.copy(node)
        save_condition = new_node.op.condition
        new_node.op = op
        new_node.name = op.name
        new_node.op.condition = save_condition
        self._multi_graph[node._node_id] = new_node
        return new_node

    def node(self, node_id):
        """Get the node in the dag.

        Args:
            node_id(int): Node identifier.

        Returns:
            node: the node.
        """
        return self._multi_graph[node_id]

    def nodes(self):
        """Iterator for node values.

        Yield:
            node: the node.
        """
        yield from self._multi_graph.nodes()

    def edges(self, nodes=None):
        """Iterator for edge values and source and dest node

        This works by returning the output edges from the specified nodes. If
        no nodes are specified all edges from the graph are returned.

        Args:
            nodes(DAGNode|list(DAGNode): Either a list of nodes or a single
                input node. If none is specified all edges are returned from
                the graph.

        Yield:
            edge: the edge in the same format as out_edges the tuple
                (source node, destination node, edge data)
        """
        if nodes is None:
            nodes = self._multi_graph.nodes()

        elif isinstance(nodes, DAGNode):
            nodes = [nodes]
        for node in nodes:
            raw_nodes = self._multi_graph.out_edges(node._node_id)
            for source, dest, edge in raw_nodes:
                yield (self._multi_graph[source], self._multi_graph[dest], edge)

    def op_nodes(self, op=None, include_directives=True):
        """Get the list of "op" nodes in the dag.

        Args:
            op (Type): :class:`qiskit.circuit.Instruction` subclass op nodes to
                return. If None, return all op nodes.
            include_directives (bool): include `barrier`, `snapshot` etc.

        Returns:
            list[DAGNode]: the list of node ids containing the given op.
        """
        nodes = []
        for node in self._multi_graph.nodes():
            if node.type == "op":
                if not include_directives and node.op._directive:
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
        for node in self._multi_graph.nodes():
            if node.type == "op" and node.op.name in names:
                named_nodes.append(node)
        return named_nodes

    def twoQ_gates(self):
        """Get list of 2-qubit gates. Ignore snapshot, barriers, and the like."""
        warnings.warn(
            "deprecated function, use dag.two_qubit_ops(). "
            "filter output by isinstance(op, Gate) to only get unitary Gates.",
            DeprecationWarning,
            stacklevel=2,
        )
        two_q_gates = []
        for node in self.gate_nodes():
            if len(node.qargs) == 2:
                two_q_gates.append(node)
        return two_q_gates

    def threeQ_or_more_gates(self):
        """Get list of 3-or-more-qubit gates: (id, data)."""
        warnings.warn(
            "deprecated function, use dag.multi_qubit_ops(). "
            "filter output by isinstance(op, Gate) to only get unitary Gates.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        return [self._multi_graph[x] for x in rx.dag_longest_path(self._multi_graph)]

    def successors(self, node):
        """Returns iterator of the successors of a node as DAGNodes."""
        return iter(self._multi_graph.successors(node._node_id))

    def predecessors(self, node):
        """Returns iterator of the predecessors of a node as DAGNodes."""
        return iter(self._multi_graph.predecessors(node._node_id))

    def is_successor(self, node, node_succ):
        """Checks if a second node is in the successors of node."""
        return self._multi_graph.has_edge(node._node_id, node_succ._node_id)

    def is_predecessor(self, node, node_pred):
        """Checks if a second node is in the predecessors of node."""
        return self._multi_graph.has_edge(node_pred._node_id, node._node_id)

    def quantum_predecessors(self, node):
        """Returns iterator of the predecessors of a node that are
        connected by a quantum edge as DAGNodes."""
        return iter(
            self._multi_graph.find_predecessors_by_edge(
                node._node_id, lambda edge_data: isinstance(edge_data, Qubit)
            )
        )

    def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGNodes."""
        return {self._multi_graph[x] for x in rx.ancestors(self._multi_graph, node._node_id)}

    def descendants(self, node):
        """Returns set of the descendants of a node as DAGNodes."""
        return {self._multi_graph[x] for x in rx.descendants(self._multi_graph, node._node_id)}

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
        and [DAGNode] is its successors in  BFS order.
        """
        return iter(rx.bfs_successors(self._multi_graph, node._node_id))

    def quantum_successors(self, node):
        """Returns iterator of the successors of a node that are
        connected by a quantum edge as DAGNodes."""
        return iter(
            self._multi_graph.find_successors_by_edge(
                node._node_id, lambda edge_data: isinstance(edge_data, Qubit)
            )
        )

    def remove_op_node(self, node):
        """Remove an operation node n.

        Add edges from predecessors to successors.
        """
        if node.type != "op":
            raise DAGCircuitError(
                'The method remove_op_node only works on op node types. An "%s" '
                "node type was wrongly provided." % node.type
            )

        self._multi_graph.remove_node_retain_edges(
            node._node_id, use_outgoing=False, condition=lambda edge1, edge2: edge1 == edge2
        )

    def remove_ancestors_of(self, node):
        """Remove all of the ancestor operation nodes of node."""
        anc = rx.ancestors(self._multi_graph, node)
        # TODO: probably better to do all at once using
        # multi_graph.remove_nodes_from; same for related functions ...

        for anc_node in anc:
            if anc_node.type == "op":
                self.remove_op_node(anc_node)

    def remove_descendants_of(self, node):
        """Remove all of the descendant operation nodes of node."""
        desc = rx.descendants(self._multi_graph, node)
        for desc_node in desc:
            if desc_node.type == "op":
                self.remove_op_node(desc_node)

    def remove_nonancestors_of(self, node):
        """Remove all of the non-ancestors operation nodes of node."""
        anc = rx.ancestors(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(anc))
        for n in comp:
            if n.type == "op":
                self.remove_op_node(n)

    def remove_nondescendants_of(self, node):
        """Remove all of the non-descendants operation nodes of node."""
        dec = rx.descendants(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(dec))
        for n in comp:
            if n.type == "op":
                self.remove_op_node(n)

    def front_layer(self):
        """Return a list of op nodes in the first layer of this dag."""
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)  # Remove input nodes
        except StopIteration:
            return []

        op_nodes = [node for node in next(graph_layers) if node.type == "op"]

        return op_nodes

    def layers(self):
        """Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.

        A layer is a circuit whose gates act on disjoint qubits, i.e.,
        a layer has depth 1. The total number of layers equals the
        circuit depth d. The layers are indexed from 0 to d-1 with the
        earliest layer at index 0. The layers are constructed using a
        greedy algorithm. Each returned layer is a dict containing
        {"graph": circuit graph, "partition": list of qubit lists}.

        The returned layer contains new (but semantically equivalent) DAGNodes.
        These are not the same as nodes of the original dag, but are equivalent
        via DAGNode.semantic_eq(node1, node2).

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
            # Drawing tools rely on _node_id to infer order of node creation
            # so we need this to be preserved by layers()
            op_nodes.sort(key=lambda nd: nd._node_id)

            # Stop yielding once there are no more op_nodes in a layer.
            if not op_nodes:
                return

            # Construct a shallow copy of self
            new_layer = self._copy_circuit_metadata()

            for node in op_nodes:
                # this creates new DAGNodes in the new_layer
                new_layer.apply_operation_back(node.op, node.qargs, node.cargs)

            # The quantum registers that have an operation in this layer.
            support_list = [
                op_node.qargs for op_node in new_layer.op_nodes() if not op_node.op._directive
            ]

            yield {"graph": new_layer, "partition": support_list}

    def serial_layers(self):
        """Yield a layer for all gates of this circuit.

        A serial layer is a circuit with one gate. The layers have the
        same structure as in layers().
        """
        for next_node in self.topological_op_nodes():
            new_layer = self._copy_circuit_metadata()

            # Save the support of the operation we add to the layer
            support_list = []
            # Operation data
            op = copy.copy(next_node.op)
            qa = copy.copy(next_node.qargs)
            ca = copy.copy(next_node.cargs)
            co = copy.copy(next_node.op.condition)
            _ = self._bits_in_condition(co)

            # Add node to new_layer
            new_layer.apply_operation_back(op, qa, ca)
            # Add operation to partition
            if not next_node.op._directive:
                support_list.append(list(qa))
            l_dict = {"graph": new_layer, "partition": support_list}
            yield l_dict

    def multigraph_layers(self):
        """Yield layers of the multigraph."""
        first_layer = [x._node_id for x in self.input_map.values()]
        return iter(rx.layers(self._multi_graph, first_layer))

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

        def filter_fn(node):
            return node.type == "op" and node.name in namelist and node.op.condition is None

        group_list = rx.collect_runs(self._multi_graph, filter_fn)
        return {tuple(x) for x in group_list}

    def collect_1q_runs(self):
        """Return a set of non-conditional runs of 1q "op" nodes."""

        def filter_fn(node):
            return (
                node.type == "op"
                and len(node.qargs) == 1
                and len(node.cargs) == 0
                and node.op.condition is None
                and not node.op.is_parameterized()
                and isinstance(node.op, Gate)
                and hasattr(node.op, "__array__")
            )

        return rx.collect_runs(self._multi_graph, filter_fn)

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
            raise DAGCircuitError("The given wire %s is not present in the circuit" % str(wire))

        more_nodes = True
        while more_nodes:
            more_nodes = False
            # allow user to just get ops on the wire - not the input/output nodes
            if current_node.type == "op" or not only_ops:
                yield current_node

            try:
                current_node = self._multi_graph.find_adjacent_node_by_edge(
                    current_node._node_id, lambda x: wire == x
                )
                more_nodes = True
            except rx.NoSuitableNeighbors:
                pass

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
        path = path[1:-1]  # remove qubits at beginning and end of path
        for node in path:
            name = node.name
            if name not in op_dict:
                op_dict[name] = 1
            else:
                op_dict[name] += 1
        return op_dict

    def properties(self):
        """Return a dictionary of circuit properties."""
        summary = {
            "size": self.size(),
            "depth": self.depth(),
            "width": self.width(),
            "qubits": self.num_qubits(),
            "bits": self.num_clbits(),
            "factors": self.num_tensor_factors(),
            "operations": self.count_ops(),
        }
        return summary

    def draw(self, scale=0.7, filename=None, style="color"):
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
