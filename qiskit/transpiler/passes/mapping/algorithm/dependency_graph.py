# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A dependency graph represents precedence relations of the gates in a quantum circuit considering
commutation rules. Each node represents gates in the circuit. Each directed edge represents
dependency of two gates. For example, gate g1 must be applied before gate g2 if and only if
there exists a path from g1 to g2.
"""
import copy
from collections import defaultdict
from typing import List, Set, Tuple

import networkx as nx

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.mapper import Layout
from qiskit.transpiler import TranspilerError


class DependencyGraph:
    """
    Create a dependency graph of a quantum circuit with a chosen commutation rule.
    """

    def __init__(self,
                 quantum_circuit: QuantumCircuit,
                 graph_type: str = "basic"):
        """
        Construct the dependency graph of `quantum_circuit` considering commutations/dependencies
        specified by `graph_type`.

        Args:
            quantum_circuit: A quantum circuit whose dependency graph to be constructed.
            graph_type: Which type of dependency is considered.
                - "xz_commute": consider four commutation rules proposed in [Itoko et. al. 2019].
                - "basic": consider only the commutation between gates without sharing qubits.
                - "layer": fix layers and add dependencies between layers to `basic`.

        Raises:
            TranspilerError: if `graph_type` is not one of the types listed above.
        """

        self._gates = quantum_circuit.data

        self._graph = nx.DiGraph()  # dependency graph (including 1-qubit gates)

        for i, _ in enumerate(self._gates):
            self._graph.add_node(i)

        self.qubits = set()
        for i, _ in enumerate(self._gates):
            self.qubits.update(self.qargs(i))

        if graph_type == "xz_commute":
            self._create_xz_graph()
        elif graph_type == "basic":
            self._create_basic_graph()
        elif graph_type == "layer":
            self._create_layer_graph()
        else:
            raise TranspilerError("Unknown graph_type:" + graph_type)

        # remove redundant edges in dependency graph
        self.remove_redundancy(self._graph)

    def _create_xz_graph(self):
        z_gates = ["u1", "rz", "s", "t", "z", "sdg", "tdg"]
        x_gates = ["rx", "x"]
        b_gates = ["u3", "h", "u2", "ry", "barrier", "measure", "swap", "y"]
        # construct commutation-rules-aware dependency graph
        for n in self._graph.nodes():
            if self._gates[n].name in x_gates:
                [b] = self._gates[n].qargs
                [pgow] = self._prior_gates_on_wire(self._gates, n)
                z_flag = False
                for m in pgow:
                    gate = self._gates[m]
                    if gate.name in b_gates:
                        self._graph.add_edge(m, n)
                        break
                    elif gate.name in x_gates or (gate.name == "cx" and gate.qargs[1] == b):
                        if z_flag:
                            break
                        else:
                            continue
                    elif gate.name in z_gates or (gate.name == "cx" and gate.qargs[0] == b):
                        self._graph.add_edge(m, n)
                        z_flag = True
                    else:
                        raise TranspilerError("Unknown gate: " + gate.name)
            elif self._gates[n].name in z_gates:
                [b] = self._gates[n].qargs
                [pgow] = self._prior_gates_on_wire(self._gates, n)
                x_flag = False
                for m in pgow:
                    gate = self._gates[m]
                    if gate.name in b_gates:
                        self._graph.add_edge(m, n)
                        break
                    elif gate.name in x_gates or (gate.name == "cx" and gate.qargs[1] == b):
                        self._graph.add_edge(m, n)
                        x_flag = True
                    elif gate.name in z_gates or (gate.name == "cx" and gate.qargs[0] == b):
                        if x_flag:
                            break
                        else:
                            continue
                    else:
                        raise TranspilerError("Unknown gate: " + gate.name)
            elif self._gates[n].name == "cx":
                cbit, tbit = self._gates[n].qargs
                [cpgow, tpgow] = self._prior_gates_on_wire(self._gates, n)

                z_flag = False
                for m in tpgow:  # target bit: bt
                    gate = self._gates[m]
                    if gate.name in b_gates:
                        self._graph.add_edge(m, n)
                        break
                    elif gate.name in x_gates or (gate.name == "cx" and gate.qargs[1] == tbit):
                        if z_flag:
                            break
                        else:
                            continue
                    elif gate.name in z_gates or (gate.name == "cx" and gate.qargs[0] == tbit):
                        self._graph.add_edge(m, n)
                        z_flag = True
                    else:
                        raise TranspilerError("Unknown gate: " + gate.name)

                x_flag = False
                for m in cpgow:  # control bit: bc
                    gate = self._gates[m]
                    if gate.name in b_gates:
                        self._graph.add_edge(m, n)
                        break
                    elif gate.name in x_gates or (gate.name == "cx" and gate.qargs[1] == cbit):
                        self._graph.add_edge(m, n)
                        x_flag = True
                    elif gate.name in z_gates or (gate.name == "cx" and gate.qargs[0] == cbit):
                        if x_flag:
                            break
                        else:
                            continue
                    else:
                        raise TranspilerError("Unknown gate: " + gate.name)
            elif self._gates[n].name in b_gates:
                for i, pgow in enumerate(self._prior_gates_on_wire(self._gates, n)):
                    b = self._all_args(n)[i]
                    x_flag, z_flag = False, False
                    for m in pgow:
                        gate = self._gates[m]
                        if gate.name in b_gates:
                            self._graph.add_edge(m, n)
                            break
                        elif gate.name in x_gates or (gate.name == "cx" and gate.qargs[1] == b):
                            if z_flag:
                                break
                            else:
                                self._graph.add_edge(m, n)
                                x_flag = True
                        elif gate.name in z_gates or (gate.name == "cx" and gate.qargs[0] == b):
                            if x_flag:
                                break
                            else:
                                self._graph.add_edge(m, n)
                                z_flag = True
                        else:
                            raise TranspilerError("Unknown gate: " + gate.name)
            else:
                raise TranspilerError("Unknown gate: " + self._gates[n].name)

    def _create_basic_graph(self):
        for n in self._graph.nodes():
            for pgow in self._prior_gates_on_wire(self._gates, n):
                m = next(pgow, -1)
                if m != -1:
                    self._graph.add_edge(m, n)

    def _create_layer_graph(self):
        self._create_basic_graph()

        # construct CNOT layers
        layers = []
        wire = defaultdict(int)
        for n in self._graph.nodes():
            qargs = self.qargs(n)
            if self.gate_name(n) == "cx":  # consider only CNOTs
                i = 1 + max(wire[qargs[0]], wire[qargs[1]])
                wire[qargs[0]] = i
                wire[qargs[1]] = i
                if len(layers) > i:
                    layers[i].append(n)
                else:
                    layers.append([n])

        # Add more edges to basic graph for fixing layers
        for i in range(len(layers) - 1):
            j = i + 1
            for icx in layers[i]:
                for jcx in layers[j]:
                    self._graph.add_edge(icx, jcx)

    def n_nodes(self) -> int:
        """Number of the nodes
        Returns:
            Number of the nodes in this graph.
        """
        return self._graph.__len__()

    def qargs(self, i: int) -> List[Tuple[QuantumRegister, int]]:
        """Qubit arguments of the gate
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            List of qubit arguments.
        """
        return self._gates[i].qargs

    def _all_args(self, i: int) -> List[Tuple[QuantumRegister, int]]:
        """Qubit and classical-bit arguments of the gate
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            List of all arguments.
        """
        return self._gates[i].qargs + self._gates[i].cargs

    def gate_name(self, i: int) -> str:
        """Name of the gate
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            Name of the gate.
        """
        return self._gates[i].name

    def head_gates(self) -> Set[int]:
        """Gates which can be applicable prior to the other gates
        Returns:
            Set of indices of the gates.
        """
        return frozenset([n for n in self._graph.nodes() if len(self._graph.in_edges(n)) == 0])

    def gr_successors(self, i: int) -> List[int]:
        """Successor gates in Gr (transitive reduction) of the gate
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            Set of indices of the successor gates.
        """
        return self._graph.successors(i)

    def descendants(self, i: int) -> Set[int]:
        """Descendant gates of gate `i`
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            Set of indices of the descendant gates.
        """
        return nx.descendants(self._graph, i)

    def ancestors(self, i: int) -> Set[int]:
        """Ancestor gates of gate `i`
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            Set of indices of the ancestor gates.
        """
        return nx.ancestors(self._graph, i)

    def gate(self, gidx: int, layout: Layout, physical_qreg: QuantumRegister) -> Instruction:
        """Convert acting qubits of gate `gidx` from virtual qubits to physical ones.
        Args:
            gidx: Index of the gate in the `self._gates`
            layout: Layout used in conversion
            physical_qreg: Register of physical qubit
        Returns:
            Converted gate with physical qubit.
        Raises:
            TranspilerError: if virtual qubit of the gate `gidx` is not found in the layout.
        """
        gate = copy.deepcopy(self._gates[gidx])
        for i, virtual_qubit in enumerate(gate.qargs):
            if virtual_qubit in layout.get_virtual_bits().keys():
                gate.qargs[i] = (physical_qreg, layout[virtual_qubit])
            else:
                raise TranspilerError("virtual_qubit must be in layout")
        return gate

    @staticmethod
    def remove_redundancy(graph):
        """remove redundant edges in DAG (= change `graph` to its transitive reduction)
        """
        edges = list(graph.edges())
        for edge in edges:
            graph.remove_edge(edge[0], edge[1])
            if not nx.has_path(graph, edge[0], edge[1]):
                graph.add_edge(edge[0], edge[1])

    @staticmethod
    def _prior_gates_on_wire(gate_list, toidx):
        res = []
        for qarg in gate_list[toidx].qargs:
            gates = []
            for i, gate in enumerate(gate_list[:toidx]):
                if qarg in gate.qargs:
                    gates.append(i)
            res.append(reversed(gates))
        for carg in gate_list[toidx].cargs:
            gates = []
            for i, gate in enumerate(gate_list[:toidx]):
                if carg in gate.cargs:
                    gates.append(i)
            res.append(reversed(gates))
        return res


if __name__ == '__main__':
    pass
