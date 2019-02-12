# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Dependency graph
"""
import copy
from collections import defaultdict
from typing import List, Set, Tuple

import networkx as nx

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.mapper import MapperError


class DependencyGraph:
    """
    A Dependency graph expresses precedence relations of gates in a quantum circuit considering commutation rules.

    """

    def __init__(self,
                 quantum_circuit: QuantumCircuit,
                 graph_type: str = "basic"):

        self.L = quantum_circuit.data

        self.G = nx.DiGraph()  # dependency graph (including 1-qubit gates)

        for i, _ in enumerate(self.L):
            self.G.add_node(i)

        self.qubits = set()
        for i, _ in enumerate(self.L):
            self.qubits.update(self.qargs(i))

        if graph_type == "xz_commute":
            self._create_xz_graph()
        elif graph_type == "basic":
            self._create_basic_graph()
        elif graph_type == "layer":
            self._create_layer_graph()
        else:
            raise MapperError("Unknown graph_type:" + graph_type)

        # remove redundant edges in dependency graph
        self.remove_redundancy(self.G)

    def _create_xz_graph(self):
        Z_GATES = ["u1", "rz", "s", "t", "z", "sdg", "tdg"]
        X_GATES = ["rx", "x"]
        B_GATES = ["u3", "h", "u2", "ry", "barrier", "measure", "swap", "y"]
        # construct commutation-rules-aware dependency graph
        for n in self.G.nodes():
            if self.L[n].name in X_GATES:
                [b] = self.L[n].qargs
                [pgow] = self._prior_gates_on_wire(self.L, n)
                Z_FLAG = False
                for m in pgow:
                    g = self.L[m]
                    if g.name in B_GATES:
                        self.G.add_edge(m, n)
                        break
                    elif g.name in X_GATES or (g.name == "cx" and g.qargs[1] == b):
                        if Z_FLAG:
                            break
                        else:
                            continue
                    elif g.name in Z_GATES or (g.name == "cx" and g.qargs[0] == b):
                        self.G.add_edge(m, n)
                        Z_FLAG = True
                    else:
                        raise MapperError("Unknown gate: " + g.name)
            elif self.L[n].name in Z_GATES:
                [b] = self.L[n].qargs
                [pgow] = self._prior_gates_on_wire(self.L, n)
                X_FLAG = False
                for m in pgow:
                    g = self.L[m]
                    if g.name in B_GATES:
                        self.G.add_edge(m, n)
                        break
                    elif g.name in X_GATES or (g.name == "cx" and g.qargs[1] == b):
                        self.G.add_edge(m, n)
                        X_FLAG = True
                    elif g.name in Z_GATES or (g.name == "cx" and g.qargs[0] == b):
                        if X_FLAG:
                            break
                        else:
                            continue
                    else:
                        raise MapperError("Unknown gate: " + g.name)
            elif self.L[n].name == "cx":
                bc, bt = self.L[n].qargs
                [cpgow, tpgow] = self._prior_gates_on_wire(self.L, n)

                Z_FLAG = False
                for m in tpgow:  # target bit: bt
                    g = self.L[m]
                    if g.name in B_GATES:
                        self.G.add_edge(m, n)
                        break
                    elif g.name in X_GATES or (g.name == "cx" and g.qargs[1] == bt):
                        if Z_FLAG:
                            break
                        else:
                            continue
                    elif g.name in Z_GATES or (g.name == "cx" and g.qargs[0] == bt):
                        self.G.add_edge(m, n)
                        Z_FLAG = True
                    else:
                        raise MapperError("Unknown gate: " + g.name)

                X_FLAG = False
                for m in cpgow:  # control bit: bc
                    g = self.L[m]
                    if g.name in B_GATES:
                        self.G.add_edge(m, n)
                        break
                    elif g.name in X_GATES or (g.name == "cx" and g.qargs[1] == bc):
                        self.G.add_edge(m, n)
                        X_FLAG = True
                    elif g.name in Z_GATES or (g.name == "cx" and g.qargs[0] == bc):
                        if X_FLAG:
                            break
                        else:
                            continue
                    else:
                        raise MapperError("Unknown gate: " + g.name)
            elif self.L[n].name in B_GATES:
                for i, pgow in enumerate(self._prior_gates_on_wire(self.L, n)):
                    b = self._all_args(n)[i]
                    X_FLAG, Z_FLAG = False, False
                    for m in pgow:
                        g = self.L[m]
                        if g.name in B_GATES:
                            self.G.add_edge(m, n)
                            break
                        elif g.name in X_GATES or (g.name == "cx" and g.qargs[1] == b):
                            if Z_FLAG:
                                break
                            else:
                                self.G.add_edge(m, n)
                                X_FLAG = True
                        elif g.name in Z_GATES or (g.name == "cx" and g.qargs[0] == b):
                            if X_FLAG:
                                break
                            else:
                                self.G.add_edge(m, n)
                                Z_FLAG = True
                        else:
                            raise MapperError("Unknown gate: " + g.name)
            else:
                raise MapperError("Unknown gate: " + self.L[n].name)

    def _create_basic_graph(self):
        for n in self.G.nodes():
            for pgow in self._prior_gates_on_wire(self.L, n):
                m = next(pgow, -1)
                if m != -1:
                    self.G.add_edge(m, n)

    def _create_layer_graph(self):
        self._create_basic_graph()

        # construct CNOT layers
        layers = []
        wire = defaultdict(int)
        for n in self.G.nodes():
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
                    self.G.add_edge(icx, jcx)

    def n_nodes(self):
        return self.G.__len__()

    def qargs(self, i: int) -> List[Tuple[QuantumRegister, int]]:
        """Qubit arguments of the gate
        Args:
            i: Index of the gate in the quantum gate list
        Returns: List of qubit arguments
        """
        return self.L[i].qargs

    def _all_args(self, i: int) -> List[Tuple[QuantumRegister, int]]:
        """Qubit and classical-bit arguments of the gate
        Args:
            i: Index of the gate in the quantum gate list
        Returns: List of all arguments
        """
        return self.L[i].qargs + self.L[i].cargs

    def gate_name(self, i: int) -> str:
        """Name of the gate
        Args:
            i: Index of the gate in the quantum gate list
        Returns: Name of the gate
        """
        return self.L[i].name

    def head_gates(self) -> Set[int]:
        """Gates which can be applicable prior to the other gates
        Returns: Set of indices of the gates
        """
        return frozenset([n for n in self.G.nodes() if len(self.G.in_edges(n)) == 0])

    def gr_successors(self, i: int) -> List[int]:
        """Successor gates in Gr (transitive reduction) of the gate
        Args:
            i: Index of the gate in the quantum gate list
        Returns: Set of indices of the successor gates
        """
        return self.G.successors(i)

    def descendants(self, i: int) -> Set[int]:
        return nx.descendants(self.G, i)

    def ancestors(self, i: int) -> Set[int]:
        return nx.ancestors(self.G, i)

    def gate(self, g, layout, physical_qreg):
        gate = copy.deepcopy(self.L[g])
        for i, logical_qubit in enumerate(gate.qargs):
            if logical_qubit in layout.get_virtual_bits().keys():
                gate.qargs[i] = (physical_qreg, layout[logical_qubit])
            else:
                raise MapperError("logical_qubit must be in layout")
        return gate

    @staticmethod
    def remove_redundancy(G):
        """remove redundant edges in DAG (= change G to its transitive reduction)
        """
        edges = list(G.edges())
        for e in edges:
            G.remove_edge(e[0], e[1])
            if not nx.has_path(G, e[0], e[1]):
                G.add_edge(e[0], e[1])

    @staticmethod
    def _prior_gates_on_wire(L, toidx):
        res = []
        for qarg in L[toidx].qargs:
            gates = []
            for i, g in enumerate(L[:toidx]):
                if qarg in g.qargs:
                    gates.append(i)
            res.append(reversed(gates))
        for carg in L[toidx].cargs:
            gates = []
            for i, g in enumerate(L[:toidx]):
                if carg in g.cargs:
                    gates.append(i)
            res.append(reversed(gates))
        return res


if __name__ == '__main__':
    pass
