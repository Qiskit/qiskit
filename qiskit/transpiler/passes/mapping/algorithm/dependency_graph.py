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
A dependency graph represents precedence relations of the gates in a quantum circuit considering
commutation rules. Each node represents gates in the circuit. Each directed edge represents
dependency of two gates. For example, gate g1 must be applied before gate g2 if and only if
there exists a path from g1 to g2. In this file, we use the term `gate` instead of `operation`
(or `instruction`) with or without its qu/cl-bit arguments.
"""
import copy
from collections import namedtuple, defaultdict
from typing import List, FrozenSet

import networkx as nx
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

InstructionContext = namedtuple("InstructionContext", "op qargs cargs")


class ArgumentedGate(InstructionContext):
    """
    A wrapper class of `InstructionContext` (operation with args context).
    """

    @property
    def name(self) -> str:
        """Name of this instruction."""
        return self.op.name


class DependencyGraph:
    """
    Create a dependency graph of a quantum circuit (say `qc`) with a chosen commutation rule.
    All of its nodes (i.e. gates) are considered as integers, which are the indices of `qc.data`.
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
                - "xz_commute": consider four commutation rules:
                    Rz-CX(control), Rx-CX(target), CX-CX(controls), CX-CX(targets).
                - "basic": consider only the commutation between gates without sharing qubits.

        Raises:
            TranspilerError: if `graph_type` is not one of the types listed above.
        """

        self._gates = [ArgumentedGate(instr, qrags, cargs)
                       for instr, qrags, cargs in quantum_circuit.data]

        self._graph = nx.DiGraph()  # dependency graph (including 1-qubit gates)

        for i, _ in enumerate(self._gates):
            self._graph.add_node(i)

        self.qubits = set()
        for i, _ in enumerate(self._gates):
            self.qubits.update(self.qargs(i))

        # for speed up of _prior_gates_on_wire
        self._create_gates_by_qubit()

        if graph_type == "xz_commute":
            self._create_xz_graph()
        elif graph_type == "basic":
            self._create_basic_graph()
        else:
            raise TranspilerError("Unknown graph_type:" + graph_type)

        # remove redundant edges in dependency graph
        self.remove_redundancy(self._graph)

    def _create_xz_graph(self):
        z_gates = ["u1", "rz", "s", "t", "z", "sdg", "tdg"]
        x_gates = ["rx", "x"]
        b_gates = ["u3", "h", "u2", "ry", "swap", "y", "barrier", "measure", "reset", "id"]
        # construct commutation-rules-aware dependency graph
        for n in self._graph.nodes():
            if self._gates[n].name in x_gates:
                b = self._gates[n].qargs[0]  # acting qubit of gate n
                # pylint: disable=unbalanced-tuple-unpacking
                [pgow] = self._prior_gates_on_sharing_qubits_of(n)
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
                b = self._gates[n].qargs[0]  # acting qubit of gate n
                # pylint: disable=unbalanced-tuple-unpacking
                [pgow] = self._prior_gates_on_sharing_qubits_of(n)
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
                # pylint: disable=unbalanced-tuple-unpacking
                [cpgow, tpgow] = self._prior_gates_on_sharing_qubits_of(n)

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
                all_args_of_n = self._gates[n].qargs + self._gates[n].cargs
                for i, pgow in enumerate(self._prior_gates_on_sharing_qubits_of(n)):
                    b = all_args_of_n[i]
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
            for pgow in self._prior_gates_on_sharing_qubits_of(n):
                m = next(pgow, -1)
                if m != -1:
                    self._graph.add_edge(m, n)

    def n_nodes(self) -> int:
        """Number of the nodes
        Returns:
            Number of the nodes in this graph.
        """
        return self._graph.__len__()

    def qargs(self, i: int) -> List[Qubit]:
        """Qubit arguments of the gate
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            List of qubit arguments.
        """
        return self._gates[i].qargs

    def gate_name(self, i: int) -> str:
        """Name of the gate
        Args:
            i: Index of the gate in the `self._gates`
        Returns:
            Name of the gate.
        """
        return self._gates[i].name

    def head_gates(self) -> FrozenSet[int]:
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

    def gate(self, gidx: int, layout: Layout, physical_qreg: QuantumRegister) -> InstructionContext:
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
                gate.qargs[i] = Qubit(physical_qreg, layout[virtual_qubit])
            else:
                raise TranspilerError("virtual_qubit must be in layout")
        return gate

    def nx_graph(self) -> nx.DiGraph:
        """Return deep copied networkx graph of this dependency graph.
        """
        return copy.deepcopy(self._graph)

    @staticmethod
    def remove_redundancy(graph):
        """Remove redundant edges in DAG (= change `graph` to its transitive reduction)
        """
        edges = list(graph.edges())
        for edge in edges:
            graph.remove_edge(edge[0], edge[1])
            if not nx.has_path(graph, edge[0], edge[1]):
                graph.add_edge(edge[0], edge[1])

    def _prior_gates_on_sharing_qubits_of(self, toidx):
        res = []
        for qarg in self._gates[toidx].qargs:
            res.append(reversed([i for i in self._gates_by_qubit[qarg] if i < toidx]))
        for carg in self._gates[toidx].cargs:
            res.append(reversed([i for i in self._gates_by_qubit[carg] if i < toidx]))
        return res

    def _create_gates_by_qubit(self):
        self._gates_by_qubit = defaultdict(list)
        for i, gate in enumerate(self._gates):
            for qarg in gate.qargs:
                self._gates_by_qubit[qarg].append(i)
            for carg in gate.cargs:
                self._gates_by_qubit[carg].append(i)
