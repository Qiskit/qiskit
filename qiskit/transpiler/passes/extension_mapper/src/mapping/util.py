"""Utility functions related to the mapping of circuits to architectures."""
import pprint
from typing import Mapping, TypeVar

import networkx as nx
import sympy
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister

from .. import util
from ..permutation import Permutation

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')


def new_mapping(current_mapping: Mapping[Reg, ArchNode],
                permutation: Permutation[ArchNode]) -> Mapping[Reg, ArchNode]:
    """Construct a new mapping from the current mapping and a permutation that is applied to it."""
    return {
        qubit: permutation[mapped_to]
        for qubit, mapped_to in current_mapping.items()
        }


# Construct an inverted CNOT circuit.
_CNOT_CIRCUIT = DAGCircuit()
_CNOT_CIRCUIT.add_basis_element('u2', number_qubits=1, number_parameters=2)
_CNOT_CIRCUIT.add_basis_element('cx', 2)
_CNOT_QREG = QuantumRegister(2)
_CNOT_CIRCUIT.add_qreg(_CNOT_QREG)
_CNOT_CIRCUIT.apply_operation_back("u2", [(_CNOT_QREG.name, 0)], params=[sympy.N(0), sympy.pi])
_CNOT_CIRCUIT.apply_operation_back("u2", [(_CNOT_QREG.name, 1)], params=[sympy.N(0), sympy.pi])
_CNOT_CIRCUIT.apply_operation_back("cx", [(_CNOT_QREG.name, 1), (_CNOT_QREG.name, 0)])
_CNOT_CIRCUIT.apply_operation_back("u2", [(_CNOT_QREG.name, 0)], params=[sympy.N(0), sympy.pi])
_CNOT_CIRCUIT.apply_operation_back("u2", [(_CNOT_QREG.name, 1)], params=[sympy.N(0), sympy.pi])


def direction_mapper(dagcircuit: DAGCircuit,
                     mapping: Mapping[Reg, ArchNode],
                     coupling_graph: nx.DiGraph) -> int:
    """Replaces CNOTs in the wrong direction by corrected CNOTs.

    Very similar to qiskit.mapper.direction_mapper but does not assume node names in graph
    and it also reuses a singleton CNOT circuit description."""
    cx_nodes = list(util.dagcircuit_get_named_nodes(dagcircuit, "cx"))
    flips = 0
    for cx_node in cx_nodes:
        node = dagcircuit.multi_graph.node[cx_node]
        # The qargs second argument is always 0 since a coupling_graph is one qubit per node.
        cx_edge = tuple(mapping[qarg] for qarg in node["qargs"])
        if cx_edge in coupling_graph.edges:
            continue
        elif (cx_edge[1], cx_edge[0]) in coupling_graph.edges:
            dagcircuit.substitute_circuit_one(cx_node,
                                              _CNOT_CIRCUIT,
                                              wires=[(_CNOT_QREG.name, 0), (_CNOT_QREG.name, 1)])
            flips += 1
        else:
            raise RuntimeError("circuit incompatible with CouplingGraph: "
                               "cx on %s" % pprint.pformat(cx_edge))
    return flips
