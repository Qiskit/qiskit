# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module contains common utils for disjoint coupling maps."""

from collections import defaultdict
from typing import List, Callable, TypeVar, Dict

import rustworkx as rx

from qiskit.circuit import Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

T = TypeVar("T")


def run_pass_over_connected_components(
    dag: DAGCircuit,
    coupling_map: CouplingMap,
    run_func: Callable[[DAGCircuit, CouplingMap], T],
    remove_barrier: bool = True,
) -> List[T]:
    """Run a transpiler pass inner function over mapped components."""
    cmap_components = coupling_map.connected_components()
    # If graph is connected we only need to run the pass once
    if len(cmap_components) == 1:
        return [run_func(dag, cmap_components[0])]
    dag_components = separate_dag(dag, remove_barrier)
    mapped_components = map_components(dag_components, cmap_components)
    out_component_pairs = []
    for cmap_index, dags in mapped_components.items():
        out_dag = dag_components[dags.pop()]
        for dag_index in dags:
            dag = dag_components[dag_index]
            out_dag.add_qubits(dag.qubits)
            out_dag.add_clbits(dag.clbits)
            out_dag.add_qreg(dag.qregs)
            out_dag.add_clbits(dag.cregs)
            out_dag.compose(dag, qubits=dag.qubits, clbits=dag.clbits)
        out_component_pairs.append((out_dag, cmap_components[cmap_index]))
    res = [run_func(out_dag, cmap) for out_dag, cmap in out_component_pairs]
    return res


def map_components(
    dag_components: List[DAGCircuit], cmap_components: List[CouplingMap]
) -> Dict[int, int]:
    """Map a list of circuit components to coupling map components."""
    free_qubits = {index: len(cmap.graph) for index, cmap in enumerate(cmap_components)}
    out_mapping = defaultdict(list)

    for dag_index, dag in enumerate(sorted(dag_components, key=lambda x: x.num_qubits())):
        for cmap_index in range(len(cmap_components)):
            # TODO: Improve heuristic to involve connectivity and estimate
            # swap cost
            if dag.num_qubits() <= free_qubits[cmap_index]:
                out_mapping[cmap_index].append(dag_index)
                free_qubits[cmap_index] -= dag.num_qubits()
                break
        else:
            raise TranspilerError(
                "A connected component of the DAGCircuit is too large for any of the connected "
                "components in the coupling map."
            )
    return out_mapping


def separate_dag(dag: DAGCircuit, barrier_before_measure=True) -> List[DAGCircuit]:
    """Separate a dag circuit into it's connected components."""
    # TODO: Fix support for barriers by splitting and reconsituting them
    # this will mean this function always lives as a separate entity
    # from DAGCircuit.seperable_circuits
    connected_components = rx.weakly_connected_components(dag._multi_graph)
    disconnected_subgraphs = []
    for components in connected_components:
        disconnected_subgraphs.append(dag._multi_graph.subgraph(list(components)))

    def _key(x):
        return x.sort_key

    decomposed_dags = []
    for subgraph in disconnected_subgraphs:
        new_dag = dag.copy_empty_like()
        new_dag.global_phase = 0
        subgraph_is_classical = True
        for node in rx.lexicographical_topological_sort(subgraph, key=_key):
            if isinstance(node, DAGInNode):
                if isinstance(node.wire, Qubit):
                    subgraph_is_classical = False
            if not isinstance(node, DAGOpNode):
                continue
            if barrier_before_measure and node.op.name == "barrier":
                continue
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            # Ignore DAGs created for empty clbits
            if not subgraph_is_classical:
                continue
        idle_qubits = []
        for qubit, node in new_dag.input_map.items():
            if isinstance(next(new_dag.successors(node)), DAGOutNode):
                idle_qubits.append(qubit)
        new_dag.remove_qubits(*idle_qubits)
        decomposed_dags.append(new_dag)

    return decomposed_dags
