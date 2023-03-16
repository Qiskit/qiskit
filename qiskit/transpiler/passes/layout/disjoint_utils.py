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
import uuid

import rustworkx as rx

from qiskit.circuit import Qubit, Barrier
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

T = TypeVar("T")


def run_pass_over_connected_components(
    dag: DAGCircuit,
    coupling_map: CouplingMap,
    run_func: Callable[[DAGCircuit, CouplingMap], T],
) -> List[T]:
    """Run a transpiler pass inner function over mapped components."""
    cmap_components = coupling_map.connected_components()
    # If graph is connected we only need to run the pass once
    if len(cmap_components) == 1:
        return [run_func(dag, cmap_components[0])]
    dag_components = separate_dag(dag)
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


def split_barriers(dag):
    """Mutate an input dag to split barriers into single qubit barriers."""
    for node in dag.op_nodes(Barrier):
        num_qubits = len(node.qargs)
        if num_qubits == 1:
            continue
        barrier_uuid = uuid.uuid4()
        split_dag = DAGCircuit()
        split_dag.add_qubits([Qubit() for _ in range(num_qubits)])
        for i in range(num_qubits):
            split_dag.apply_operation_back(
                Barrier(1, label=barrier_uuid), qargs=[split_dag.qubits[i]]
            )
        dag.substitute_node_with_dag(node, split_dag)


def combine_barriers(dag, retain_uuid=True):
    """Mutate input dag to combine barriers with UUID labels into a single barrier."""
    qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
    uuid_map = {}
    for node in dag.op_nodes(Barrier):
        if isinstance(node.op.label, uuid.UUID):
            barrier_uuid = node.op.label
            if barrier_uuid in uuid_map:
                other_node = uuid_map[node.op.label]
                num_qubits = len(other_node.qargs) + len(node.qargs)
                new_op = Barrier(num_qubits, label=barrier_uuid)
                new_node = dag.replace_block_with_op([node, other_node], new_op, qubit_indices)
                uuid_map[barrier_uuid] = new_node
            else:
                uuid_map[barrier_uuid] = node
    if not retain_uuid:
        for node in dag.op_nodes(Barrier):
            if isinstance(node.op.label, uuid.UUID):
                node.op.label = None


def separate_dag(dag: DAGCircuit) -> List[DAGCircuit]:
    """Separate a dag circuit into it's connected components."""
    # Split barriers into single qubit barrieries before connected components
    split_barriers(dag)
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
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            # Ignore DAGs created for empty clbits
            if not subgraph_is_classical:
                continue
        idle_qubits = []
        idle_clbits = []
        for bit, node in new_dag.input_map.items():
            succ_node = next(new_dag.successors(node))
            if isinstance(succ_node, DAGOutNode):
                if isinstance(succ_node.wire, Qubit):
                    idle_qubits.append(bit)
                else:
                    idle_clbits.append(bit)
        new_dag.remove_qubits(*idle_qubits)
        new_dag.remove_clbits(*idle_clbits)
        combine_barriers(new_dag)
        decomposed_dags.append(new_dag)

    return decomposed_dags
