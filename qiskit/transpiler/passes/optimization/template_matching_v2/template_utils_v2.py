# This code is part of Qiskit.
#
# (C) Copyright IBM 2020-2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Template matching in the forward direction, it takes an initial
match, a configuration of qubits and both circuit and template as inputs. The
result is a list of matches between the template and the circuit.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""

import rustworkx as rx


def get_node(dag, node_id):
    return dag._multi_graph[node_id]


def get_qindices(dag, node):
    return [dag.find_bit(qarg).index for qarg in node.qargs]


def get_cindices(dag, node):
    return [dag.find_bit(carg).index for carg in node.cargs]


def get_descendants(dag, node_id):
    return list(rx.descendants(dag._multi_graph, node_id))


def get_ancestors(dag, node_id):
    return list(rx.ancestors(dag._multi_graph, node_id))


def get_successors(dag, node_id):
    return [succ._node_id for succ in dag._multi_graph.successors(node_id)]


def get_predecessors(dag, node_id):
    return [pred._node_id for pred in dag._multi_graph.predecessors(node_id)]
