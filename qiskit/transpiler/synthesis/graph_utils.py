# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper functions for hadling graphs"""

import numpy as np
import retworkx as rx

from qiskit.transpiler import CouplingMap


def pydigraph_to_pygraph(pydigraph: rx.PyDiGraph) -> rx.PyGraph:
    """Changes directed Pydigraph into an undirected Pygraph"""
    return pydigraph.to_undirected()


def noncutting_vertices(coupling_map: CouplingMap) -> np.ndarray:
    """Extracts noncutting vertices from a given coupling map. Direction is not taken into account.

    Args:
        coupling_map (CouplingMap): topology

    Returns:
        np.ndarray: array of non-cutting node indices
    """
    pygraph = pydigraph_to_pygraph(coupling_map.graph)
    cutting_vertices = rx.articulation_points(pygraph)
    vertices = set(pygraph.node_indices())
    noncutting = np.array(list(vertices - cutting_vertices))

    return noncutting


def postorder_traversal(tree: rx.PyGraph, node: int, edges: list, parent: int = None):
    """Traverse the given tree in postorder. Traversed edges are saved as tuples.
    The first element is the parent and second the child.
    Children are visited in increasing order.

    Args:
        tree (rx.PyGraph): tree to traverse
        node (int): root node
        edges (list): edge list
        parent (int, optional): parent node. Defaults to None.
    """
    if node == None:
        return
    for n in sorted(tree.neighbors(node)):
        if n == parent:
            continue
        postorder_traversal(tree, n, edges, node)
    if parent != None:
        edges.append((parent, node))


def preorder_traversal(tree: rx.PyGraph, node: int, edges: list, parent: int = None):
    """Preorder traversal of the edges of the given tree. Traversed edges are saved as tuples,
    where the first element is the parent and second the child. Children are visited in
    increasing order.

    Args:
        tree (rx.PyGraph): tree to traverse
        node (int): root node
        edges (list): list of edges
        parent (int, optional): parent node. Defaults to None.
    """
    if node == None:
        return
    if parent != None:
        edges.append((parent, node))
    for n in sorted(tree.neighbors(node)):
        if n == parent:
            continue
        preorder_traversal(tree, n, edges, node)
