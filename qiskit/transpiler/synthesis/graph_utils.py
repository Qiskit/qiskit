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
