# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Visualization function for DAG circuit representation.
"""

import sys
import copy
try:
    import nxpd
    import pydot  # pylint: disable=unused-import
except ImportError:
    raise ImportError("dag_drawer requires nxpd, pydot, and Graphviz. "
                      "Run 'pip install nxpd pydot', and install graphviz")
from ._error import VisualizationError


def dag_drawer(dag, scale=0.7, filename=None, style='color'):
    """Plot the directed acyclic graph (dag) to represent operation dependencies
    in a quantum circuit.

    Note this function leverages
    `pydot <https://github.com/erocarrera/pydot>`_ (via
    `nxpd <https://github.com/chebee7i/nxpd`_) to generate the graph, which
    means that having `Graphviz <https://www.graphviz.org/>`_ installed on your
    system is required for this to work.

    Args:
        dag (DAGCircuit): The dag to draw.
        scale (float): scaling factor
        filename (str): file path to save image to (format inferred from name)
        style (str): 'plain': B&W graph
                     'color' (default): color input/output/op nodes

    Returns:
        Ipython.display.Image: if in Jupyter notebook and not saving to file,
            otherwise None.

    Raises:
        VisualizationError: when style is not recognized.
    """
    G = copy.deepcopy(dag.multi_graph)  # don't modify the original graph attributes
    G.graph['dpi'] = 100 * scale

    if style == 'plain':
        pass
    elif style == 'color':
        for node in G.nodes:
            n = G.nodes[node]
            if n['type'] == 'op':
                n['label'] = n['name']
                n['color'] = 'blue'
                n['style'] = 'filled'
                n['fillcolor'] = 'lightblue'
            if n['type'] == 'in':
                n['label'] = n['name']
                n['color'] = 'black'
                n['style'] = 'filled'
                n['fillcolor'] = 'green'
            if n['type'] == 'out':
                n['label'] = n['name']
                n['color'] = 'black'
                n['style'] = 'filled'
                n['fillcolor'] = 'red'
        for e in G.edges(data=True):
            e[2]['label'] = e[2]['name']
    else:
        raise VisualizationError("Unrecognized style for the dag_drawer.")

    show = nxpd.nxpdParams['show']
    if filename:
        show = False
    elif ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
        show = 'ipynb'
    else:
        show = True

    return nxpd.draw(G, filename=filename, show=show)
