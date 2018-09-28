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
import nxpd
from ._error import VisualizationError


def dag_drawer(dag, scale=0.7, filename=None, style='color'):
    """Plot the directed acyclic graph (dag) to represent operation dependencies
    in a quantum circuit.

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
        for n in G.nodes:
            G.nodes[n]['label'] = str(G.nodes[n]['name'])
            if G.nodes[n]['type'] == 'op':
                G.nodes[n]['color'] = 'blue'
                G.nodes[n]['style'] = 'filled'
                G.nodes[n]['fillcolor'] = 'lightblue'
            if G.nodes[n]['type'] == 'in':
                G.nodes[n]['color'] = 'black'
                G.nodes[n]['style'] = 'filled'
                G.nodes[n]['fillcolor'] = 'green'
            if G.nodes[n]['type'] == 'out':
                G.nodes[n]['color'] = 'black'
                G.nodes[n]['style'] = 'filled'
                G.nodes[n]['fillcolor'] = 'red'
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
