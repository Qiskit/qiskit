# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Visualization function for DAG circuit representation.
"""

import sys
from .exceptions import VisualizationError


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
        ImportError: when nxpd or pydot not installed.

    Example:
        .. code-block::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag
            from qiskit.visualization import dag_drawer

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)

            dag = circuit_to_dag(circ)
            dag_drawer(dag)
    """
    try:
        import nxpd
        import pydot  # pylint: disable=unused-import
    except ImportError:
        raise ImportError("dag_drawer requires nxpd and pydot. "
                          "Run 'pip install nxpd pydot'.")

    G = dag.to_networkx()
    G.graph['dpi'] = 100 * scale

    if style == 'plain':
        pass
    elif style == 'color':
        for node in G.nodes:
            n = G.nodes[node]
            n['label'] = node.name
            if node.type == 'op':
                n['color'] = 'blue'
                n['style'] = 'filled'
                n['fillcolor'] = 'lightblue'
            if node.type == 'in':
                n['color'] = 'black'
                n['style'] = 'filled'
                n['fillcolor'] = 'green'
            if node.type == 'out':
                n['color'] = 'black'
                n['style'] = 'filled'
                n['fillcolor'] = 'red'
        for e in G.edges(data=True):
            e[2]['label'] = e[2]['name']
    else:
        raise VisualizationError("Unrecognized style for the dag_drawer.")

    if filename:
        show = False
    elif ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
        show = 'ipynb'
    else:
        show = True

    try:
        return nxpd.draw(G, filename=filename, show=show)
    except nxpd.pydot.InvocationException:
        raise VisualizationError("dag_drawer requires GraphViz installed in the system. "
                                 "Check https://www.graphviz.org/download/ for details on "
                                 "how to install GraphViz in your system.")
