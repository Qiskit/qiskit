# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2020.
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

import os
import sys
import tempfile

from networkx.drawing.nx_pydot import to_pydot

from .exceptions import VisualizationError

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def dag_drawer(dag, scale=0.7, filename=None, style='color', category=None):
    """Plot the directed acyclic graph (dag) to represent operation dependencies
    in a quantum circuit.

    Note this function leverages
    `pydot <https://github.com/erocarrera/pydot>`_ to generate the graph, which
    means that having `Graphviz <https://www.graphviz.org/>`_ installed on your
    system is required for this to work.

    The current release of Graphviz can be downloaded here: <https://graphviz.gitlab.io/download/>.
    Download the version of the sotware that matches your environment and follow the instructions to
    install Graph Visualization Software (Graphviz) on your operating system.

    Args:
        dag (DAGCircuit): The dag to draw.
        scale (float): scaling factor
        filename (str): file path to save image to (format inferred from name)
        style (str): 'plain': B&W graph
                     'color' (default): color input/output/op nodes
        category (str): 'dependency' for drawing DAG dependency

    Returns:
        PIL.Image: if in Jupyter notebook and not saving to file,
            otherwise None.

    Raises:
        VisualizationError: when style is not recognized.
        ImportError: when pydot or pillow are not installed.

    Example:
        .. jupyter-execute::

            %matplotlib inline
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
        import pydot  # pylint: disable=unused-import
    except ImportError:
        raise ImportError("dag_drawer requires pydot. "
                          "Run 'pip install pydot'.")
    if category is None:
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

    elif category == 'dependency':
        G = dag.to_networkx()
        G.graph['dpi'] = 100 * scale

        if style == 'plain':
            pass
        elif style == 'color':
            for node in G.nodes:
                n = G.nodes[node]
                n['label'] = str(node.node_id) + ': ' + node.name
                if node.name == 'measure':
                    n['color'] = 'blue'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'lightblue'
                if node.name == 'barrier':
                    n['color'] = 'black'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'green'
                if node.name == 'snapshot':
                    n['color'] = 'black'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'red'
                if node.condition:
                    n['label'] = str(node.node_id) + ': ' + node.name + ' (conditional)'
                    n['color'] = 'black'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'lightgreen'
        else:
            raise VisualizationError("Unrecognized style for the dag_drawer.")
    else:
        raise VisualizationError("Unrecognized category of DAG")

    dot = to_pydot(G)

    if filename:
        extension = filename.split('.')[-1]
        dot.write(filename, format=extension)
        return None
    elif ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
        if not HAS_PIL:
            raise ImportError(
                "dag_drawer requires pillow for drawing in jupyter directly. "
                "Run 'pip install pillow'.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, 'dag.png')
            dot.write_png(tmp_path)
            image = Image.open(tmp_path)
            os.remove(tmp_path)
            return image
    else:
        if not HAS_PIL:
            raise ImportError(
                "dag_drawer requires pillow for drawing to a window directly. "
                "Run 'pip install pillow'.")
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, 'dag.png')
            dot.write_png(tmp_path)
            image = Image.open(tmp_path)
            image.show()
            os.remove(tmp_path)
            return None
