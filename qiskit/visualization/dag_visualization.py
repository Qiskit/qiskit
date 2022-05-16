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

from qiskit.dagcircuit.dagnode import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit import Qubit
from qiskit.utils import optionals as _optionals
from qiskit.exceptions import InvalidFileError
from .exceptions import VisualizationError

FILENAME_EXTENSIONS = {
    "bmp",
    "canon",
    "cgimage",
    "cmap",
    "cmapx",
    "cmapx_np",
    "dot",
    "dot_json",
    "eps",
    "exr",
    "fig",
    "gd",
    "gd2",
    "gif",
    "gv",
    "icns",
    "ico",
    "imap",
    "imap_np",
    "ismap",
    "jp2",
    "jpe",
    "jpeg",
    "jpg",
    "json",
    "json0",
    "mp",
    "pct",
    "pdf",
    "pic",
    "pict",
    "plain",
    "plain-ext",
    "png",
    "pov",
    "ps",
    "ps2",
    "psd",
    "sgi",
    "svg",
    "svgz",
    "tga",
    "tif",
    "tiff",
    "tk",
    "vdx",
    "vml",
    "vmlz",
    "vrml",
    "wbmp",
    "webp",
    "xdot",
    "xdot1.2",
    "xdot1.4",
    "xdot_json",
}


@_optionals.HAS_PYDOT.require_in_call
def dag_drawer(dag, scale=0.7, filename=None, style="color"):
    """Plot the directed acyclic graph (dag) to represent operation dependencies
    in a quantum circuit.

    Note this function leverages
    `pydot <https://github.com/erocarrera/pydot>`_ to generate the graph, which
    means that having `Graphviz <https://www.graphviz.org/>`_ installed on your
    system is required for this to work.

    The current release of Graphviz can be downloaded here: <https://graphviz.gitlab.io/download/>.
    Download the version of the software that matches your environment and follow the instructions
    to install Graph Visualization Software (Graphviz) on your operating system.

    Args:
        dag (DAGCircuit): The dag to draw.
        scale (float): scaling factor
        filename (str): file path to save image to (format inferred from name)
        style (str): 'plain': B&W graph
                     'color' (default): color input/output/op nodes

    Returns:
        PIL.Image: if in Jupyter notebook and not saving to file,
            otherwise None.

    Raises:
        VisualizationError: when style is not recognized.
        MissingOptionalLibraryError: when pydot or pillow are not installed.
        InvalidFileError: when filename provided is not valid

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
    import pydot

    # NOTE: use type str checking to avoid potential cyclical import
    # the two tradeoffs ere that it will not handle subclasses and it is
    # slower (which doesn't matter for a visualization function)
    type_str = str(type(dag))
    if "DAGDependency" in type_str:
        graph_attrs = {"dpi": str(100 * scale)}

        def node_attr_func(node):
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                n["label"] = str(node.node_id) + ": " + str(node.name)
                if node.name == "measure":
                    n["color"] = "blue"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightblue"
                if node.name == "barrier":
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                if node.op._directive:
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                if node.op.condition:
                    n["label"] = str(node.node_id) + ": " + str(node.name) + " (conditional)"
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightgreen"
                return n
            else:
                raise VisualizationError("Unrecognized style %s for the dag_drawer." % style)

        edge_attr_func = None

    else:
        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        clbit_indices = {bit: index for index, bit in enumerate(dag.clbits)}
        register_bit_labels = {
            bit: f"{reg.name}[{idx}]"
            for reg in list(dag.qregs.values()) + list(dag.cregs.values())
            for (idx, bit) in enumerate(reg)
        }

        graph_attrs = {"dpi": str(100 * scale)}

        def node_attr_func(node):
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                if isinstance(node, DAGOpNode):
                    n["label"] = node.name
                    n["color"] = "blue"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightblue"
                if isinstance(node, DAGInNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(node.wire, f"q_{qubit_indices[node.wire]}")
                    else:
                        label = register_bit_labels.get(node.wire, f"c_{clbit_indices[node.wire]}")
                    n["label"] = label
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                if isinstance(node, DAGOutNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(node.wire, f"q[{qubit_indices[node.wire]}]")
                    else:
                        label = register_bit_labels.get(node.wire, f"c[{clbit_indices[node.wire]}]")
                    n["label"] = label
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                return n
            else:
                raise VisualizationError("Invalid style %s" % style)

        def edge_attr_func(edge):
            e = {}
            if isinstance(edge, Qubit):
                label = register_bit_labels.get(edge, f"q_{qubit_indices[edge]}")
            else:
                label = register_bit_labels.get(edge, f"c_{clbit_indices[edge]}")
            e["label"] = label
            return e

    dot_str = dag._multi_graph.to_dot(node_attr_func, edge_attr_func, graph_attrs)
    dot = pydot.graph_from_dot_data(dot_str)[0]

    if filename:
        if "." not in filename:
            raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
        extension = filename.split(".")[-1]
        if extension not in FILENAME_EXTENSIONS:
            raise InvalidFileError(
                "Filename extension must be one of: " + " ".join(FILENAME_EXTENSIONS)
            )
        dot.write(filename, format=extension)
        return None
    elif ("ipykernel" in sys.modules) and ("spyder" not in sys.modules):
        _optionals.HAS_PIL.require_now("dag_drawer")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag.png")
            dot.write_png(tmp_path)
            with Image.open(tmp_path) as test_image:
                image = test_image.copy()
            os.remove(tmp_path)
            return image
    else:
        _optionals.HAS_PIL.require_now("dag_drawer")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag.png")
            dot.write_png(tmp_path)
            image = Image.open(tmp_path)
            image.show()
            os.remove(tmp_path)
            return None
