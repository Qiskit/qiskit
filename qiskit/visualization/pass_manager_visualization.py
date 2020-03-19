# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Visualization function for a pass manager. Passes are grouped based on their
flow controller, and coloured based on the type of pass.
"""
import os
import inspect
import tempfile

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from qiskit.visualization import utils
from qiskit.visualization.exceptions import VisualizationError
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass

DEFAULT_STYLE = {AnalysisPass: 'red',
                 TransformationPass: 'blue'}


def pass_manager_drawer(pass_manager, filename, style=None, raw=False):
    """
    Draws the pass manager.

    This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs
    Graphviz <https://www.graphviz.org/>` to be installed.

    Args:
        pass_manager (PassManager): the pass manager to be drawn
        filename (str): file path to save image to
        style (dict or OrderedDict): keys are the pass classes and the values are
            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered
            dict can be used to ensure a priority coloring when pass falls into multiple
            categories. Any values not included in the provided dict will be filled in from
            the default dict
        raw (Bool) : True if you want to save the raw Dot output not an image. The
            default is False.
    Returns:
        PIL.Image or None: an in-memory representation of the pass manager. Or None if
        no image was generated or PIL is not installed.
    Raises:
        ImportError: when nxpd or pydot not installed.
        VisualizationError: If raw=True and filename=None.

    Example:
        .. code-block::

             %matplotlib inline
            from qiskit import QuantumCircuit
            from qiskit.compiler import transpile
            from qiskit.transpiler import PassManager
            from qiskit.visualization import pass_manager_drawer
            from qiskit.transpiler.passes import Unroller

            circ = QuantumCircuit(3)
            circ.ccx(0, 1, 2)
            circ.draw()

            pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
            pm = PassManager(pass_)
            new_circ = pm.run(circ)
            new_circ.draw(output='mpl')

            pass_manager_drawer(pm, "passmanager.jpg")
    """

    try:
        import subprocess

        _PROC = subprocess.Popen(['dot', '-V'],  # pylint: disable=invalid-name
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        _PROC.communicate()
        if _PROC.returncode != 0:
            has_graphviz = False
        else:
            has_graphviz = True
    except Exception:  # pylint: disable=broad-except
        # this is raised when the dot command cannot be found, which means GraphViz
        # isn't installed
        has_graphviz = False

    HAS_GRAPHVIZ = has_graphviz  # pylint: disable=invalid-name

    try:
        import pydot
        if not HAS_GRAPHVIZ:
            raise ImportError
    except ImportError:
        raise ImportError("pass_manager_drawer requires pydot and graphviz. "
                          "Run 'pip install pydot'. "
                          "Graphviz can be installed using 'brew install graphviz' on Mac"
                          " or by downloading it from the website.")

    passes = pass_manager.passes()

    if not style:
        style = DEFAULT_STYLE

    # create the overall graph
    graph = pydot.Dot()

    # identifiers for nodes need to be unique, so assign an id
    # can't just use python's id in case the exact same pass was
    # appended more than once
    component_id = 0

    prev_node = None

    for index, controller_group in enumerate(passes):

        # label is the name of the flow controller parameter
        label = "[%s] %s" % (index, ', '.join(controller_group['flow_controllers']))

        # create the subgraph for this controller
        subgraph = pydot.Cluster(str(component_id), label=label, fontname='helvetica',
                                 labeljust='l')
        component_id += 1

        for pass_ in controller_group['passes']:

            # label is the name of the pass
            node = pydot.Node(str(component_id),
                              label=str(type(pass_).__name__),
                              color=_get_node_color(pass_, style),
                              shape="rectangle",
                              fontname='helvetica')

            subgraph.add_node(node)
            component_id += 1

            # the arguments that were provided to the pass when it was created
            arg_spec = inspect.getfullargspec(pass_.__init__)
            # 0 is the args, 1: to remove the self arg
            args = arg_spec[0][1:]

            num_optional = len(arg_spec[3]) if arg_spec[3] else 0

            # add in the inputs to the pass
            for arg_index, arg in enumerate(args):
                nd_style = 'solid'
                # any optional args are dashed
                # the num of optional counts from the end towards the start of the list
                if arg_index >= (len(args) - num_optional):
                    nd_style = 'dashed'

                input_node = pydot.Node(component_id, label=arg,
                                        color="black",
                                        shape="ellipse",
                                        fontsize=10,
                                        style=nd_style,
                                        fontname='helvetica')
                subgraph.add_node(input_node)
                component_id += 1
                subgraph.add_edge(pydot.Edge(input_node, node))

            # if there is a previous node, add an edge between them
            if prev_node:
                subgraph.add_edge(pydot.Edge(prev_node, node))

            prev_node = node

        graph.add_subgraph(subgraph)

    if raw:
        if filename:
            graph.write(filename, format='raw')
            return None
        else:
            raise VisualizationError("if format=raw, then a filename is required.")

    if not HAS_PIL and filename:
        # linter says this isn't a method - it is
        graph.write_png(filename)  # pylint: disable=no-member
        return None

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, 'pass_manager.png')

        # linter says this isn't a method - it is
        graph.write_png(tmppath)  # pylint: disable=no-member

        image = Image.open(tmppath)
        image = utils._trim(image)
        os.remove(tmppath)
        if filename:
            image.save(filename, 'PNG')
        return image


def _get_node_color(pss, style):
    # look in the user provided dict first
    for typ, color in style.items():
        if isinstance(pss, typ):
            return color

    # failing that, look in the default
    for typ, color in DEFAULT_STYLE.items():
        if isinstance(pss, typ):
            return color

    return "black"
