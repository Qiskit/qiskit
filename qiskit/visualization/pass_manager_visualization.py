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
import inspect
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
DEFAULT_STYLE = {AnalysisPass: 'red',
                 TransformationPass: 'blue'}


try:
    import subprocess
    print(subprocess.check_output(['dot', '-V']))
    HAS_GRAPHVIZ = True
except FileNotFoundError:
    # this is raised when the dot command cannot be found, which means GraphViz
    # isn't installed
    HAS_GRAPHVIZ = False


def pass_manager_drawer(pass_manager, filename, style=None):
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

    Raises:
        ImportError: when nxpd or pydot not installed.
    """

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
    node_id = 0

    prev_node = None

    for controller_group in passes:

        # label is the name of the flow controller (without the word controller)
        label = controller_group['type'].__name__.replace('Controller', '')

        # create the subgraph for this controller
        subgraph = pydot.Cluster(str(id(controller_group)), label=label)

        for pss in controller_group['passes']:

            # label is the name of the pass
            node = pydot.Node(str(node_id),
                              label=str(type(pss).__name__),
                              color=_get_node_color(pss, style),
                              shape="rectangle")

            subgraph.add_node(node)
            node_id += 1

            # the arguments that were provided to the pass when it was created
            arg_spec = inspect.getfullargspec(pss.__init__)
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

                input_node = pydot.Node(node_id, label=arg,
                                        color="black",
                                        shape="ellipse",
                                        fontsize=10,
                                        style=nd_style)
                subgraph.add_node(input_node)
                node_id += 1
                subgraph.add_edge(pydot.Edge(input_node, node))

            # if there is a previous node, add an edge between them
            if prev_node:
                subgraph.add_edge(pydot.Edge(prev_node, node))

            prev_node = node

        graph.add_subgraph(subgraph)

    if filename:
        # linter says this isn't a method - it is
        graph.write_png(filename)  # pylint: disable=no-member


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
