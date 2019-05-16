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
from qiskit.transpiler import AnalysisPass, TransformationPass
DEFAULT_STYLE = {AnalysisPass: 'red',
                 TransformationPass: 'blue'}


# DEFAULT_STYLE is considered to be a dangerous default, as it could be modified
# It is never modified, so it is ok to use as the default value
# pylint: disable=dangerous-default-value
def pass_manager_drawer(pass_manager, filename=None, style=DEFAULT_STYLE):
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
    except ImportError:
        raise ImportError("pass_manager_drawer requires pydot. "
                          "Run 'pip install pydot'.")

    passes = pass_manager.passes()

    if not style:
        style = DEFAULT_STYLE

    # create the overall graph
    graph = pydot.Dot()

    # identifiers for nodes need to be unique, so assign an id
    # can't just use python's id in case the exact same pass was
    # appended more than once
    node_id = 0

    prev_nd = None

    for pass_group in passes:

        # label is the name of the flow controller (without the word controller)
        label = pass_group['type'].__name__.replace('Controller', '')
        # create the subgraph
        subgraph = pydot.Cluster(str(id(pass_group)), label=label)

        for pss in pass_group['passes']:

            # label is the name of the pass
            nd = pydot.Node(str(node_id), label=str(type(pss).__name__),
                            color=_get_node_color(pss, style),
                            shape="rectangle")

            subgraph.add_node(nd)
            node_id += 1

            # the arguments that were provided to the pass when it was created
            arg_spec = inspect.getfullargspec(pss.__init__)
            # 0 is the args, 1: to remove the self arg
            args = arg_spec[0][1:]
            num_defaults = len(arg_spec[3]) if arg_spec[3] else 0

            for arg_index, arg in enumerate(args):
                nd_style = 'solid'
                # any optional args are dashed
                if arg_index >= (len(args) - num_defaults):
                    nd_style = 'dashed'

                input_nd = pydot.Node(node_id, label=arg,
                                      color="black",
                                      shape="ellipse",
                                      fontsize=10,
                                      style=nd_style)
                subgraph.add_node(input_nd)
                node_id += 1
                subgraph.add_edge(pydot.Edge(input_nd, nd))

            # if there is a previous node, add an edge between them
            if prev_nd:
                subgraph.add_edge(pydot.Edge(prev_nd, nd))

            prev_nd = nd

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
