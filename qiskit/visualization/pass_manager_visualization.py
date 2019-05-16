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

import pydot

from qiskit.transpiler import AnalysisPass, TransformationPass
DEFAULT_STYLE = {AnalysisPass: 'red',
                 TransformationPass: 'blue'}


# DEFAULT_STYLE is considered to be a dangerous default, as it could be modified
# It is never modified, so it is ok to use as the default value
# pylint: disable=dangerous-default-value
def pass_manager_drawer(pass_manager, filename=None, style=DEFAULT_STYLE):
    """
    Draws the pass manager

    Args:
        pass_manager (PassManager): the pass manager to be drawn
        filename (str): file path to save image to
        style (dict or OrderedDict): keys are the pass classes and the values are
            the colors to make them. An ordered dict can be used to ensure a priority
            coloring when pass falls into multiple categories. Any values not included
            in the dict will be filled in from the default dict
    """

    passes = pass_manager.passes()

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
                            color=_get_node_color(pss, style))

            subgraph.add_node(nd)

            # if there is a previous node, add an edge between them
            if prev_nd:
                subgraph.add_edge(pydot.Edge(prev_nd, nd))

            prev_nd = nd
            node_id += 1

        graph.add_subgraph(subgraph)

    if filename:
        # linter says this isn't a method - it is
        graph.write_png(filename) # pylint: disable=no-member


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
