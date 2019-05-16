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
import pydot


def pass_manager_drawer(passes, filename=None, title=None):

    # create the overall graph
    graph = pydot.Dot(comment=title)

    # identifiers for nodes need to be unique, so assign an id
    # can't just use python's id in case the exact same pass was
    # appended more than once
    node_id = 0

    prev_nd = None

    for pass_group in passes:

        # label is the name of the flow controller
        label = pass_group['type'].__name__
        # create the subgraph
        subgraph = pydot.Cluster(str(id(pass_group)), label=label)

        for pss in pass_group['passes']:

            # label is the name of the pass
            nd = pydot.Node(str(node_id), label=str(type(pss).__name__))

            subgraph.add_node(nd)

            # if there is a previous node, add an edge between them
            if prev_nd:
                subgraph.add_edge(pydot.Edge(prev_nd, nd))

            prev_nd = nd
            node_id += 1

        graph.add_subgraph(subgraph)

    if filename:
        graph.write_png(filename)
