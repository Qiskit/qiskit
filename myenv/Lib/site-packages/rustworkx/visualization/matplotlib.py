# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# NetworkX is distributed with the 3-clause BSD license.
#
#   Copyright (C) 2004-2020, NetworkX Developers
#   Aric Hagberg <hagberg@lanl.gov>
#   Dan Schult <dschult@colgate.edu>
#   Pieter Swart <swart@lanl.gov>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of the NetworkX Developers nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This code is forked from networkx's networkx_pylab.py module and adapted to
# work with rustworkx instead. The original source can be found at:
#
# https://github.com/networkx/networkx/blob/80b1afa2ae50314a8312998c214a8c1a356adcf1/networkx/drawing/nx_pylab.py

"""Draw a rustworkx graph with matplotlib."""

from collections.abc import Iterable
from itertools import islice, cycle
from numbers import Number

import numpy as np

import rustworkx


__all__ = [
    "mpl_draw",
]


def mpl_draw(graph, pos=None, ax=None, arrows=True, with_labels=False, **kwds):
    r"""Draw a graph with Matplotlib.

    .. note::

        Matplotlib is an optional dependency and will not be installed with
        rustworkx by default. If you intend to use this function make sure that
        you install matplotlib with either ``pip install matplotlib`` or
        ``pip install 'rustworkx[mpl]'``

    :param graph: A rustworkx graph, either a :class:`~rustworkx.PyGraph` or a
        :class:`~rustworkx.PyDiGraph`.
    :param dict pos: An optional dictionary (or
        a :class:`~rustworkx.Pos2DMapping` object) with nodes as keys and
        positions as values. If not specified a spring layout positioning will
        be computed. See `layout_functions` for functions that compute
        node positions.
    :param matplotlib.Axes ax: An optional Matplotlib Axes object to draw the
        graph in.
    :param bool arrows: For :class:`~rustworkx.PyDiGraph` objects if ``True``
        draw arrowheads. (defaults to ``True``) Note, that the Arrows will
        be the same color as edges.
    :param str arrowstyle: An optional string for directed graphs to choose
        the style of the arrowsheads. See
        :class:`matplotlib.patches.ArrowStyle` for more options. By default the
        value is set to ``'-\|>'``.
    :param int arrow_size: For directed graphs, choose the size of the arrow
        head's length and width. See
        :class:`matplotlib.patches.FancyArrowPatch` attribute and constructor
        kwarg ``mutation_scale`` for more info. Defaults to 10.
    :param bool with_labels: Set to ``True`` to draw labels on the nodes. Edge
        labels will only be drawn if the ``edge_labels`` parameter is set to a
        function. Defaults to ``False``.
    :param list node_list: An optional list of node indices in the graph to
        draw. If not specified all nodes will be drawn.
    :param list edge_list: An option list of edges in the graph to draw. If not
        specified all edges will be drawn
    :param int|list node_size: Optional size of nodes. If an array is
        specified it must be the same length as node_list. Defaults to 300
    :param node_color: Optional node color. Can be a single color or
        a sequence of colors with the same length as node_list. Color can be
        string or rgb (or rgba) tuple of floats from 0-1. If numeric values
        are specified they will be mapped to colors using the ``cmap`` and
        ``vmin``,``vmax`` parameters. See :func:`matplotlib.scatter` for more
        details. Defaults to ``'#1f78b4'``)
    :param str node_shape: The optional shape node. The specification is the
        same as the :func:`matplotlib.pyplot.scatter` function's ``marker``
        kwarg, valid options are one of
        ``['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8']``. Defaults to
        ``'o'``
    :param float alpha: Optional value for node and edge transparency
    :param matplotlib.colors.Colormap cmap: An optional Matplotlib colormap
        object for mapping intensities of nodes
    :param float vmin: Optional minimum value for node colormap scaling
    :param float vmax: Optional minimum value for node colormap scaling
    :param float|sequence linewidths: An optional line width for symbol
        borders. If a sequence is specified it must be the same length as
        node_list. Defaults to 1.0
    :param float|sequence width: An optional width to use for edges. Can
        either be a float or sequence  of floats. If a sequence is specified
        it must be the same length as node_list. Defaults to 1.0
    :param str|sequence edge_color: color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edge_list. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the ``edge_cmap`` and ``edge_vmin``,
        ``edge_vmax`` parameters.
    :param matplotlib.colors.Colormap edge_cmap: An optional Matplotlib
        colormap for mapping intensities of edges.
    :param float edge_vmin: Optional minimum value for edge colormap scaling
    :param float edge_vmax: Optional maximum value for node colormap scaling
    :param str style: An optional string to specify the edge line style.
        For example, ``'-'``, ``'--'``, ``'-.'``, ``':'`` or words like
        ``'solid'`` or ``'dashed'``. See the
        :class:`matplotlib.patches.FancyArrowPatch` attribute and kwarg
        ``linestyle`` for more details. Defaults to ``'solid'``.
    :param func labels: An optional callback function that will be passed a
        node payload and return a string label for the node. For example::

            labels=str

        could be used to just return a string cast of the node's data payload.
        Or something like::

            labels=lambda node: node['label']

        could be used if the node payloads are dictionaries.
    :param func edge_labels: An optional callback function that will be passed
        an edge payload and return a string label for the edge. For example::

            edge_labels=str

        could be used to just return a string cast of the edge's data payload.
        Or something like::

            edge_labels=lambda edge: edge['label']

        could be used if the edge payloads are dictionaries. If this is set
        edge labels will be drawn in the visualization.
    :param int font_size: An optional fontsize to use for text labels, By
        default a value of 12 is used for nodes and 10 for edges.
    :param str font_color: An optional font color for strings. By default
        ``'k'`` (ie black) is set.
    :param str font_weight: An optional string used to specify the font weight.
        By default a value of ``'normal'`` is used.
    :param str font_family: An optional font family to use for strings. By
        default ``'sans-serif'`` is used.
    :param str label: An optional string label to use for the graph legend.
    :param str connectionstyle: An optional value used to create a curved arc
        of rounding radius rad. For example,
        ``connectionstyle='arc3,rad=0.2'``. See
        :class:`matplotlib.patches.ConnectionStyle` and
        :class:`matplotlib.patches.FancyArrowPatch` for more info. By default
        this is set to ``"arc3"``.

    :returns: A matplotlib figure for the visualization if not running with an
        interactive backend (like in jupyter) or if ``ax`` is not set.
    :rtype: matplotlib.figure.Figure

    For Example:

    .. jupyter-execute::

        import matplotlib.pyplot as plt

        import rustworkx as rx
        from rustworkx.visualization import mpl_draw

        G = rx.generators.directed_path_graph(25)
        mpl_draw(G)
        plt.draw()
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e
    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor("w")
    if ax is None:
        if cf.axes:
            ax = cf.gca()
        else:
            ax = cf.add_axes((0, 0, 1, 1))

    draw_graph(graph, pos=pos, ax=ax, arrows=arrows, with_labels=with_labels, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    if not plt.isinteractive() or ax is None:
        return cf


def draw_graph(graph, pos=None, arrows=True, with_labels=False, **kwds):
    r"""Draw the graph using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    graph: A rustworkx :class:`~rustworkx.PyDiGraph` or
        :class:`~rustworkx.PyGraph`

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :mod:`rustworkx.drawing.layout` for functions that
        compute node positions.


    Notes
    -----
    For directed graphs, arrows  are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e

    valid_node_kwds = {
        "node_list",
        "node_size",
        "node_color",
        "node_shape",
        "alpha",
        "cmap",
        "vmin",
        "vmax",
        "ax",
        "linewidths",
        "edgecolors",
        "label",
    }

    valid_edge_kwds = {
        "edge_list",
        "width",
        "edge_color",
        "style",
        "alpha",
        "arrowstyle",
        "arrow_size",
        "edge_cmap",
        "edge_vmin",
        "edge_vmax",
        "ax",
        "label",
        "node_size",
        "node_list",
        "node_shape",
        "connectionstyle",
        "min_source_margin",
        "min_target_margin",
    }

    valid_label_kwds = {
        "labels",
        "font_size",
        "font_color",
        "font_family",
        "font_weight",
        "alpha",
        "bbox",
        "ax",
        "horizontalalignment",
        "verticalalignment",
    }

    valid_edge_label_kwds = {
        "edge_labels",
        "font_size",
        "font_color",
        "font_family",
        "font_weight",
        "alpha",
        "bbox",
        "ax",
        "rotate",
        "horizontalalignment",
        "verticalalignment",
    }

    valid_kwds = valid_node_kwds | valid_edge_kwds | valid_label_kwds | valid_edge_label_kwds

    if any([k not in valid_kwds for k in kwds]):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    label_fn = kwds.pop("labels", None)
    if label_fn:
        kwds["labels"] = {x: label_fn(graph[x]) for x in graph.node_indices()}
    edge_label_fn = kwds.pop("edge_labels", None)
    if edge_label_fn:
        kwds["edge_labels"] = {
            (x[0], x[1]): edge_label_fn(x[2]) for x in graph.weighted_edge_list()
        }

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    if isinstance(edge_kwds.get("alpha"), list):
        del edge_kwds["alpha"]
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}
    edge_label_kwds = {k: v for k, v in kwds.items() if k in valid_edge_label_kwds}

    if pos is None:
        pos = rustworkx.spring_layout(graph)  # default to spring layout

    draw_nodes(graph, pos, **node_kwds)
    draw_edges(graph, pos, arrows=arrows, **edge_kwds)
    if with_labels:
        draw_labels(graph, pos, **label_kwds)
    if edge_label_fn:
        draw_edge_labels(graph, pos, **edge_label_kwds)
    plt.draw_if_interactive()


def draw_nodes(
    graph,
    pos,
    node_list=None,
    node_size=300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
):
    """Draw the nodes of the graph.

    This draws only the nodes of the graph.

    :param graph: A rustworkx graph, either a :class:`~rustworkx.PyGraph` or a
        :class:`~rustworkx.PyDiGraph`.

    :param dict pos: A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    :param Axes ax: An optional Matplotlib Axes object, if specified it will
        draw the graph in the specified Matplotlib axes.

    :param list node_list: If specified only draw the specified node indices.
        If not specified all nodes in the graph will be drawn.

    :param float|array node_size: Size of nodes. If an array it must be the
        same length as node_list. Defaults to 300

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as node_list. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or array of floats (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap (default=None)
        Colormap for mapping intensities of nodes

    vmin,vmax : floats or None (default=None)
        Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence] (default=1.0)
        Line width of symbol border

    edgecolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders

    label : [None | string]
        Label for legend

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    """
    try:
        import matplotlib as mpl
        import matplotlib.collections  # type: ignore
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e

    if ax is None:
        ax = plt.gca()

    if node_list is None:
        node_list = graph.node_indices()

    # empty node_list, no drawing
    if len(node_list) == 0:
        return mpl.collections.PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in node_list])
    except KeyError as e:
        raise IndexError(f"Node {e} has no position.") from e

    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, node_list, cmap, vmin, vmax)
        alpha = None

    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    node_collection.set_zorder(2)
    return node_collection


def draw_edges(
    graph,
    pos,
    edge_list=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrow_size=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=True,
    label=None,
    node_size=300,
    node_list=None,
    node_shape="o",
    connectionstyle="arc3",
    min_source_margin=0,
    min_target_margin=0,
):
    r"""Draw the edges of the graph.

    This draws only the edges of the graph.

    Parameters
    ----------
    graph: A rustworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_list : collection of edge tuples (default=graph.edge_list())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edge_list. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax
        parameters.

    style : string (default=solid line)
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or None (default=None)
        The edge transparency

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
        For directed graphs, if True set default to drawing arrowheads.
        Otherwise set default to no arrowheads. Ignored if `arrowstyle` is set.

        Note: Arrows will be the same color as edges.

    arrowstyle : str (default='-\|>' if directed else '-')
        For directed graphs and `arrows==True` defaults to '-\|>',
        otherwise defaults to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrow_size : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        ``mutation_scale`` for more info.

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    node_list : list, optional (default=graph.node_indices())
       This provides the node order for the `node_size` array (if it is an
       array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of
        'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the beginning of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.
    """
    try:
        import matplotlib as mpl
        import matplotlib.colors  # type: ignore
        import matplotlib.patches  # type: ignore
        import matplotlib.path  # type: ignore
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e

    if arrowstyle is None:
        if isinstance(graph, rustworkx.PyDiGraph) and arrows:
            arrowstyle = "-|>"
        else:
            arrowstyle = "-"

    if ax is None:
        ax = plt.gca()

    if edge_list is None:
        edge_list = graph.edge_list()

    if len(edge_list) == 0:  # no edges!
        return []

    if node_list is None:
        node_list = list(graph.node_indices())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos_keys = dict()
    for e in edge_list:
        edge_pos_keys[(tuple(pos[e[0]]), tuple(pos[e[1]]))] = None
    edge_pos = edge_pos_keys.keys()

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.all([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    # Note: Waiting for someone to implement arrow to intersection with
    # marker.  Meanwhile, this works well for polygons with more than 4
    # sides and circle.

    def to_marker_edge(marker_size, marker):
        if marker in "s^>v<d":  # `large` markers need extra space
            return np.sqrt(2 * marker_size) / 2
        else:
            return np.sqrt(marker_size) / 2

    # Draw arrows with `matplotlib.patches.FancyarrowPatch`
    arrow_collection = []
    mutation_scale = arrow_size  # scale factor of arrow head

    base_connectionstyle = mpl.patches.ConnectionStyle(connectionstyle)

    # Fallback for self-loop scale. Left outside of _connectionstyle so it is
    # only computed once
    max_nodesize = np.array(node_size).max()

    # FancyArrowPatch doesn't handle color strings
    arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
    for i, edge in enumerate(edge_pos):
        x1, y1 = edge[0][0], edge[0][1]
        x2, y2 = edge[1][0], edge[1][1]
        shrink_source = 0  # space from source to tail
        shrink_target = 0  # space from  head to target
        if np.iterable(node_size):  # many node sizes
            source, target = edge_list[i][:2]
            source_node_size = node_size[node_list.index(source)]
            target_node_size = node_size[node_list.index(target)]
            shrink_source = to_marker_edge(source_node_size, node_shape)
            shrink_target = to_marker_edge(target_node_size, node_shape)
        else:
            shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

        if shrink_source < min_source_margin:
            shrink_source = min_source_margin

        if shrink_target < min_target_margin:
            shrink_target = min_target_margin

        if len(arrow_colors) == len(edge_pos):
            arrow_color = arrow_colors[i]
        elif len(arrow_colors) == 1:
            arrow_color = arrow_colors[0]
        else:  # Cycle through colors
            arrow_color = arrow_colors[i % len(arrow_colors)]

        if np.iterable(width):
            if len(width) == len(edge_pos):
                line_width = width[i]
            else:
                line_width = width[i % len(width)]
        else:
            line_width = width

        # radius of edges
        if tuple(reversed(edge)) in edge_pos:
            rad = 0.25
        else:
            rad = 0.0

        arrow = mpl.patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=arrowstyle,
            shrinkA=shrink_source,
            shrinkB=shrink_target,
            mutation_scale=mutation_scale,
            color=arrow_color,
            linewidth=line_width,
            connectionstyle=connectionstyle + f", rad = {rad}",
            linestyle=style,
            zorder=1,
        )  # arrows go behind nodes

        arrow_collection.append(arrow)
        ax.add_patch(arrow)

    edge_pos = np.asarray(tuple(edge_pos))

    # compute view
    mirustworkx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - mirustworkx
    h = maxy - miny

    def _connectionstyle(posA, posB, *args, **kwargs):
        # check if we need to do a self-loop
        if np.all(posA == posB):
            # Self-loops are scaled by view extent, except in cases the extent
            # is 0, e.g. for a single node. In this case, fall back to scaling
            # by the maximum node size
            selfloop_ht = 0.005 * max_nodesize if h == 0 else h
            # this is called with _screen space_ values so covert back
            # to data space
            data_loc = ax.transData.inverted().transform(posA)
            v_shift = 0.1 * selfloop_ht
            h_shift = v_shift * 0.5
            # put the top of the loop first so arrow is not hidden by node
            path = [
                # 1
                data_loc + np.asarray([0, v_shift]),
                # 4 4 4
                data_loc + np.asarray([h_shift, v_shift]),
                data_loc + np.asarray([h_shift, 0]),
                data_loc,
                # 4 4 4
                data_loc + np.asarray([-h_shift, 0]),
                data_loc + np.asarray([-h_shift, v_shift]),
                data_loc + np.asarray([0, v_shift]),
            ]

            ret = mpl.path.Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])
        # if not, fall back to the user specified behavior
        else:
            ret = base_connectionstyle(posA, posB, *args, **kwargs)

        return ret

    # update view
    padx, pady = 0.05 * w, 0.05 * h
    corners = (mirustworkx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection


def draw_labels(
    graph,
    pos,
    labels=None,
    font_size=12,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    clip_on=True,
):
    """Draw node labels on the graph.

    Parameters
    ----------
    graph: A rustworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    labels : dictionary (default={n: n for n in graph})
        Node labels in a dictionary of text labels keyed by node.
        Node-keys in labels should appear as keys in `pos`.
        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`

    font_size : int (default=12)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline',
                            'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed on the nodes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = {n: n for n in graph.node_indices()}

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=clip_on,
        )
        text_items[n] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def draw_edge_labels(
    graph,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
):
    """Draw edge labels.

    Parameters
    ----------
    graph: A rustworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline',
                            'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (default=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in graph.weighted_edge_list()}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        if (n2, n1) in labels.keys():  # loop
            dy = np.abs(y2 - y1)
            if n2 > n1:
                y -= 0.25 * dy
            else:
                y += 0.25 * dy

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)), xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    """Apply an alpha (or list of alphas) to the colors provided.

    Parameters
    ----------

    colors : color string or array of floats (default='r')
        Color of element. Can be a single color format string,
        or a sequence of colors with the same length as node_list.
        If numeric values are specified they will be mapped to
        colors using the cmap and vmin,vmax parameters.  See
        matplotlib.scatter for more details.

    alpha : float or array of floats
        Alpha values for elements. This can be a single alpha value, in
        which case it will be applied to all the elements of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    elem_list : array of rustworkx objects
        The list of elements which are being colored. These could be nodes,
        edges or labels.

    cmap : matplotlib colormap
        Color map for use if colors is a list of floats corresponding to points
        on a color mapping.

    vmin, vmax : float
        Minimum and maximum values for normalizing colors if a colormap is used

    Returns
    -------

    rgba_colors : numpy ndarray
        Array containing RGBA format values for each of the node colours.

    """
    try:
        import matplotlib as mpl
        import matplotlib.colors  # call as mpl.colors
        import matplotlib.cm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "matplotlib needs to be installed prior to running "
            "rustworkx.visualization.mpl_draw(). You can install "
            "matplotlib with:\n'pip install matplotlib'"
        ) from e

    # If we have been provided with a list of numbers as long as elem_list,
    # apply the color mapping.
    if len(colors) == len(elem_list) and isinstance(colors[0], Number):
        mapper = mpl.cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_colors = mapper.to_rgba(colors)
    # Otherwise, convert colors to matplotlib's RGB using the colorConverter
    # object.  These are converted to numpy ndarrays to be consistent with the
    # to_rgba method of ScalarMappable.
    else:
        try:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(colors)])
        except ValueError:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(color) for color in colors])
    # Set the final column of the rgba_colors to have the relevant alpha values
    try:
        # If alpha is longer than the number of colors, resize to the number of
        # elements.  Also, if rgba_colors.size (the number of elements of
        # rgba_colors) is the same as the number of elements, resize the array,
        # to avoid it being interpreted as a colormap by scatter()
        if len(alpha) > len(rgba_colors) or rgba_colors.size == len(elem_list):
            rgba_colors = np.resize(rgba_colors, (len(elem_list), 4))
            rgba_colors[1:, 0] = rgba_colors[0, 0]
            rgba_colors[1:, 1] = rgba_colors[0, 1]
            rgba_colors[1:, 2] = rgba_colors[0, 2]
        rgba_colors[:, 3] = list(islice(cycle(alpha), len(rgba_colors)))
    except TypeError:
        rgba_colors[:, -1] = alpha
    return rgba_colors
