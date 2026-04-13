# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import importlib
import sys
import functools

from .rustworkx import *

# flake8: noqa
import rustworkx.visit

__rustworkx_mod__ = importlib.import_module(".rustworkx", package="rustworkx")
sys.modules["rustworkx.generators"] = __rustworkx_mod__.generators


class PyDAG(PyDiGraph):
    """A class for creating direct acyclic graphs.

    PyDAG is just an alias of the PyDiGraph class and behaves identically to
    the :class:`~rustworkx.PyDiGraph` class and can be used interchangeably
    with ``PyDiGraph``. It currently exists solely as a backwards
    compatibility alias for users of rustworkx from prior to the
    0.4.0 release when there was no PyDiGraph class.

    The PyDAG class is used to create a directed graph. It can be a
    multigraph (have multiple edges between nodes). Each node and edge
    (although rarely used for edges) is indexed by an integer id. These ids
    are stable for the lifetime of the graph object and on node or edge
    deletions you can have holes in the list of indices for the graph.
    Node indices will be reused on additions after removal. For example:

    .. jupyter-execute::

        import rustworkx as rx

        graph = rx.PyDAG()
        graph.add_nodes_from(list(range(5)))
        graph.add_nodes_from(list(range(2)))
        graph.remove_node(2)
        print("After deletion:", graph.node_indices())
        res_manual = graph.add_parent(6, None, None)
        print("After adding a new node:", graph.node_indices())

    Additionally, each node and edge contains an arbitrary Python object as a
    weight/data payload.

    You can use the index for access to the data payload as in the
    following example:

    .. jupyter-execute::

        import rustworkx as rx

        graph = rx.PyDAG()
        data_payload = "An arbitrary Python object"
        node_index = graph.add_node(data_payload)
        print(f"Node Index: {node_index}")
        print(graph[node_index])

    The PyDAG class implements the Python mapping protocol for nodes so in
    addition to access you can also update the data payload with:

    .. jupyter-execute::

        import rustworkx as rx

        graph = rx.PyDAG()
        data_payload = "An arbitrary Python object"
        node_index = graph.add_node(data_payload)
        graph[node_index] = "New Payload"
        print(f"Node Index: {node_index}")
        print(graph[node_index])

    The PyDAG class has an option for real time cycle checking which can
    be used to ensure any edges added to the graph does not introduce a cycle.
    By default the real time cycle checking feature is disabled for
    performance, however you can enable it by setting the ``check_cycle``
    attribute to True. For example::

        import rustworkx as rx
        dag = rx.PyDAG()
        dag.check_cycle = True

    or at object creation::

        import rustworkx as rx
        dag = rx.PyDAG(check_cycle=True)

    With check_cycle set to true any calls to :meth:`PyDAG.add_edge` will
    ensure that no cycles are added, ensuring that the PyDAG class truly
    represents a directed acyclic graph. Do note that this cycle checking on
    :meth:`~PyDAG.add_edge`, :meth:`~PyDigraph.add_edges_from`,
    :meth:`~PyDAG.add_edges_from_no_data`,
    :meth:`~PyDAG.extend_from_edge_list`,  and
    :meth:`~PyDAG.extend_from_weighted_edge_list` comes with a performance
    penalty that grows as the graph does.  If you're adding a node and edge at
    the same time, leveraging :meth:`PyDAG.add_child` or
    :meth:`PyDAG.add_parent` will avoid this overhead.

    By default a ``PyDAG`` is a multigraph (meaning there can be parallel
    edges between nodes) however this can be disabled by setting the
    ``multigraph`` kwarg to ``False`` when calling the ``PyDAG`` constructor.
    For example::

        import rustworkx as rx
        dag = rx.PyDAG(multigraph=False)

    This can only be set at ``PyDiGraph`` initialization and not adjusted after
    creation. When :attr:`~rustworkx.PyDiGraph.multigraph` is set to ``False``
    if a method call is made that would add a parallel edge it will instead
    update the existing edge's weight/data payload.

    The maximum number of nodes and edges allowed on a ``PyGraph`` object is
    :math:`2^{32} - 1` (4,294,967,294) each. Attempting to add more nodes or
    edges than this will result in an exception being raised.

    :param bool check_cycle: When this is set to ``True`` the created
        ``PyDAG`` has runtime cycle detection enabled.
    :param bool multigraph: When this is set to ``False`` the created
        ``PyDAG`` object will not be a multigraph. When ``False`` if a method
        call is made that would add parallel edges the weight/weight from
        that method call will be used to update the existing edge in place.
    """

    pass


def _rustworkx_dispatch(func):
    """Decorator to dispatch rustworkx universal functions to the correct typed function"""

    func_name = func.__name__
    wrapped_func = functools.singledispatch(func)

    wrapped_func.register(PyDiGraph, vars(__rustworkx_mod__)[f"digraph_{func_name}"])
    wrapped_func.register(PyGraph, vars(__rustworkx_mod__)[f"graph_{func_name}"])

    return wrapped_func


@_rustworkx_dispatch
def distance_matrix(graph, parallel_threshold=300, as_undirected=False, null_value=0.0):
    """Get the distance matrix for a graph

    This differs from functions like :func:`~rustworkx.floyd_warshall_numpy` in
    that the edge weight/data payload is not used and each edge is treated as a
    distance of 1.

    This function is also multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 300). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.

    :param graph: The graph to get the distance matrix for, can be either a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param int parallel_threshold: The number of nodes to calculate the
        the distance matrix in parallel at. It defaults to 300, but this can
        be tuned
    :param bool as_undirected: If set to ``True`` the input directed graph
        will be treat as if each edge was bidirectional/undirected in the
        output distance matrix.
    :param float null_value: An optional float that will treated as a null
        value. This is the default value in the output matrix and it is used
        to indicate the absence of an edge between 2 nodes. By default this
        is ``0.0``.

    :returns: The distance matrix
    :rtype: numpy.ndarray
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def unweighted_average_shortest_path_length(graph, parallel_threshold=300, disconnected=False):
    r"""Return the average shortest path length with unweighted edges.

    The average shortest path length is calculated as

    .. math::

        a =\sum_{s,t \in V, s \ne t} \frac{d(s, t)}{n(n-1)}

    where :math:`V` is the set of nodes in ``graph``, :math:`d(s, t)` is the
    shortest path length from :math:`s` to :math:`t`, and :math:`n` is the
    number of nodes in ``graph``. If ``disconnected`` is set to ``True``,
    the average will be taken only between connected nodes.

    This function is also multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 300). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
    By default it will use all available CPUs if the environment variable is
    not specified.

    :param graph: The graph to compute the average shortest path length for,
        can be either a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param int parallel_threshold: The number of nodes to calculate the
        the distance matrix in parallel at. It defaults to 300, but this can
        be tuned to any number of nodes.
    :param bool as_undirected: If set to ``True`` the input directed graph
        will be treated as if each edge was bidirectional/undirected while
        finding the shortest paths. Default: ``False``.
    :param bool disconnected: If set to ``True`` only connected vertex pairs
        will be included in the calculation. If ``False``, infinity is returned
        for disconnected graphs. Default: ``False``.

    :returns: The average shortest path length. If no vertex pairs can be included
        in the calculation this will return NaN.

    :rtype: float
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def adjacency_matrix(graph, weight_fn=None, default_weight=1.0, null_value=0.0):
    """Return the adjacency matrix for a graph object

    In the case where there are multiple edges between nodes the value in the
    output matrix will be the sum of the edges' weights.

    :param graph: The graph used to generate the adjacency matrix from. Can
        either be a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param callable weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells rustworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            adjacency_matrix(graph, weight_fn: lambda x: 1)

        to return a weight of 1 for all edges. Also::

            adjacency_matrix(graph, weight_fn: lambda x: float(x))

        to cast the edge object as a float as the weight. If this is not
        specified a default value (either ``default_weight`` or 1) will be used
        for all edges.
    :param float default_weight: If ``weight_fn`` is not used this can be
        optionally used to specify a default weight to use for all edges.
    :param float null_value: An optional float that will treated as a null
        value. This is the default value in the output matrix and it is used
        to indicate the absence of an edge between 2 nodes. By default this is
        ``0.0``.

     :return: The adjacency matrix for the input dag as a numpy array
     :rtype: numpy.ndarray
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_simple_paths(graph, from_, to, min_depth=None, cutoff=None):
    """Return all simple paths between 2 nodes in a PyGraph object

    A simple path is a path with no repeated nodes.

    :param graph: The graph to find the path in. Can either be a
        class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int from_: The node index to find the paths from
    :param int | Iterable[int] to: The node index(es) to find the paths to
    :param int min_depth: The minimum depth of the path to include in the
        output list of paths. By default all paths are included regardless of
        depth, setting to 0 will behave like the default.
    :param int cutoff: The maximum depth of path to include in the output list
        of paths. By default includes all paths regardless of depth, setting to
        0 will behave like default.

    :returns: A list of lists where each inner list is a path of node indices
    :rtype: list
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def floyd_warshall(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    """Find all-pairs shortest path lengths using Floyd's algorithm

    Floyd's algorithm is used for finding shortest paths in dense graphs
    or graphs with negative weights (where Dijkstra's algorithm fails).

    This function is multithreaded and will launch a pool with threads equal
    to the number of CPUs by default if the number of nodes in the graph is
    above the value of ``parallel_threshold`` (it defaults to 300).
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads if parallelization was enabled.

    :param graph: The graph to run Floyd's algorithm on. Can
        either be a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param callable weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells rustworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            floyd_warshall(graph, weight_fn= lambda x: 1)

        to return a weight of 1 for all edges. Also::

            floyd_warshall(graph, weight_fn=float)

        to cast the edge object as a float as the weight. If this is not
        specified a default value (either ``default_weight`` or 1) will be used
        for all edges.
    :param float default_weight: If ``weight_fn`` is not used this can be
        optionally used to specify a default weight to use for all edges.
    :param int parallel_threshold: The number of nodes to execute
        the algorithm in parallel at. It defaults to 300, but this can
        be tuned

    :return: A read-only dictionary of path lengths. The keys are the source
        node indices and the values are a dict of the target node and the
        length of the shortest path to that node. For example::

            {
                0: {0: 0.0, 1: 2.0, 2: 2.0},
                1: {1: 0.0, 2: 1.0},
                2: {0: 1.0, 2: 0.0},
            }

    :rtype: AllPairsPathLengthMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def floyd_warshall_numpy(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    """Find all-pairs shortest path lengths using Floyd's algorithm

    Floyd's algorithm is used for finding shortest paths in dense graphs
    or graphs with negative weights (where Dijkstra's algorithm fails).

    This function is multithreaded and will launch a pool with threads equal
    to the number of CPUs by default if the number of nodes in the graph is
    above the value of ``parallel_threshold`` (it defaults to 300).
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads if parallelization was enabled.

    :param graph: The graph to run Floyd's algorithm on. Can
        either be a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param callable weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells rustworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            floyd_warshall_numpy(graph, weight_fn: lambda x: 1)

        to return a weight of 1 for all edges. Also::

            floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))

        to cast the edge object as a float as the weight. If this is not
        specified a default value (either ``default_weight`` or 1) will be used
        for all edges.
    :param float default_weight: If ``weight_fn`` is not used this can be
        optionally used to specify a default weight to use for all edges.
    :param int parallel_threshold: The number of nodes to execute
        the algorithm in parallel at. It defaults to 300, but this can
        be tuned

    :returns: A matrix of shortest path distances between nodes. If there is no
        path between two nodes then the corresponding matrix entry will be
        ``np.inf``.
    :rtype: numpy.ndarray
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def astar_shortest_path(graph, node, goal_fn, edge_cost_fn, estimate_cost_fn):
    """Compute the A* shortest path for a graph

    :param graph: The input graph to use. Can
        either be a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int node: The node index to compute the path from
    :param goal_fn: A python callable that will take in 1 parameter, a node's
        data object and will return a boolean which will be True if it is the
        finish node.
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the cost
        of that edge. It must be non-negative.
    :param estimate_cost_fn: A python callable that will take in 1 parameter, a
        node's data object and will return a float which represents the
        estimated cost for the next node. The return must be non-negative. For
        the algorithm to find the actual shortest path, it should be
        admissible, meaning that it should never overestimate the actual cost
        to get to the nearest goal node.

    :returns: The computed shortest path between node and finish as a list
        of node indices.
    :rtype: NodeIndices
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def dijkstra_shortest_paths(
    graph,
    source,
    target=None,
    weight_fn=None,
    default_weight=1.0,
    as_undirected=False,
):
    """Find the shortest path from a node

    This function will generate the shortest path from a source node using
    Dijkstra's algorithm.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int source: The node index to find paths from
    :param int target: An optional target to find a path to
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float
        which will be used to represent the weight/cost of the edge
    :param float default_weight: If ``weight_fn`` isn't specified this optional
        float value will be used for the weight/cost of each edge.
    :param bool as_undirected: If set to true the graph will be treated as
        undirected for finding the shortest path. This only works with a
        :class:`~rustworkx.PyDiGraph` input for ``graph``

    :return: Dictionary of paths. The keys are destination node indices and
        the dict values are lists of node indices making the path.
    :rtype: dict
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def has_path(
    graph,
    source,
    target,
    as_undirected=False,
):
    """Checks if a path exists between a source and target node

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int source: The node index to find paths from
    :param int target: The index of the target node
    :param bool as_undirected: If set to true the graph will be treated as
        undirected for finding existence of a path. This only works with a
        :class:`~rustworkx.PyDiGraph` input for ``graph``

    :return: True if a path exists, False if not
    :rtype: bool
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn):
    """For each node in the graph, finds the shortest paths to all others.

    This function will generate the shortest path from all nodes in the graph
    using Dijkstra's algorithm. This function is multithreaded and will run
    launch a thread pool with threads equal to the number of CPUs by default.
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of paths. The keys are source node
        indices and the values are a dict of target node indices and a list
        of node indices making the path. For example::

            {
                0: {1: [0, 1],  2: [0, 1, 2]},
                1: {2: [1, 2]},
                2: {0: [2, 0]},
            }

    :rtype: AllPairsPathMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_pairs_all_simple_paths(graph, min_depth=None, cutoff=None):
    """Return all the simple paths between all pairs of nodes in the graph

    This function is multithreaded and will launch a thread pool with threads
    equal to the number of CPUs by default. You can tune the number of threads
    with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
    ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.

    :param graph: The graph to find all simple paths in. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`
    :param int min_depth: The minimum depth of the path to include in the output
        list of paths. By default all paths are included regardless of depth,
        setting to 0 will behave like the default.
    :param int cutoff: The maximum depth of path to include in the output list
        of paths. By default includes all paths regardless of depth, setting to
        0 will behave like default.

    :returns: A mapping of source node indices to a mapping of target node
        indices to a list of paths between the source and target nodes.
    :rtype: AllPairsMultiplePathMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_pairs_dijkstra_path_lengths(graph, edge_cost_fn):
    """For each node in the graph, calculates the lengths of the shortest paths to all others.

    This function will generate the shortest path lengths from all nodes in the
    graph using Dijkstra's algorithm. This function is multithreaded and will
    launch a thread pool with threads equal to the number of CPUs by
    default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of path lengths. The keys are the source
        node indices and the values are a dict of the target node and the
        length of the shortest path to that node. For example::

            {
                0: {1: 2.0, 2: 2.0},
                1: {2: 1.0},
                2: {0: 1.0},
            }

    :rtype: AllPairsPathLengthMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    """Compute the lengths of the shortest paths for a graph object using
    Dijkstra's algorithm.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int node: The node index to use as the source for finding the
        shortest paths from
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the
        cost/weight of that edge. It must be non-negative
    :param int goal: An optional node index to use as the end of the path.
        When specified the traversal will stop when the goal is reached and
        the output dictionary will only have a single entry with the length
        of the shortest path to the goal node.

    :returns: A dictionary of the shortest paths from the provided node where
        the key is the node index of the end of the path and the value is the
        cost/sum of the weights of path
    :rtype: dict
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def k_shortest_path_lengths(graph, start, k, edge_cost, goal=None):
    """Compute the length of the kth shortest path

    Computes the lengths of the kth shortest path from ``start`` to every
    reachable node.

    Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).

    :param graph: The graph to find the shortest paths in. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int start: The node index to find the shortest paths from
    :param int k: The kth shortest path to find the lengths of
    :param edge_cost: A python callable that will receive an edge payload and
        return a float for the cost of that edge
    :param int goal: An optional goal node index, if specified the output
        dictionary

    :returns: A dict of lengths where the key is the destination node index and
        the value is the length of the path.
    :rtype: dict
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def dfs_edges(graph, source=None):
    """Get an edge list of the tree edges from a depth-first traversal

    The pseudo-code for the DFS algorithm is listed below. The output
    contains the tree edges found by the procedure.

    ::

        DFS(G, v)
          let S be a stack
          label v as discovered
          PUSH(S, (v, iterator of G.neighbors(v)))
          while (S != Ø)
              let (v, iterator) := LAST(S)
              if hasNext(iterator) then
                  w := next(iterator)
                  if w is not labeled as discovered then
                      label w as discovered                   # (v, w) is a tree edge
                      PUSH(S, (w, iterator of G.neighbors(w)))
              else
                  POP(S)
          end while

    .. note::

        If the input is an undirected graph with a single connected component,
        the output of this function is a spanning tree.

    :param graph: The graph to get the DFS edge list from. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int source: An optional node index to use as the starting node
        for the depth-first search. The edge list will only return edges in
        the components reachable from this index. If this is not specified
        then a source will be chosen arbitrarily and repeated until all
        components of the graph are searched.

    :returns: A list of edges as a tuple of the form ``(source, target)`` in
        depth-first order
    :rtype: EdgeList
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def is_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    call_limit=None,
):
    """Determine if 2 graphs are isomorphic

    This checks if 2 graphs are isomorphic both structurally and also
    comparing the node and edge data using the provided matcher functions.
    The matcher functions take in 2 data objects and will compare them. A
    simple example that checks if they're just equal would be::

            graph_a = rustworkx.PyGraph()
            graph_b = rustworkx.PyGraph()
            rustworkx.is_isomorphic(graph_a, graph_b,
                                lambda x, y: x == y)

    .. note::

        For better performance on large graphs, consider setting
        `id_order=False`.

    :param first: The first graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable node_matcher: A python callable object that takes 2
        positional one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are viewed
        as matching.
    :param callable edge_matcher: A python callable object that takes 2
        positional one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop and return ``False``.

    :returns: ``True`` if the 2 graphs are isomorphic, ``False`` if they are
        not.
    :rtype: bool

    .. [VF2] VF2++  An Improved Subgraph Isomorphism Algorithm
        by Alpár Jüttner and Péter Madarasi
    """
    raise TypeError(f"Invalid Input Type {type(first)} for graph")


def is_isomorphic_node_match(first, second, matcher, id_order=True):
    """Determine if 2 graphs are isomorphic

    This checks if 2 graphs are isomorphic both structurally and also
    comparing the node data using the provided matcher function. The matcher
    function takes in 2 node data objects and will compare them. A simple
    example that checks if they're just equal would be::

        graph_a = rustworkx.PyDAG()
        graph_b = rustworkx.PyDAG()
        rustworkx.is_isomorphic_node_match(graph_a, graph_b,
                                        lambda x, y: x == y)

    .. note::

        For better performance on large graphs, consider setting
        `id_order=False`.

    :param first: The first graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable matcher: A python callable object that takes 2 positional
        one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are viewed
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.

    :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
        not.
    :rtype: bool
    """
    return is_isomorphic(first, second, matcher, None, id_order)


@_rustworkx_dispatch
def is_subgraph_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=False,
    induced=True,
    call_limit=None,
):
    """Determine if 2 graphs are subgraph isomorphic

    This checks if 2 graphs are subgraph isomorphic both structurally and also
    comparing the node and edge data using the provided matcher functions.
    The matcher functions take in 2 data objects and will compare them.
    Since there is an ambiguity in the term 'subgraph', do note that we check
    for an node-induced subgraph if argument `induced` is set to `True`. If it is
    set to `False`, we check for a non induced subgraph, meaning the second graph
    can have fewer edges than the subgraph of the first. By default it's `True`. A
    simple example that checks if they're just equal would be::

            graph_a = rustworkx.PyGraph()
            graph_b = rustworkx.PyGraph()
            rustworkx.is_subgraph_isomorphic(graph_a, graph_b,
                                            lambda x, y: x == y)


    :param first: The first graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable node_matcher: A python callable object that takes 2
        positional one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are viewed
        as matching.
    :param callable edge_matcher: A python callable object that takes 2
        positional one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``True`` this function will match the nodes
        in order specified by their ids. Otherwise it will default to a heuristic
        matching order based on [VF2]_ paper.
    :param bool induced: If set to ``True`` this function will check the existence
        of a node-induced subgraph of first isomorphic to second graph.
        Default: ``True``.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop and return ``False``.

    :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`
        , ``False`` if there is not.
    :rtype: bool
    """
    raise TypeError(f"Invalid Input Type {type(first)} for graph")


@_rustworkx_dispatch
def transitivity(graph):
    """Compute the transitivity of a graph.

    This function is multithreaded and will run
    launch a thread pool with threads equal to the number of CPUs by default.
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    .. note::

        The function implicitly assumes that there are no parallel edges
        or self loops. It may produce incorrect/unexpected results if the
        input graph has self loops or parallel edges.

    :param graph: The graph to be used. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.

    :returns: Transitivity of the graph.
    :rtype: float
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def core_number(graph):
    """Return the core number for each node in the graph.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    .. note::

        The function implicitly assumes that there are no parallel edges
        or self loops. It may produce incorrect/unexpected results if the
        input graph has self loops or parallel edges.

    :param graph: The graph to get core numbers. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`

    :returns: A dictionary keyed by node index to the core number
    :rtype: dict
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def complement(graph):
    """Compute the complement of a graph.

    :param graph: The graph to be used, can be either a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.

    :returns: The complement of the graph.
    :rtype: :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`

    .. note::
        Parallel edges and self-loops are never created,
        even if the ``multigraph`` is set to ``True``
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def random_layout(graph, center=None, seed=None):
    """Generate a random layout

    :param PyGraph graph: The graph to generate the layout for
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position
    :param int seed: An optional seed to set for the random number generator.

    :returns: The random layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def spring_layout(
    graph,
    pos=None,
    fixed=None,
    k=None,
    repulsive_exponent=2,
    adaptive_cooling=True,
    num_iter=50,
    tol=1e-6,
    weight_fn=None,
    default_weight=1,
    scale=1,
    center=None,
    seed=None,
):
    """
    Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.

    :param graph: Graph to be used. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param dict pos:
        Initial node positions as a dictionary with node ids as keys and values
        as a coordinate list. If ``None``, then use random initial positions.
        (``default=None``)
    :param set fixed: Nodes to keep fixed at initial position.
        Error raised if fixed specified and ``pos`` is not. (``default=None``)
    :param float  k:
        Optimal distance between nodes. If ``None`` the distance is set to
        :math:`\\frac{1}{\\sqrt{n}}` where :math:`n` is the number of nodes.
        Increase this value to move nodes farther apart. (``default=None``)
    :param int repulsive_exponent:
        Repulsive force exponent. (``default=2``)
    :param bool adaptive_cooling:
        Use an adaptive cooling scheme. If set to ``False``,
        a linear cooling scheme is used. (``default=True``)
    :param int num_iter:
        Maximum number of iterations. (``default=50``)
    :param float tol:
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.
        (``default = 1e-6``)
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float
        which will be used to represent the weight of the edge.
    :param float (default=1) default_weight: If ``weight_fn`` isn't specified
        this optional float value will be used for the weight/cost of each edge
    :param float|None scale: Scale factor for positions.
        Not used unless fixed is None. If scale is ``None``, no re-scaling is
        performed. (``default=1.0``)
    :param list center: Coordinate pair around which to center
        the layout. Not used unless fixed is ``None``. (``default=None``)
    :param int seed: An optional seed to use for the random number generator

    :returns: A dictionary of positions keyed by node id.
    :rtype: dict
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


def networkx_converter(graph, keep_attributes: bool = False):
    """Convert a networkx graph object into a rustworkx graph object.

    .. note::

        networkx is **not** a dependency of rustworkx and this function
        is provided as a convenience method for users of both networkx and
        rustworkx. This function will not work unless you install networkx
        independently.

    :param networkx.Graph graph: The networkx graph to convert.
    :param bool keep_attributes: If ``True``, add networkx node attributes to
        the data payload in the nodes of the output rustworkx graph. When set to
        ``True``, the node data payloads in the output rustworkx graph object
        will be dictionaries with the node attributes from the input networkx
        graph where the ``"__networkx_node__"`` key contains the node from the
        input networkx graph.

    :returns: A rustworkx graph, either a :class:`~rustworkx.PyDiGraph` or a
        :class:`~rustworkx.PyGraph` based on whether the input graph is directed
        or not.
    :rtype: :class:`~rustworkx.PyDiGraph` or :class:`~rustworkx.PyGraph`
    """
    if graph.is_directed():
        new_graph = PyDiGraph(multigraph=graph.is_multigraph())
    else:
        new_graph = PyGraph(multigraph=graph.is_multigraph())
    nodes = list(graph.nodes)
    node_indices = dict(zip(nodes, new_graph.add_nodes_from(nodes)))
    new_graph.add_edges_from(
        [(node_indices[x[0]], node_indices[x[1]], x[2]) for x in graph.edges(data=True)]
    )

    if keep_attributes:
        for node, node_index in node_indices.items():
            attributes = graph.nodes[node]
            attributes["__networkx_node__"] = node
            new_graph[node_index] = attributes

    return new_graph


@_rustworkx_dispatch
def bipartite_layout(
    graph,
    first_nodes,
    horizontal=False,
    scale=1,
    center=None,
    aspect_ratio=4 / 3,
):
    """Generate a bipartite layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param set first_nodes: The set of node indices on the left (or top if
        horizontal is true)
    :param bool horizontal: An optional bool specifying the orientation of the
        layout
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position
    :param float aspect_ratio: An optional number for the ratio of the width to
        the height of the layout.

    :returns: The bipartite layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def circular_layout(graph, scale=1, center=None):
    """Generate a circular layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position

    :returns: The circular layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def shell_layout(graph, nlist=None, rotate=None, scale=1, center=None):
    """
    Generate a shell layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param list nlist: The list of lists of indices which represents each shell
    :param float rotate: Angle (in radians) by which to rotate the starting
        position of each shell relative to the starting position of the
        previous shell
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position

    :returns: The shell layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def spiral_layout(graph, scale=1, center=None, resolution=0.35, equidistant=False):
    """
    Generate a spiral layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position
    :param float resolution: The compactness of the spiral layout returned.
        Lower values result in more compressed spiral layouts.
    :param bool equidistant: If true, nodes will be plotted equidistant from
        each other.

    :returns: The spiral layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def num_shortest_paths_unweighted(graph, source):
    """Get the number of unweighted shortest paths from a source node

    :param PyDiGraph graph: The graph to find the number of shortest paths on
    :param int source: The source node to find the shortest paths from

    :returns: A mapping of target node indices to the number of shortest paths
        from ``source`` to that node. If there is no path from ``source`` to
        a node in the graph that node will not be preset in the output mapping.
    :rtype: NodesCountMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def betweenness_centrality(graph, normalized=True, endpoints=False, parallel_threshold=50):
    r"""Returns the betweenness centrality of each node in the graph.

    Betweenness centrality of a node :math:`v` is the sum of the
    fraction of all-pairs shortest paths that pass through :math:`v`

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the number of
    shortest :math:`(s, t)` paths, and :math:`\sigma(s, t|v)` is the number of
    those paths  passing through some  node :math:`v` other than :math:`s, t`.
    If :math:`s = t`, :math:`\sigma(s, t) = 1`, and if :math:`v \in {s, t}`,
    :math:`\sigma(s, t|v) = 0`

    The algorithm used in this function is based on:

    Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
    Journal of Mathematical Sociology 25(2):163-177, 2001.

    This function is multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 50). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.

    See Also
    --------
    :func:`~rustworkx.edge_betweenness_centrality`

    :param PyDiGraph graph: The input graph
    :param bool normalized: Whether to normalize the betweenness scores by
        the number of distinct paths between all pairs of nodes.
    :param bool endpoints: Whether to include the endpoints of paths in
        path lengths used to compute the betweenness.
    :param int parallel_threshold: The number of nodes to calculate the
        the betweenness centrality in parallel at if the number of nodes in
        the graph is less than this value it will run in a single thread. The
        default value is 50

    :returns: A dictionary mapping each node index to its betweenness centrality.
    :rtype: dict
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def closeness_centrality(graph, wf_improved=True, parallel_threshold=50):
    r"""Compute the closeness centrality of each node in a graph object.

    The closeness centrality of a node :math:`u` is defined as the
    reciprocal of the average shortest path distance to :math:`u` over all
    :math:`n-1` reachable nodes in the graph. In its general form this can
    be expressed as:

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where:

      * :math:`d(v, u)` - the shortest-path distance between :math:`v` and
        :math:`u`
      * :math:`n` - the number of nodes that can reach :math:`u`.

    In the case of a graphs with more than one connected component there is
    an alternative improved formula that calculates the closeness centrality
    as "a ratio of the fraction of actors in the group who are reachable, to
    the average distance" [WF]_. This can be expressed as

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where :math:`N` is the number of nodes in the graph. This alternative
    formula can be used with the ``wf_improved`` argument.

    This function is multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 50). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.

    :param graph: The input graph. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param bool wf_improved: This is optional; the default is True. If True,
        scale by the fraction of nodes reachable.
    :param int parallel_threshold: The number of nodes to calculate the
        the closeness centrality in parallel at if the number of nodes in
        the graph is less than this value it will run in a single thread. The
        default value is 50.

    :returns: A dictionary mapping each node index to its closeness centrality.
    :rtype: dict

    .. [WF] Wasserman, S., & Faust, K. (1994). Social Network Analysis:
      Methods and Applications (Structural Analysis in the Social Sciences).
      Cambridge: Cambridge University Press. doi:10.1017/CBO9780511815478
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def newman_weighted_closeness_centrality(
    graph, weight_fn=None, wf_improved=True, default_weight=1.0, parallel_threshold=50
):
    r"""Compute the weighted closeness centrality of each node in the graph.

    The weighted closeness centrality is an extension of the standard closeness
    centrality measure where edge weights represent connection strength rather
    than distance. To properly compute shortest paths, weights are inverted
    so that stronger connections correspond to shorter effective distances.
    The algorithm follows the method described by Newman (2001) in analyzing
    weighted graphs.[Newman]

    The edges originally represent connection strength between nodes.
    The idea is that if two nodes have a strong connection, the computed
    distance between them should be small (shorter), and vice versa.
    Note that this assume that the graph is modelling a measure of
    connection strength (e.g. trust, collaboration, or similarity).
    If the graph is not modelling a measure of connection strength,
    the function `weight_fn` should invert the weights before calling this
    function, if not it is considered as a logical error.

    In the case of a graphs with more than one connected component there is
    an alternative improved formula that calculates the closeness centrality
    as "a ratio of the fraction of actors in the group who are reachable, to
    the average distance".[WF]
    You can enable this by setting `wf_improved` to `true`.

    This function is multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 50). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.

    :param PyGraph graph: The input graph. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param weight_fn: An optional input callable that will be passed the edge's
       payload object and is expected to return a `float` weight for that edge.
       If this is not specified ``default_weight`` will be used as the weight
       for every edge in ``graph``
    :param bool wf_improved: This is optional; the default is True. If True,
      scale by the fraction of nodes reachable.
    :param float default_weight: If ``weight_fn`` is not set the default weight
        value to use for the weight of all edges
    :param int parallel_threshold: The number of nodes to calculate the
        the closeness centrality in parallel at if the number of nodes in
        the graph is less than this value it will run in a single thread. The
        default value is 50.

    :returns: A dictionary mapping each node index to its closeness centrality.
    :rtype: CentralityMapping

    .. [Newman]: Newman, M. E. J. (2001). Scientific collaboration networks.
        II. Shortest paths, weighted networks, and centrality.
        Physical Review E, 64(1), 016132.

    .. [WF]: Wasserman, S., & Faust, K. (1994). Social Network Analysis:
        Methods and Applications (Structural Analysis in the Social Sciences).
        Cambridge: Cambridge University Press.
        <https://doi.org/10.1017/CBO9780511815478>
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def degree_centrality(graph):
    r"""Compute the degree centrality of each node in a graph object.

    :param graph: The input graph. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.

    :returns: a read-only dict-like object whose keys are edges and values are the
        degree centrality score for each node.
    :rtype: CentralityMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def edge_betweenness_centrality(graph, normalized=True, parallel_threshold=50):
    r"""Compute the edge betweenness centrality of all edges in a graph.

    Edge betweenness centrality of an edge :math:`e` is the sum of the
    fraction of all-pairs shortest paths that pass through :math:`e`

    .. math::

       c_B(e) = \sum_{s,t \in V} \frac{\sigma(s, t|e)}{\sigma(s, t)}

    where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the
    number of shortest :math:`(s, t)`-paths, and :math:`\sigma(s, t|e)` is
    the number of those paths passing through edge :math:`e`.

    The above definition and the algorithm used in this function is based on:

    Ulrik Brandes, On Variants of Shortest-Path Betweenness Centrality
    and their Generic Computation. Social Networks 30(2):136-145, 2008.

    This function is multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 50). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.

    See Also
    --------
    :func:`~rustworkx.betweenness_centrality`

    :param PyGraph graph: The input graph
    :param bool normalized: Whether to normalize the betweenness scores by the
        number of distinct paths between all pairs of nodes.
    :param int parallel_threshold: The number of nodes to calculate
        the edge betweenness centrality in parallel at if the number of nodes in
        the graph is less than this value it will run in a single thread. The
        default value is 50

    :returns: a read-only dict-like object whose keys are edges and values are the
        betweenness score for each node.
    :rtype: EdgeCentralityMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def eigenvector_centrality(graph, weight_fn=None, default_weight=1.0, max_iter=100, tol=1e-6):
    """Compute the eigenvector centrality of a graph.

    For details on the eigenvector centrality refer to:

    Phillip Bonacich. “Power and Centrality: A Family of Measures.”
    American Journal of Sociology 92(5):1170–1182, 1986
    <https://doi.org/10.1086/228631>

    This function uses a power iteration method to compute the eigenvector
    and convergence is not guaranteed. The function will stop when `max_iter`
    iterations is reached or when the computed vector between two iterations
    is smaller than the error tolerance multiplied by the number of nodes.
    The implementation of this algorithm is based on the NetworkX
    `eigenvector_centrality() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html>`__
    function.

    In the case of multigraphs the weights of any parallel edges will be
    summed when computing the eigenvector centrality.

    :param graph: Graph to be used. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param weight_fn: An optional input callable that will be passed the edge's
        payload object and is expected to return a `float` weight for that edge.
        If this is not specified ``default_weight`` will be used as the weight
        for every edge in ``graph``
    :param float default_weight: If ``weight_fn`` is not set the default weight
        value to use for the weight of all edges
    :param int max_iter: The maximum number of iterations in the power method. If
        not specified a default value of 100 is used.
    :param float tol: The error tolerance used when checking for convergence in the
        power method. If this is not specified default value of 1e-6 is used.

    :returns: a read-only dict-like object whose keys are the node indices and values are the
         centrality score for that node.
    :rtype: CentralityMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def katz_centrality(
    graph, alpha=0.1, beta=1.0, weight_fn=None, default_weight=1.0, max_iter=100, tol=1e-6
):
    """Compute the Katz centrality of a graph.

    For details on the Katz centrality refer to:

    Leo Katz. “A New Status Index Derived from Sociometric Index.”
    Psychometrika 18(1):39–43, 1953
    <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>

    This function uses a power iteration method to compute the eigenvector
    and convergence is not guaranteed. The function will stop when `max_iter`
    iterations is reached or when the computed vector between two iterations
    is smaller than the error tolerance multiplied by the number of nodes.
    The implementation of this algorithm is based on the NetworkX
    `katz_centrality() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html>`__
    function.

    In the case of multigraphs the weights of any parallel edges will be
    summed when computing the Katz centrality.

    :param graph: Graph to be used. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param float alpha: Attenuation factor. If this is not specified default value of 0.1 is used.
    :param float | dict beta: Immediate neighbourhood weights. If a float is provided, the neighbourhood
        weight is used for all nodes. If a dictionary is provided, it must contain all node indices.
        If beta is not specified, a default value of 1.0 is used.
    :param weight_fn: An optional input callable that will be passed the edge's
        payload object and is expected to return a `float` weight for that edge.
        If this is not specified ``default_weight`` will be used as the weight
        for every edge in ``graph``
    :param float default_weight: If ``weight_fn`` is not set the default weight
        value to use for the weight of all edges
    :param int max_iter: The maximum number of iterations in the power method. If
        not specified a default value of 100 is used.
    :param float tol: The error tolerance used when checking for convergence in the
        power method. If this is not specified default value of 1e-6 is used.

    :returns: a read-only dict-like object whose keys are the node indices and values are the
         centrality score for that node.
    :rtype: CentralityMapping
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def vf2_mapping(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    subgraph=False,
    induced=True,
    call_limit=None,
):
    """
    Return an iterator over all vf2 mappings between two graphs.

    This function will run the vf2 algorithm used from
    :func:`~rustworkx.is_isomorphic` and :func:`~rustworkx.is_subgraph_isomorphic`
    but instead of returning a boolean it will return an iterator over all possible
    mapping of node ids found from ``first`` to ``second``. If the graphs are not
    isomorphic then the iterator will be empty. A simple example that retrieves
    one mapping would be::

            graph_a = rustworkx.generators.path_graph(3)
            graph_b = rustworkx.generators.path_graph(2)
            vf2 = rustworkx.vf2_mapping(graph_a, graph_b, subgraph=True)
            try:
                mapping = next(vf2)
            except StopIteration:
                pass

    :param first: The first graph to find the mapping for
    :param second: The second graph to find the mapping for
    :param node_matcher: An optional python callable object that takes 2
        positional arguments, one for each node data object in either graph.
        If the return of this function evaluates to True then the nodes
        passed to it are viewed as matching.
    :param edge_matcher: A python callable object that takes 2 positional
        one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.
    :param bool subgraph: If set to ``True`` the function will return the
        subgraph isomorphic found between the graphs.
    :param bool induced: If set to ``True`` this function will check the existence
        of a node-induced subgraph of first isomorphic to second graph.
        Default: ``True``.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop. Default: ``None``.

    :returns: An iterator over dictionaries of node indices from ``first`` to node
        indices in ``second`` representing the mapping found.
    :rtype: Iterable[NodeMap]
    """
    raise TypeError(f"Invalid Input Type {type(first)} for graph")


@_rustworkx_dispatch
def union(
    first,
    second,
    merge_nodes=False,
    merge_edges=False,
):
    """Return a new graph by forming a union from two input graph objects

    The algorithm in this function operates in three phases:

    1. Add all the nodes from  ``second`` into ``first``. operates in
    :math:`\\mathcal{O}(n_2)`, with :math:`n_2` being number of nodes in
    ``second``.
    2. Merge nodes from ``second`` over ``first`` given that:

       - The ``merge_nodes`` is ``True``. operates in :math:`\\mathcal{O}(n_1 n_2)`,
         with :math:`n_1` being the number of nodes in ``first`` and :math:`n_2`
         the number of nodes in ``second``
       - The respective node in ``second`` and ``first`` share the same
         weight/data payload.

    3. Adds all the edges from ``second`` to ``first``. If the ``merge_edges``
       parameter is ``True`` and the respective edge in ``second`` and
       ``first`` share the same weight/data payload they will be merged together.

    :param first: The first graph object
    :param second: The second graph object
    :param bool merge_nodes: If set to ``True`` nodes will be merged between
        ``second`` and ``first`` if the weights are equal. Default: ``False``.
    :param bool merge_edges: If set to ``True`` edges will be merged between
        ``second`` and ``first`` if the weights are equal. Default: ``False``.

    :returns: A new graph object that is the union of ``second`` and
        ``first``. It's worth noting the weight/data payload objects are
        passed by reference from ``first`` and ``second`` to this new object.
    :rtype: :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    """
    raise TypeError(f"Invalid Input Type {type(first)} for graph")


@_rustworkx_dispatch
def tensor_product(
    first,
    second,
):
    """Return a new graph by forming the tensor product
    from two input graph objects

    :param first: The first graph object
    :param second: The second graph object

    :returns: A new graph object that is the tensor product of ``second`` and
        ``first``. It's worth noting the weight/data payload objects are
        passed by reference from ``first`` and ``second`` to this new object.
        A read-only dictionary of the product of nodes is also returned. The keys
        are a tuple where the first element is a node of the first graph and the
        second element is a node of the second graph, and the values are the map
        of those elements to node indices in the product graph. For example::

            {
                (0, 0): 0,
                (0, 1): 1,
            }

    :rtype: Tuple[:class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`,
        :class:`~rustworkx.ProductNodeMap`]
    """
    raise TypeError(f"Invalid Input Type {type(first)} for graph")


@_rustworkx_dispatch
def cartesian_product(
    first,
    second,
):
    """Return a new graph by forming the cartesian product
    from two input graph objects

    :param first: The first graph object
    :param second: The second graph object

    :returns: A new graph object that is the union of ``second`` and
        ``first``. It's worth noting the weight/data payload objects are
        passed by reference from ``first`` and ``second`` to this new object.
        A read-only dictionary of the product of nodes is also returned. The keys
        are a tuple where the first element is a node of the first graph and the
        second element is a node of the second graph, and the values are the map
        of those elements to node indices in the product graph. For example::

            {
                (0, 0): 0,
                (0, 1): 1,
            }

    :rtype: Tuple[:class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`,
        :class:`~rustworkx.ProductNodeMap`]
    """
    raise TypeError(f"Invalid Input Type {type(first)} for graph")


@_rustworkx_dispatch
def bfs_search(graph, source, visitor):
    """Breadth-first traversal of a directed/undirected graph with several source vertices.

    Pseudo-code for the breadth-first search algorithm with a single source vertex is
    listed below with annotated event points at which a method of the given
    :class:`~rustworkx.visit.BFSVisitor` is called.

    ::

      # G - graph, s - single source node
      BFS(G, s)
        let color be a mapping             # color[u] - vertex u color WHITE/GRAY/BLACK
        for each u in G                    # u - vertex in G
          color[u] := WHITE                # color all vertices as undiscovered
        end for
        let Q be a queue
        ENQUEUE(Q, s)
        color[s] := GRAY                   # event: discover_vertex(s)
        while (Q is not empty)
          u := DEQUEUE(Q)
          for each v, w in OutEdges(G, u)  # v - target vertex, w - edge weight
            if (WHITE = color[v])          # event: tree_edge((u, v, w))
              color[v] := GRAY             # event: discover_vertex(v)
              ENQUEUE(Q, v)
            else                           # event: non_tree_edge((u, v, w))
              if (GRAY = color[v])         # event: gray_target_edge((u, v, w))
                ...
              elif (BLACK = color[v])      # event: black_target_edge((u, v, w))
                ...
          end for
          color[u] := BLACK                # event: finish_vertex(u)
        end while

    For several source nodes, the BFS algorithm is applied on source nodes by the given order.

    If an exception is raised inside the callback method of the
    :class:`~rustworkx.visit.BFSVisitor` instance, the graph traversal
    will be stopped immediately. You can exploit this to exit early by raising a
    :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
    will return but without raising back the exception. You can also prune part of
    the search tree by raising :class:`~rustworkx.visit.PruneSearch`.

    In the following example we keep track of the tree edges:

    .. jupyter-execute::

        import rustworkx as rx
        from rustworkx.visit import BFSVisitor

        class TreeEdgesRecorder(BFSVisitor):

            def __init__(self):
                self.edges = []

            def tree_edge(self, edge):
                self.edges.append(edge)

        graph = rx.PyDiGraph()
        graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
        vis = TreeEdgesRecorder()
        rx.bfs_search(graph, [0], vis)
        print('Tree edges:', vis.edges)

    Here is another example, using the :class:`~rustworkx.visit.PruneSearch`
    exception, to find the shortest path between two vertices with some
    restrictions on the edges
    (for a more efficient ways to find the shortest path, see :ref:`shortest-paths`):

    .. jupyter-execute::

        import rustworkx as rx
        from rustworkx.visit import BFSVisitor, PruneSearch


        graph = rx.PyGraph()
        home, market, school = graph.add_nodes_from(['home', 'market', 'school'])
        graph.add_edges_from_no_data(
            [(school, home), (school, market), (market, home)]
        )

        class DistanceHomeFinder(BFSVisitor):

            def __init__(self):
                self.distance = {}

            def discover_vertex(self, vertex):
                self.distance.setdefault(vertex, 0)

            def tree_edge(self, edge):
                source, target, _ = edge
                # the road directly from home to school is closed
                if {source, target} == {home, school}:
                    raise PruneSearch
                self.distance[target] = self.distance[source] + 1

        vis = DistanceHomeFinder()
        rx.bfs_search(graph, [school], vis)
        print('Distance from school to home:', vis.distance[home])

    .. note::

        Graph can **not** be mutated while traversing.
        Trying to do so raises an exception.


    .. note::

        An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
        raised in the :class:`~rustworkx.visit.BFSVisitor.finish_vertex` event.


    :param graph: The graph to be used. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`
    :param source: An optional list of node indices to use as the starting
        nodes for the breadth-first search. If ``None`` or not specified then a source
        will be chosen arbitrarily and repeated until all components of the
        graph are searched.
        This can be a ``Sequence[int]`` or ``None``.
    :param visitor: A visitor object that is invoked at the event points inside the
        algorithm. This should be a subclass of :class:`~rustworkx.visit.BFSVisitor`.
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def dfs_search(graph, source, visitor):
    """Depth-first traversal of a directed/undirected graph with several source vertices.

    Pseudo-code for the depth-first search algorithm with a single source vertex is
    listed below with annotated event points at which a method of the given
    :class:`~rustworkx.visit.DFSVisitor` is called.

    ::

      # G - graph, s - single source node
      DFS(G, s)
        let color be a mapping                        # color[u] - vertex u color WHITE/GRAY/BLACK
        for each u in G                               # u - vertex in G
          color[u] := WHITE                           # color all as undiscovered
        end for
        time := 0
        let S be a stack
        PUSH(S, (s, iterator of OutEdges(G, s)))      # S - stack of vertices and edge iterators
        color[s] := GRAY                              # event: discover_vertex(s, time)
        while (S is not empty)
          let (u, iterator) := LAST(S)
          flag := False                               # whether edge to undiscovered vertex found
          for each v, w in iterator                   # v - target vertex, w - edge weight
            if (WHITE = color[v])                     # event: tree_edge((u, v, w))
              time := time + 1
              color[v] := GRAY                        # event: discover_vertex(v, time)
              flag := True
              break
            elif (GRAY = color[v])                    # event: back_edge((u, v, w))
              ...
            elif (BLACK = color[v])                   # event: forward_or_cross_edge((u, v, w))
              ...
          end for
          if (flag is True)
            PUSH(S, (v, iterator of OutEdges(G, v)))
          elif (flag is False)
            time := time + 1
            color[u] := BLACK                         # event: finish_vertex(u, time)
            POP(S)
        end while

    For several source nodes, the DFS algorithm is applied on source nodes by the given order.

    If an exception is raised inside the callback method of the
    :class:`~rustworkx.visit.DFSVisitor` instance, the graph traversal
    will be stopped immediately. You can exploit this to exit early by raising a
    :class:`~rustworkx.visit.StopSearch` exception. You can also prune part of the
    search tree by raising :class:`~rustworkx.visit.PruneSearch`.

    In the following example we keep track of the tree edges:

    .. jupyter-execute::

           import rustworkx as rx
           from rustworkx.visit import DFSVisitor

           class TreeEdgesRecorder(DFSVisitor):

               def __init__(self):
                   self.edges = []

               def tree_edge(self, edge):
                   self.edges.append(edge)

           graph = rx.PyGraph()
           graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
           vis = TreeEdgesRecorder()
           rx.dfs_search(graph, [0], vis)
           print('Tree edges:', vis.edges)

    .. note::

        Graph can *not* be mutated while traversing.
        Trying to do so raises an exception.


    .. note::

        An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
        raised in the :class:`~rustworkx.visit.DFSVisitor.finish_vertex` event.

    :param graph: The graph to be used. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`
    :param source: An optional list of node indices to use as the starting
        nodes for the depth-first search. If ``None`` or not specified then a source
        will be chosen arbitrarily and repeated until all components of the
        graph are searched.
        This can be a ``Sequence[int]`` or ``None``.
    :param visitor: A visitor object that is invoked at the event points inside the
        algorithm. This should be a subclass of :class:`~rustworkx.visit.DFSVisitor`.
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def dijkstra_search(graph, source, weight_fn, visitor):
    """Dijkstra traversal of a graph with several source vertices.

    Pseudo-code for the Dijkstra algorithm with a single source vertex is
    listed below with annotated event points at which a method of the given
    :class:`~rustworkx.visit.DijkstraVisitor` is called.

    ::

      # G - graph, s - single source node, weight - edge cost function
      DIJKSTRA(G, s, weight)
        let score be empty mapping
        let visited be empty set
        let Q be a priority queue
        score[s] := 0.0
        PUSH(Q, (score[s], s))                # only score determines the priority
        while Q is not empty
          cost, u := POP-MIN(Q)
          if u in visited
            continue
          PUT(visited, u)                     # event: discover_vertex(u, cost)
          for each _, v, w in OutEdges(G, u)  # v - target vertex, w - edge weight
            ...                               # event: examine_edge((u, v, w))
            if v in visited
              continue
            next_cost = cost + weight(w)
            if {(v is key in score)
                and (score[v] <= next_cost)}  # event: edge_not_relaxed((u, v, w))
              ...
            else:                             # v not scored or scored higher
              score[v] = next_cost            # event: edge_relaxed((u, v, w))
              PUSH(Q, (next_cost, v))
          end for                             # event: finish_vertex(u)
        end while

    For several source nodes, the Dijkstra algorithm is applied on source nodes by the given order.

    If an exception is raised inside the callback method of the
    :class:`~rustworkx.visit.DijkstraVisitor` instance, the graph traversal
    will be stopped immediately. You can exploit this to exit early by raising a
    :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
    will return but without raising back the exception. You can also prune part of the
    search tree by raising :class:`~rustworkx.visit.PruneSearch`.

    In the following example we find the shortest path from vertex 0 to 5, and exit the visit as
    soon as we reach the goal vertex:

    .. jupyter-execute::

        import rustworkx as rx
        from rustworkx.visit import DijkstraVisitor, StopSearch

        graph = rx.PyDiGraph()
        graph.extend_from_edge_list([
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 3),
            (2, 3), (2, 4),
            (4, 5),
        ])

        class PathFinder(DijkstraVisitor):

            def __init__(self, start, goal):
                self.start = start
                self.goal = goal
                self.predecessors = {}

            def get_path(self):
                n = self.goal
                rev_path = [n]
                while n != self.start:
                    n = self.predecessors[n]
                    rev_path.append(n)
                return list(reversed(rev_path))

            def discover_vertex(self, vertex, cost):
                if vertex == self.goal:
                    raise StopSearch

            def edge_relaxed(self, edge):
                self.predecessors[edge[1]] = edge[0]

        start = 0
        vis = PathFinder(start=start, goal=5)
        rx.dijkstra_search(graph, [start], weight_fn=None, visitor=vis)
        print('Path:', vis.get_path())

    .. note::

        Graph can **not** be mutated while traversing.
        Trying to do so raises an exception.


    .. note::

        An exception is raised if the :class:`~rustworkx.visit.PruneSearch` is
        raised in the :class:`~rustworkx.visit.DijkstraVisitor.finish_vertex` event.

    :param graph: The graph to be used. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`.
    :param source: An optional list of node indices to use as the starting nodes
        for the dijkstra search. If ``None`` or not specified then a source
        will be chosen arbitrarily and repeated until all components of the
        graph are searched.
        This can be a ``Sequence[int]`` or ``None``.
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float which
        will be used to represent the weight/cost of the edge. If not specified,
        a default value of cost ``1.0`` will be used for each edge.
    :param visitor: A visitor object that is invoked at the event points inside the
        algorithm. This should be a subclass of :class:`~rustworkx.visit.DijkstraVisitor`.
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def bellman_ford_shortest_paths(
    graph,
    source,
    target=None,
    weight_fn=None,
    default_weight=1.0,
    as_undirected=False,
):
    """Find the shortest path from a node

    This function will generate the shortest path from a source node using
    the Bellman-Ford algorithm wit the SPFA heuristic.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int source: The node index to find paths from
    :param int target: An optional target to find a path to
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float
        which will be used to represent the weight/cost of the edge
    :param float default_weight: If ``weight_fn`` isn't specified this optional
        float value will be used for the weight/cost of each edge.
    :param bool as_undirected: If set to true the graph will be treated as
        undirected for finding the shortest path. This only works with a
        :class:`~rustworkx.PyDiGraph` input for ``graph``

    :return: A read-only dictionary of paths. The keys are destination node indices
        and the dict values are lists of node indices making the path.
    :rtype: PathMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def bellman_ford_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    """Compute the lengths of the shortest paths for a graph object using
    the Bellman-Ford algorithm with the SPFA heuristic.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int node: The node index to use as the source for finding the
        shortest paths from
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the
        cost/weight of that edge. It can be negative.
    :param int goal: An optional node index to use as the end of the path.
        When specified the output dictionary will only have a single entry with
        the length of the shortest path to the goal node.

    :returns: A read-only dictionary of the shortest paths from the provided node
        where the key is the node index of the end of the path and the value is the
        cost/sum of the weights of path
    :rtype: PathLengthMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_pairs_bellman_ford_path_lengths(graph, edge_cost_fn):
    """For each node in the graph, calculates the lengths of the shortest paths to all others.

    This function will generate the shortest path lengths from all nodes in the
    graph using the Bellman-Ford algorithm. This function is multithreaded and will
    launch a thread pool with threads equal to the number of CPUs by
    default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of path lengths. The keys are the source
        node indices and the values are a dict of the target node and the
        length of the shortest path to that node. For example::

            {
                0: {1: 2.0, 2: 2.0},
                1: {2: 1.0},
                2: {0: 1.0},
            }

    :rtype: AllPairsPathLengthMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_pairs_bellman_ford_shortest_paths(graph, edge_cost_fn):
    """For each node in the graph, finds the shortest paths to all others.

    This function will generate the shortest path from all nodes in the graph
    using the Bellman-Ford algorithm. This function is multithreaded and will run
    launch a thread pool with threads equal to the number of CPUs by default.
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of paths. The keys are source node
        indices and the values are a dict of target node indices and a list
        of node indices making the path. For example::

            {
                0: {1: [0, 1],  2: [0, 1, 2]},
                1: {2: [1, 2]},
                2: {0: [2, 0]},
            }

    :rtype: AllPairsPathMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def node_link_json(graph, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None):
    """Generate a JSON object representing a graph in a node-link format

    :param graph: The graph to generate the JSON for. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param str path: An optional path to write the JSON output to. If specified
        the function will not return anything and instead will write the JSON
        to the file specified.
    :param graph_attrs: An optional callable that will be passed the
        :attr:`~.PyGraph.attrs` attribute of the graph and is expected to
        return a dictionary of string keys to string values representing the
        graph attributes. This dictionary will be included as attributes in
        the output JSON. If anything other than a dictionary with string keys
        and string values is returned an exception will be raised.
    :param node_attrs: An optional callable that will be passed the node data
        payload for each node in the graph and is expected to return a
        dictionary of string keys to string values representing the data payload.
        This dictionary will be used as the ``data`` field for each node.
    :param edge_attrs:  An optional callable that will be passed the edge data
        payload for each node in the graph and is expected to return a
        dictionary of string keys to string values representing the data payload.
        This dictionary will be used as the ``data`` field for each edge.

    :returns: Either the JSON string for the payload or ``None`` if ``path`` is specified
    :rtype: str
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def longest_simple_path(graph):
    """Return a longest simple path in the graph

    This function searches computes all pairs of all simple paths and returns
    a path of the longest length from that set. It is roughly equivalent to
    running something like::

        from rustworkx import all_pairs_all_simple_paths

        max((y.values for y in all_pairs_all_simple_paths(graph).values()), key=lambda x: len(x))

    but this function will be more efficient than using ``max()`` as the search
    is evaluated in parallel before returning to Python. In the case of multiple
    paths of the same maximum length being present in the graph only one will be
    provided. There are no guarantees on which of the multiple longest paths
    will be returned (as it is determined by the parallel execution order). This
    is a tradeoff to improve runtime performance. If a stable return is required
    in such case consider using the ``max()`` equivalent above instead.

    This function is multithreaded and will launch a thread pool with threads
    equal to the number of CPUs by default. You can tune the number of threads
    with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
    ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.

    :param PyGraph graph: The graph to find the longest path in

    :returns: A sequence of node indices that represent the longest simple graph
        found in the graph. If the graph is empty ``None`` will be returned instead.
    :rtype: NodeIndices
    """


@_rustworkx_dispatch
def isolates(graph):
    """Return a list of isolates in a graph object

    An isolate is a node without any neighbors meaning it has a degree of 0. For
    directed graphs this means the in-degree and out-degree are both 0.

    :param graph: The input graph to find isolates in
    :returns: A list of node indices for isolates in the graph
    :rtype: NodeIndices
    """


@_rustworkx_dispatch
def two_color(graph):
    """Compute a two-coloring of a directed graph

    If a two coloring is not possible for the input graph (meaning it is not
    bipartite), ``None`` is returned.

    :param graph: The graph to find the coloring for
    :returns: If a coloring is possible return a dictionary of node indices to the color as an integer (0 or 1)
    :rtype: dict
    """


@_rustworkx_dispatch
def is_bipartite(graph):
    """Determine if a given graph is bipartite

    :param graph: The graph to check if it's bipartite
    :returns: ``True`` if the graph is bipartite and ``False`` if it is not
    :rtype: bool
    """


@_rustworkx_dispatch
def floyd_warshall_successor_and_distance(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    """
    Find all-pairs shortest path lengths using Floyd's algorithm.

    Floyd's algorithm is used for finding shortest paths in dense graphs
    or graphs with negative weights (where Dijkstra's algorithm fails).

    This function is multithreaded and will launch a pool with threads equal
    to the number of CPUs by default if the number of nodes in the graph is
    above the value of ``parallel_threshold`` (it defaults to 300).
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads if parallelization was enabled.

    :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
    :param weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells rustworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            floyd_warshall_successor_and_distance(graph, weight_fn=lambda _: 1)

        to return a weight of 1 for all edges. Also:

            floyd_warshall_successor_and_distance(graph, weight_fn=float)

        to cast the edge object as a float as the weight.
    :param as_undirected: If set to true each directed edge will be treated as
        bidirectional/undirected.
    :param int parallel_threshold: The number of nodes to execute
        the algorithm in parallel at. It defaults to 300, but this can
        be tuned

    :returns: A tuple of two matrices.
        First one is a matrix of shortest path distances between nodes. If there is no
        path between two nodes then the corresponding matrix entry will be
        ``np.inf``.
        Second one is a matrix of **next** nodes for given source and target. If there is no
        path between two nodes then the corresponding matrix entry will be the same as
        a target node. To reconstruct the shortest path among nodes::

            def reconstruct_path(source, target, successors):
                path = []
                if source == target:
                    return path
                curr = source
                while curr != target:
                    path.append(curr)
                    curr = successors[curr, target]
                path.append(target)
                return path

    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def all_shortest_paths(
    graph,
    source,
    target,
    weight_fn=None,
    default_weight=1.0,
    as_undirected=False,
):
    """
    Find all shortest paths between two nodes

    This function will generate all possible shortest paths from a source node to a
    target using Dijkstra's algorithm.

    :param graph: The input graph to find the shortest paths for
    :param int source: The node index to find paths from
    :param int target: A target to find paths to
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float which
        will be used to represent the weight/cost of the edge
    :param float default_weight: If ``weight_fn`` isn't specified this optional
        float value will be used for the weight/cost of each edge.

    :return: List of paths. Each paths are lists of node indices,
        starting at ``source`` and ending at ``target``.
    :rtype: list
    :raises ValueError: when an edge weight with NaN or negative value
        is provided.
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def condensation(graph, /, sccs=None):
    """Return the condensation of a directed or undirected graph
    The condensation of a directed graph is a directed acyclic graph (DAG) in which
    each node represents a strongly connected component (SCC) of the original graph.
    The edges of the DAG represent the connections between these components.
    The condensation of an undirected graph is a directed graph in which each node
    represents a connected component of the original graph. The edges of the DAG
    represent the connections between these components.

    The condensation is computed using Tarjan's algorithm.

    :param graph: The input graph to condense. This can be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param sccs: An optional list of strongly connected components (SCCs) to use.
        If not specified, the function will compute the SCCs internally.
        If the input graph is undirected, this parameter is ignored.
    :returns: A PyGraph or PyDiGraph object representing the condensation of the input graph.
    :rtype: PyGraph or PyDiGraph
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def single_source_all_shortest_paths(
    graph, source, weight_fn=None, default_weight=1.0, as_undirected=False
):
    """
    Find all shortest paths from a single source node to all other nodes.

    This function will generate all possible shortest paths from a source node to all other nodes
    using Dijkstra's algorithm. If ``as_undirected`` is True for directed graphs, it will be treated as undirected.

    :param graph: The input graph :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param int source: The node index to find paths from.
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float which
        will be used to represent the weight/cost of the edge.
    :param float default_weight: If ``weight_fn`` isn't specified this optional
        float value will be used for the weight/cost of each edge.
    :param bool as_undirected: If True, treat the directed graph as undirected (only for PyDiGraph).

    :return: A dictionary where keys are node indices and values are lists of all shortest paths
        from the source to that node. Each path is a list of node indices starting with the source.
    :rtype: dict
    :raises ValueError: when an edge weight with NaN or negative value is provided.
    :raises IndexError: if the source node index is out of range.

    .. warning::
        This function can return an exponential number of paths in certain graphs, especially with zero-weight edges.
        For most use cases, consider using `dijkstra_shortest_paths` for a single shortest path, which runs much faster.
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")


@_rustworkx_dispatch
def write_graphml(graph, path, /, keys=None, compression=None):
    """Write a graph to a file in GraphML format.

    GraphML is a comprehensive and easy-to-use file format for graphs. It consists
    of a language core to describe the structural properties of a graph and a flexible
    extension mechanism to add application-specific data.

    For more information see:
    http://graphml.graphdrawing.org/

    .. note::

        This implementation does not support mixed graphs (directed and undirected edges together),
        hyperedges, nested graphs, or ports.

    .. note::

        GraphML attributes with `graph` domain are written from the graph's attrs field.

    :param graph: The graph to write to the file. This can be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param path: The path of the output file to write.
    :param keys: Optional list of key definitions for GraphML attributes.
        If not specified, keys will be inferred from the graph data.
    :param compression: Optional compression format for the output file.
        If not specified, no compression is applied.
    :raises RuntimeError: when an error is encountered while writing the GraphML file.
    """
    raise TypeError(f"Invalid Input Type {type(graph)} for graph")
