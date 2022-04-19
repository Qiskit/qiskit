# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""VF2Layout pass to find a layout using subgraph isomorphism"""
from enum import Enum
import logging
import random
import time

from retworkx import PyGraph, PyDiGraph, vf2_mapping

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.exceptions import BackendPropertyError


logger = logging.getLogger(__name__)


class VF2LayoutStopReason(Enum):
    """Stop reasons for VF2Layout pass."""

    SOLUTION_FOUND = "solution found"
    NO_SOLUTION_FOUND = "nonexistent solution"
    MORE_THAN_2Q = ">2q gates in basis"


class VF2Layout(AnalysisPass):
    """A pass for choosing a Layout of a circuit onto a Coupling graph, as a
    a subgraph isomorphism problem, solved by VF2++.

    If a solution is found that means there is a "perfect layout" and that no
    further swap mapping or routing is needed. If a solution is found the layout
    will be set in the property set as ``property_set['layout']``. However, if no
    solution is found, no ``property_set['layout']`` is set. The stopping reason is
    set in ``property_set['VF2Layout_stop_reason']`` in all the cases and will be
    one of the values enumerated in ``VF2LayoutStopReason`` which has the
    following values:

        * ``"solution found"``: If a perfect layout was found.
        * ``"nonexistent solution"``: If no perfect layout was found.
        * ``">2q gates in basis"``: If VF2Layout can't work with basis

    """

    def __init__(
        self,
        coupling_map=None,
        strict_direction=False,
        seed=None,
        call_limit=None,
        time_limit=None,
        properties=None,
        max_trials=None,
        target=None,
    ):
        """Initialize a ``VF2Layout`` pass instance

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            strict_direction (bool): If True, considers the direction of the coupling map.
                                     Default is False.
            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.
            call_limit (int): The number of state visits to attempt in each execution of
                VF2.
            time_limit (float): The total time limit in seconds to run ``VF2Layout``
            properties (BackendProperties): The backend properties for the backend. If
                :meth:`~qiskit.providers.models.BackendProperties.readout_error` is available
                it is used to score the layout.
            max_trials (int): The maximum number of trials to run VF2 to find
                a layout. If this is not specified the number of trials will be limited
                based on the number of edges in the interaction graph or the coupling graph
                (whichever is larger). If set to a value <= 0 no limit on the number of trials
                will be set.
            target (Target): A target representing the backend device to run ``VF2Layout`` on.
                If specified it will supersede a set value for ``properties`` and
                ``coupling_map``.

        Raises:
            TypeError: At runtime, if neither ``coupling_map`` or ``target`` are provided.
        """
        super().__init__()
        self.target = target
        if target is not None:
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
        self.properties = properties
        self.strict_direction = strict_direction
        self.seed = seed
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.max_trials = max_trials

    def run(self, dag):
        """run the layout method"""
        if self.coupling_map is None:
            raise TranspilerError("coupling_map or target must be specified.")

        qubits = dag.qubits
        qubit_indices = {qubit: index for index, qubit in enumerate(qubits)}

        interactions = []
        for node in dag.op_nodes(include_directives=False):
            len_args = len(node.qargs)
            if len_args == 2:
                interactions.append((qubit_indices[node.qargs[0]], qubit_indices[node.qargs[1]]))
            if len_args >= 3:
                self.property_set["VF2Layout_stop_reason"] = VF2LayoutStopReason.MORE_THAN_2Q
                return

        if self.strict_direction:
            cm_graph = self.coupling_map.graph
            im_graph = PyDiGraph(multigraph=False)
        else:
            cm_graph = self.coupling_map.graph.to_undirected()
            im_graph = PyGraph(multigraph=False)

        cm_nodes = list(cm_graph.node_indexes())
        if self.seed != -1:
            random.Random(self.seed).shuffle(cm_nodes)
            shuffled_cm_graph = type(cm_graph)()
            shuffled_cm_graph.add_nodes_from(cm_nodes)
            new_edges = [(cm_nodes[edge[0]], cm_nodes[edge[1]]) for edge in cm_graph.edge_list()]
            shuffled_cm_graph.add_edges_from_no_data(new_edges)
            cm_nodes = [k for k, v in sorted(enumerate(cm_nodes), key=lambda item: item[1])]
            cm_graph = shuffled_cm_graph

        im_graph.add_nodes_from(range(len(qubits)))
        im_graph.add_edges_from_no_data(interactions)
        # To avoid trying to over optimize the result by default limit the number
        # of trials based on the size of the graphs. For circuits with simple layouts
        # like an all 1q circuit we don't want to sit forever trying every possible
        # mapping in the search space
        if self.max_trials is None:
            im_graph_edge_count = len(im_graph.edge_list())
            cm_graph_edge_count = len(cm_graph.edge_list())
            self.max_trials = max(im_graph_edge_count, cm_graph_edge_count) + 15

        logger.debug("Running VF2 to find mappings")
        mappings = vf2_mapping(
            cm_graph,
            im_graph,
            subgraph=True,
            id_order=False,
            induced=False,
            call_limit=self.call_limit,
        )
        chosen_layout = None
        chosen_layout_score = None
        start_time = time.time()
        trials = 0
        for mapping in mappings:
            trials += 1
            logger.debug("Running trial: %s", trials)
            stop_reason = VF2LayoutStopReason.SOLUTION_FOUND
            layout = Layout({qubits[im_i]: cm_nodes[cm_i] for cm_i, im_i in mapping.items()})
            # If the graphs have the same number of nodes we don't need to score or do multiple
            # trials as the score heuristic currently doesn't weigh nodes based on gates on a
            # qubit so the scores will always all be the same
            if len(cm_graph) == len(im_graph):
                chosen_layout = layout
                break
            layout_score = self._score_layout(layout)
            logger.debug("Trial %s has score %s", trials, layout_score)
            if chosen_layout is None:
                chosen_layout = layout
                chosen_layout_score = layout_score
            elif layout_score < chosen_layout_score:
                logger.debug(
                    "Found layout %s has a lower score (%s) than previous best %s (%s)",
                    layout,
                    layout_score,
                    chosen_layout,
                    chosen_layout_score,
                )
                chosen_layout = layout
                chosen_layout_score = layout_score
            if self.max_trials > 0 and trials >= self.max_trials:
                logger.debug("Trial %s is >= configured max trials %s", trials, self.max_trials)
                break
            elapsed_time = time.time() - start_time
            if self.time_limit is not None and elapsed_time >= self.time_limit:
                logger.debug(
                    "VF2Layout has taken %s which exceeds configured max time: %s",
                    elapsed_time,
                    self.time_limit,
                )
                break
        if chosen_layout is None:
            stop_reason = VF2LayoutStopReason.NO_SOLUTION_FOUND
        else:
            self.property_set["layout"] = chosen_layout
            for reg in dag.qregs.values():
                self.property_set["layout"].add_register(reg)

        self.property_set["VF2Layout_stop_reason"] = stop_reason

    def _score_layout(self, layout):
        """Score heurstic to determine the quality of the layout by looking at the readout fidelity
        on the chosen qubits. If BackendProperties are not available it uses the coupling map degree
        to weight against higher connectivity qubits."""
        bits = layout.get_physical_bits()
        score = 0
        if self.target is not None and "measure" in self.target:
            for bit in bits:
                props = self.target["measure"].get((bit,))
                if props is None or props.error is None:
                    score += (
                        self.coupling_map.graph.out_degree(bit)
                        + self.coupling_map.graph.in_degree(bit)
                    ) / len(self.coupling_map.graph)
                else:
                    score += props.error
        else:
            if self.properties is None:
                # Sum qubit degree for each qubit in chosen layout as really rough estimate of error
                for bit in bits:
                    score += self.coupling_map.graph.out_degree(
                        bit
                    ) + self.coupling_map.graph.in_degree(bit)
                return score
            for bit in bits:
                try:
                    score += self.properties.readout_error(bit)
                # If readout error can't be found in properties fallback to degree
                # divided by number of qubits as a terrible approximation
                except BackendPropertyError:
                    score += (
                        self.coupling_map.graph.out_degree(bit)
                        + self.coupling_map.graph.in_degree(bit)
                    ) / len(self.coupling_map.graph)
        return score
