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
A pass implementing the flexible-layer mapping algorithm.

That is a swap mapping algorithm proposed in the paper:
T. Itoko, R. Raymond, T. Imamichi, A. Matsuo, and A. W. Cross.
Quantum circuit compilers using gate commutation rules.
In Proceedings of ASP-DAC, pp. 191--196. ACM, 2019.
 (Its extended version is available at https://arxiv.org/abs/1907.02686 )
Note: This implementation does not include a post process for removing meaningless head swaps,
i.e. changing initial layout, which was applied in the experiments in the paper.
This is due to the limitation of swap mapper passes which are not allowed to change initial layout.

This algorithm considers the *dependency graph* of a given circuit
with less dependencies by considering commutativity of consecutive gates,
and updates `blocking gates` in the dependency graph by changing qubit layout
(= adding SWAPs). The blocking gates are the leading unresolved gates for
a current layout, and they can be seen as a kind of *flexible layer*
in contrast to many other swap passes assumes fixed layers as their input.
That's why this pass is named FlexlayerSwap pass.

(For the general role of the swap mapping pass, see :doc:`lookahed_swap`.)
"""
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

from .algorithm.dependency_graph import DependencyGraph
from .algorithm.flexlayer_heuristics import FlexlayerHeuristics
from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements


class FlexlayerSwap(TransformationPass):
    """
    Map input circuit onto a backend topology via insertion of SWAPs
    using flexible-layer mapping algorithm.
    """

    def __init__(self,
                 coupling_map,
                 dependency_graph_type="xz_commute",
                 lookahead_depth=10,
                 decay_rate=0.5):
        """
        Initialize a FlexlayerSwap instance.
        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            dependency_graph_type (str): Type of dependency graph:
                - "basic": consider only the commutation between gates without sharing qubits.
                - "xz_commute": consider four more commutation rules.
            lookahead_depth (int): How far gates from blocking gates should be looked ahead
            decay_rate (float): Decay rate of look-ahead weight (0 < decay_rate < 1)
        """
        super().__init__()
        self.requires.append(BarrierBeforeFinalMeasurements())
        self._coupling_map = coupling_map
        self._graph_type = dependency_graph_type
        self._lookahead_depth = lookahead_depth
        self._decay_rate = decay_rate

    def run(self, dag):
        """
        Runs the FlexlayerSwap pass on `dag`.
        Args:
            dag (DAGCircuit): A circuit to map.
        Returns:
            DAGCircuit: A mapped circuit compatible with the coupling_map.
        """
        initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        qc = dag_to_circuit(dag)
        dependency_graph = DependencyGraph(qc, graph_type=self._graph_type)
        algo = FlexlayerHeuristics(qc=qc,
                                   dependency_graph=dependency_graph,
                                   coupling=self._coupling_map,
                                   initial_layout=initial_layout,
                                   lookahead_depth=self._lookahead_depth,
                                   decay_rate=self._decay_rate)
        res_dag, _ = algo.search()
        return res_dag
