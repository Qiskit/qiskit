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
A pass implementing the flexible-layer mapper.

That is the swap mapper proposed in the paper:
T. Itoko, R. Raymond, T. Imamichi, A. Matsuo, and A. W. Cross.
Quantum circuit compilers using gate commutation rules.
In Proceedings of ASP-DAC, pp. 191--196. ACM, 2019.

This algorithm considers the *dependency graph* of a given circuit
with less dependencies by considering commutativity of consecutive gates,
and updates `blocking gates` in the dependency graph by changing qubit layout
(= adding SWAPs). The blocking gates are the leading unresolved gates for
a current layout, and they can be seen as a kind of *flexible layer*
in contrast to many other swap passes assumes fixed layers as their input.
That's why this pass is named FlexlayerSwap pass.

(For the general role of the swap mapper pass, see `lookahed_swap.py`.)
"""
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.layout import Layout

from .algorithm.dependency_graph import DependencyGraph
from .algorithm.flexlayer_heuristics import FlexlayerHeuristics
from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements


class FlexlayerSwap(TransformationPass):
    """
    Maps a DAGCircuit onto a `coupling_map` inserting swap gates.
    """

    def __init__(self,
                 coupling_map: CouplingMap,
                 lookahead_depth: int = 10,
                 decay_rate: float = 0.5):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates.
        Args:
            coupling_map: Directed graph represented a coupling map.
            lookahead_depth: how far gates from blocking gates should be looked ahead
            decay_rate: decay rate of look-ahead weight (0 < decay_rate < 1)
        """
        super().__init__()
        self.requires.append(BarrierBeforeFinalMeasurements())
        self._coupling_map = coupling_map
        self._lookahead_depth = lookahead_depth
        self._decay_rate = decay_rate

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Runs the FlexlayerSwap pass on `dag`.
        Args:
            dag: DAG to map.
        Returns:
            A mapped DAG (with virtual qubits).
        """
        initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        qc = dag_to_circuit(dag)
        dependency_graph = DependencyGraph(qc, graph_type="xz_commute")
        algo = FlexlayerHeuristics(qc=qc,
                                   dependency_graph=dependency_graph,
                                   coupling=self._coupling_map,
                                   initial_layout=initial_layout,
                                   lookahead_depth=self._lookahead_depth,
                                   decay_rate=self._decay_rate)
        res_dag, _ = algo.search()
        return res_dag
