# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import CouplingMap, Layout
from qiskit.transpiler import TransformationPass
from .algorithm.dependency_graph import DependencyGraph
from .algorithm.flexlayer_heuristics import FlexlayerHeuristics
from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements


class FlexlayerSwap(TransformationPass):
    """
    Maps a DAGCircuit onto a `coupling_map` inserting swap gates.
    """

    def __init__(self,
                 coupling_map: CouplingMap,
                 initial_layout: Layout = None,
                 lookahead_depth: int = 10,
                 decay_rate: float = 0.5):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates for a given `initial_layout`.
        Args:
            coupling_map: Directed graph represented a coupling map.
            initial_layout: initial layout of qubits in mapping
            lookahead_depth: how far gates from blocking gates should be looked ahead
            decay_rate: decay rate of look-ahead weight (0 < decay_rate < 1)
        """
        super().__init__()
        self.requires.append(BarrierBeforeFinalMeasurements())
        self._coupling_map = coupling_map
        self._initial_layout = initial_layout
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
        if not self._initial_layout:
            self._initial_layout = self.property_set["layout"]
        if not self._initial_layout:
            self._initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        qc = dag_to_circuit(dag)
        dependency_graph = DependencyGraph(qc, graph_type="xz_commute")
        algo = FlexlayerHeuristics(qc=qc,
                                   dependency_graph=dependency_graph,
                                   coupling=self._coupling_map,
                                   initial_layout=self._initial_layout,
                                   lookahead_depth=self._lookahead_depth,
                                   decay_rate=self._decay_rate)
        res_dag, layout = algo.search()
        res_dag = physical_to_virtual(res_dag, layout)
        return res_dag


def physical_to_virtual(dag: DAGCircuit, initial_layout: Layout) -> DAGCircuit:
    """
    Convert a physical circuit `dag` into the virtual circuit under a given `initial_layout`.
    Args:
        dag: a physical circuit, assuming 'q' is the register name of its physical qubits.
        initial_layout: given initial layout.
    Returns:
        A converted circuit with virtual qubits
    """
    layout = {}
    qubits = dag.qubits()
    for k, v in initial_layout.get_physical_bits().items():
        layout[qubits[k]] = v

    circuit = dag_to_circuit(dag)
    circuit.qregs = initial_layout.get_registers()
    for gate in circuit.data:
        for i, q in enumerate(gate.qargs):
            gate.qargs[i] = layout[q]

    return circuit_to_dag(circuit)
