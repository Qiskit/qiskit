# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-function-args

"""SATLayout pass to find a layout for commuting gates using subgraph isomorphism and a SAT solver"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from threading import Timer

import networkx as nx
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout, TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

logger = logging.getLogger(__name__)


@dataclass
class SATResult:
    """A data class to hold the result of a SAT solver."""

    satisfiable: bool  # Satisfiable is True if the SAT model could be solved in a given time.
    solution: dict  # The solution to the SAT problem if it is satisfiable.
    mapping: list  # The permutation of program qubits to the qubits of the swap strategy.
    elapsed_time: float  # The time it took to solve the SAT model.


class SATLayoutForCommutingGateRouter(AnalysisPass):
    r"""A pass to find a layout for commuting gates using a SAT solver.
    When this pass is run on a DAG it will look for the first instance of
    :class:`.Commuting2qBlock` and use the program graph :math:`P` of this block of gates to
    find a layout for a given swap strategy. This layout is found with a
    binary search over the layers :math:`l` of the swap strategy. At each considered layer
    a subgraph isomorphism problem formulated as a SAT is solved by a SAT solver. Each instance
    is whether it is possible to embed the program graph :math:`P` into the effective
    connectivity graph :math:`C_l` that is achieved by applying :math:`l` layers of the
    swap strategy to the coupling map :math:`C_0` of the backend. Since solving SAT problems
    can be hard, a ``time_out`` fixes the maximum time allotted to the SAT solver for each
    instance. If this time is exceeded the considered problem is deemed unsatisfiable and
    the binary search proceeds to the next number of swap layers :math:``l``.
    """

    def __init__(self, swap_strategy: SwapStrategy | None = None, timeout: int = 60):
        """Initialize the ``SATLayoutForCommutingGateRouter`` pass.

        Args:
            swap_strategy: The swap strategy that will later be used to route the commuting gates.
                If this variable is not set at initialization time then it will be obtained
                from the ``property_set`` of the transpiler.
            time_out: The allowed time in seconds for each iteration of the SAT solver. This
                variable defaults to 60 seconds.
        """
        self._time_out = timeout
        self._swap_strategy = None
        if swap_strategy is not None:
            self._swap_strategy = swap_strategy

        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the `SATLayoutForCommutingGateRouter` pass to the DAG."""

        if self._swap_strategy is None:
            if "swap_strategy" not in self.property_set:
                raise TranspilerError("The SAT mapping pass requires a swap strategy to be set.")
            swap_strategy = self.property_set["swap_strategy"]
        else:
            if "swap_strategy" in self.property_set:
                logger.warning(
                    "The swap strategy in the property set is overwritten by the SAT mapping pass."
                )
            swap_strategy = self._swap_strategy

        # 1. Construct program graph from the Commuting2qBlock
        program_graph = self._build_program_graph(dag)
        nx.draw(program_graph, with_labels=True)
        # 2. Binary search over the connectivity graph.
        sat_results = self.binary_sat_search(program_graph, swap_strategy)
        # 3. Build the layout from the SAT results and set it in the property set.
        layout = self._build_layout(sat_results, dag.qregs)
        self.property_set["layout"] = layout

    @staticmethod
    def _build_program_graph(dag: DAGCircuit) -> nx.Graph:
        """Build the program graph from the first encountered ``Commuting2qBlock``.

        Args:
            dag: A dag in which there is a ``Commuting2qBlock``.

        Returns:
            The program graph representing the two-qubit gates in the first encountered ``Commuting2qBlock``.

        Raises:
            TranspilerError: If the ``Commuting2qBlock`` contains more than two qubits.
        """
        program_graph = nx.Graph()
        program_graph.add_nodes_from([i for i in range(dag.num_qubits())])
        for node in dag.topological_op_nodes():
            if node.name == "commuting_2q_block":
                for commuting_node in node.op:
                    if isinstance(commuting_node.op, PauliEvolutionGate):
                        if commuting_node.op.num_qubits > 2:
                            raise TranspilerError(
                                "The SAT layout only works for commuting two-qubit PauliEvolutionGates.",
                                commuting_node.op,
                            )
                        program_graph.add_edge(
                            commuting_node.qargs[0].index, commuting_node.qargs[1].index
                        )
                # We only need to consider the first commuting block.
                break
        return program_graph

    @staticmethod
    def _build_connectivity_graph(swap_strategy: SwapStrategy, num_layers: int) -> nx.Graph:
        """Build the connectivity graph for a given swap strategy and number of layers.

        Args:
            swap_strategy: The swap strategy to use.
            num_layers: The number of layers to apply the swap strategy.

        Returns:
            The effective coupling map that is achieved by applying ``num_layers`` layers of the
            swap strategy to the coupling map of the backend.
        """
        distance_matrix = swap_strategy.distance_matrix
        rows, cols = np.where(distance_matrix <= num_layers)
        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from([i for i in range(distance_matrix.shape[0])])
        connectivity_graph.add_edges_from([(i, j) for i, j in zip(rows, cols) if i < j])
        return connectivity_graph

    @staticmethod
    def _build_connectivity_matrix(swap_strategy: SwapStrategy, num_layers: int) -> np.ndarray:
        """Build the connectivity matrix for a given swap strategy and number of layers.

        Args:
            swap_strategy: The swap strategy to use.
            num_layers: The number of layers to apply the swap strategy.

        Returns:
            An adjacency matrix representing the effective coupling map that is achieved by
            applying ``num_layers`` layers of the swap strategy to the coupling map of the backend.
        """
        distance_matrix = swap_strategy.distance_matrix
        return (distance_matrix <= num_layers).astype(int)

    @staticmethod
    def _build_adjacency_cnf(
        program_graph: nx.Graph,
        connectivity_matrix: np.ndarray,
        x: dict[tuple[int, int], int],
    ) -> list[list[int]]:
        """Build the adjacency constraints in CNF.

        Args:
            program_graph: The program graph representing the two-qubit gates in the circuit.
            connectivity_matrix: An adjacency matrix representing the effective coupling map
                achieved by the swap strategy after a certain number of layers.
            x: The SAT variables

        Returns:
            A SAT formulation of the initial mapping problem expressed on conformal normal form.
        """
        n = connectivity_matrix.shape[0]
        cnf = []
        for e0, e1 in program_graph.edges:
            clause_matrix = connectivity_matrix * x[e1, :]
            clause = np.concatenate(([[-x[e0, i]] for i in range(n)], clause_matrix), axis=1)
            # Remove 0s from each clause
            cnf.extend([c[c != 0].tolist() for c in clause])
        return cnf

    @staticmethod
    def _build_layout(sat_results: dict[int, SATResult], qregs) -> Layout:
        """Build a layout from the SAT results.

        The satisfiable SAT instance with the smallest number of layers ``k`` will be used to
        perform the initial mapping from program qubits to physical qubits.

        Args:
            sat_results: A dict of results from the sat solver. Each key is the number of
                of layers of the swap strategy considered. Each value are the results from the
                SAT solvers.
            qregs: The quantum registers in the dag.

        Returns:
            An instance of :class:`.Layout` generated from the SAT result.
        """
        min_k = min([k for k in sat_results if sat_results[k].satisfiable])
        qreg = list(qregs.values())[0]
        return Layout.from_intlist([cn for _, cn in sat_results[min_k].mapping], qreg)

    @staticmethod
    def binary_sat_search(
        program_graph: nx.Graph,
        swap_strategy: SwapStrategy,
        start: int | None = None,
        last: int | None = None,
        timeout: int = 60,
    ) -> dict[int, SATResult]:
        """Perform a binary search to find the minimum number of swap layers required to map the
        program graph to the connectivity graph.

        Args:
            program_graph: The program graph representing the two-qubit gates in the circuit.
            swap_strategy: The swap strategy to use.
            start: The smallest number of swap layers used to initialize the binary search.
                Defaults to ``None`` in which case the maximum degree of the program graph - 2 is used.
            last: The largest number of swap layers used to initialize the binary search.
                Defaults to ``None`` in which case the number of qubits in the program graph - 2 is used.
            timeout: The timeout in seconds for the SAT solver. Defaults to 60 seconds.

        Returns:
            A dictionary of SAT results. The keys are the number of swap layers considered at each
            step of the binary search and the corresponding value is a result from the SAT
            solver.
        """
        n = program_graph.number_of_nodes()
        if start is None:
            # use the maximum degree of the program graph - 2 as the lower bound.
            start = max([d for _, d in program_graph.degree]) - 2
        if last is None:
            last = n - 2

        variable_pool = IDPool(start_from=1)
        x = np.array(
            [[variable_pool.id(f"v_{i}_{j}") for j in range(n)] for i in range(n)], dtype=int
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(x)}
        ret = {}

        def interrupt(solver):
            # This function is called to interrupt the solver when the timeout is reached.
            solver.interrupt()

        # Make a cnf for the one-to-one mapping constraint
        cnf1 = []
        for i in range(n):
            clause = x[i, :].tolist()
            cnf1.append(clause)
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])
        for j in range(n):
            clause = x[:, j].tolist()
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])

        # Perform a binary search to find the minimum number of layers required to map the program
        while start < last:
            mid = (start + last) // 2
            # Build the connectivity matrix.
            connectivity_matrix = SATLayoutForCommutingGateRouter._build_connectivity_matrix(
                swap_strategy, mid
            )
            # Build the adjacency constraints and add them to the one-to-one mapping constraints.
            cnf2 = cnf1 + SATLayoutForCommutingGateRouter._build_adjacency_cnf(
                program_graph, connectivity_matrix, x
            )
            # Convert the constraints to a CNF object.
            cnf = CNF(from_clauses=cnf2)

            # Solve the SAT problem.
            with Solver(bootstrap_with=cnf, use_timer=True) as solver:
                timer = Timer(timeout, interrupt, [solver])
                timer.start()
                status = solver.solve_limited(expect_interrupt=True)
                sol = solver.get_model()
                e_time = solver.time()
                timer.cancel()
                if status:
                    # If the SAT problem is satisfiable, convert the solution to a mapping.
                    mapping = [vid2mapping[idx] for idx in sol if idx > 0]
                    ret[mid] = SATResult(status, sol, mapping, e_time)
                    last = mid
                else:
                    mapping = []
                    if sol is not None:
                        mapping = [vid2mapping[idx] for idx in sol if idx > 0]
                    ret[mid] = SATResult(status, sol, mapping, e_time)
                    start = mid + 1
        return ret
