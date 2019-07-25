import copy
from typing import Callable, Mapping, Iterable, Optional, List, Tuple, Dict

import networkx as nx
import numpy as np

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.size import SizeMapper, Reg, ArchNode, logger
from qiskit.transpiler.passes.mapping.size_simple_mapper import SimpleSizeMapper
from qiskit.transpiler.routing import Swap


class QiskitSizeMapper(SizeMapper[Reg, ArchNode]):
    """A mapper that combines the Qiskit mapper and the extension size mapper."""

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 trials: int = 40,
                 seed: Optional[int] = None) -> None:
        super().__init__(arch_graph, arch_permuter)
        self.simple_mapper: SimpleSizeMapper[Reg, ArchNode] = SimpleSizeMapper(arch_graph, arch_permuter)
        self.trials = trials
        self.seed = seed

    def size_map(self, circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """A mapper based on qiskit.mapping.swap_mapper"""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Filter out registers that are not used by binary operations from the current mapping.
        binop_regs = {qarg for binop in binops for qarg in binop.qargs}
        binop_current_mapping = {k: v for k, v in current_mapping.items() if k in binop_regs}

        # Try to map everything using the qiskit mapper.
        # Begin loop over trials of randomized algorithm
        trial_layouts = (self._qiskit_trial(binops, binop_current_mapping)
                         for _ in range(self.trials))
        # Filter out None results
        filtered_layouts = (trial for trial in trial_layouts if trial is not None)
        try:
            # Minimize over size
            best_layout = min(filtered_layouts, key=lambda t: t[0])
            logger.debug("qiskit mapper: done")
            return best_layout[1]
        except ValueError:
            logger.debug("qiskit mapper: failed!")
            # The qiskit mapper did not find a mapping so we just map a single gate.
            return self.simple_mapper.size_map(circuit, current_mapping, binops)

    def _qiskit_trial(self,
                      binops: List[DAGNode],
                      initial_layout: Mapping[Reg, ArchNode]) \
            -> Optional[Tuple[int, Mapping[Reg, ArchNode]]]:
        """One trial in computing a mapping as used in qiskit.

        Tries to swap edges that reduce the cost function up to a maximimum size."""
        trial_layout = copy.copy(initial_layout)
        inv_trial_layout: Mapping[ArchNode, Reg] = {v: k for k, v in trial_layout.items()}

        # Compute Sergey's randomized distance.
        # IDEA: Rewrite to numpy matrix
        xi: Dict[ArchNode, Dict[ArchNode, float]] = {}
        for i in self.arch_graph.nodes:
            xi[i] = {}
        for i in self.arch_graph.nodes:
            for j in self.arch_graph.nodes:
                scale = 1 + np.random.normal(0, 1 / self.arch_graph.number_of_nodes())
                xi[i][j] = scale * self.distance[i][j] ** 2
                xi[j][i] = xi[i][j]

        def cost(layout: Mapping[Reg, ArchNode]) -> float:
            """Compute the objective cost function."""
            return sum([xi[layout[binop.qargs[0]]][layout[binop.qargs[1]]] for binop in binops])

        def swap(node0: ArchNode, node1: ArchNode) \
                -> Tuple[Mapping[Reg, ArchNode], Mapping[ArchNode, Reg]]:
            """Swap qarg0 and qarg1 based on trial layout and inv_trial layout.

            Supports partial mappings."""
            inv_new_layout = dict(inv_trial_layout)
            qarg0: Optional[Reg] = inv_new_layout.pop(node0, None)
            qarg1: Optional[Reg] = inv_new_layout.pop(node1, None)
            if qarg1 is not None:
                inv_new_layout[node0] = qarg1
            if qarg0 is not None:
                inv_new_layout[node1] = qarg0

            return {v: k for k, v in inv_new_layout.items()}, inv_new_layout

        # Loop over sizes up to a max size (nr of swaps) of |V|^2
        size = 0
        for _ in range(len(self.arch_graph.nodes) ** 2):
            # Find the layout which minimize the objective function
            # by trying all possible swaps.
            new_layouts = (swap(*edge) for edge in self.arch_graph.edges)
            min_layout = min(new_layouts, key=lambda t: cost(t[0]))

            # Were there any good choices?
            if cost(min_layout[0]) < cost(trial_layout):
                trial_layout, inv_trial_layout = min_layout
                size += 1
            else:
                # If there weren't any good choices, there also won't be in the future. So abort.
                break

        # Compute the coupling graph distance
        # If all gates can be applied now, we have found a layout.
        dist = sum(self.distance[trial_layout[binop.qargs[0]]][trial_layout[binop.qargs[1]]]
                   for binop in binops)
        if dist == len(binops):
            # We have succeeded in finding a layout
            return size, trial_layout
        return None