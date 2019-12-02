# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The size-optimizing mapper inspired by the stochastic_swap mapper."""

import copy
from typing import Callable, Mapping, Iterable, Optional, List, Tuple, Dict

import networkx as nx
import numpy as np

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.size import SizeMapper, Reg, ArchNode, logger
from qiskit.transpiler.passes.mapping.size_simple_mapper import SimpleSizeMapper
from qiskit.transpiler.routing import Swap


class QiskitSizeMapper(SizeMapper[Reg, ArchNode]):
    """A mapper that is inspired by the stochastic_swap mapper."""

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 trials: int = 40,
                 seed: Optional[int] = None) -> None:
        super().__init__(arch_graph, arch_permuter)
        self.simple_mapper =\
            SimpleSizeMapper(arch_graph, arch_permuter)  # type: SimpleSizeMapper[Reg, ArchNode]
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
        inv_trial_layout = {v: k for k, v in trial_layout.items()}  # type: Mapping[ArchNode, Reg]

        # Compute Sergey's randomized distance.
        # IDEA: Rewrite to numpy matrix
        x = {}  # type: Dict[ArchNode, Dict[ArchNode, float]]
        for i in self.arch_graph.nodes:
            x[i] = {}
        for i in self.arch_graph.nodes:
            for j in self.arch_graph.nodes:
                scale = 1 + np.random.normal(0, 1 / self.arch_graph.number_of_nodes())
                x[i][j] = scale * self.distance[i][j] ** 2
                x[j][i] = x[i][j]

        def cost(layout: Mapping[Reg, ArchNode]) -> float:
            """Compute the objective cost function."""
            return sum([x[layout[binop.qargs[0]]][layout[binop.qargs[1]]] for binop in binops])

        def swap(node0: ArchNode, node1: ArchNode) \
                -> Tuple[Mapping[Reg, ArchNode], Mapping[ArchNode, Reg]]:
            """Swap qarg0 and qarg1 based on trial layout and inv_trial layout.
            
            Supports partial mappings."""
            inv_new_layout = dict(inv_trial_layout)
            qarg0 = inv_new_layout.pop(node0, None)  # type: Optional[Reg]
            qarg1 = inv_new_layout.pop(node1, None)  # type: Optional[Reg]
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
