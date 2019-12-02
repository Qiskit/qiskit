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

"""The depth-optimizing mapper inspired by the stochastic_swap mapper"""

from typing import Callable, Mapping, Iterable, List

import networkx as nx

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.size_qiskit_mapper import QiskitSizeMapper
from qiskit.transpiler.routing import Swap, util


class QiskitDepthMapper(DepthMapper[Reg, ArchNode]):
    """This mapper will try to place a layer of gates in a similar manner to the stochast_swap
    mapper."""

    def __init__(self,
                 arch_graph: nx.Graph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[List[Swap[ArchNode]]]]) -> None:
        """Construct a QiskitDepthMapper"""
        super().__init__(arch_graph, arch_permuter)
        self.qiskit_size_mapper = \
            QiskitSizeMapper(arch_graph.to_directed(as_view=True),
                             # Convert depth permuter to size permuter
                             # This is only used for SimpleSizeMapper,
                             # so it's fine.
                             util.sequential_permuter(arch_permuter)
                             )  # type: QiskitSizeMapper[Reg, ArchNode]

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Map gates from the circuit to the architecture. """
        # We call the size_map function to avoid size-based preconditions.
        binops = Mapper._binops_circuit(circuit)
        return self.qiskit_size_mapper.size_map(circuit, current_mapping, binops)
