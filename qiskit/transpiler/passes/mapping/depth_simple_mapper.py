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

"""A simple depth-optimizing mapper"""

from typing import Mapping, Set, FrozenSet, Dict

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper


class SimpleDepthMapper(DepthMapper[Reg, ArchNode]):
    """This mapper places gates from the first layer arbitrarily on a maximum matching
    of the architecture."""
    def map(self, circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode] = None  # pylint: disable=unused-argument
            ) -> Mapping[Reg, ArchNode]:
        """Try to map as many two-qubit gates to a maximum matching as possible.
        
        Note: Does not take into account the scoring function, nor the weights on the graph."""
        binops = Mapper._binops_circuit(circuit)
        matching = Mapper.construct_matching(self.arch_graph)  # type: Set[FrozenSet[ArchNode]]
        # First assign the two-qubit gates, because they are restricted by the architecture
        mapping = {}  # type: Dict[Reg, ArchNode]
        for binop in binops:
            if matching:
                # pick an available matching and map this operation to that matching
                node0, node1 = matching.pop()
                mapping[binop.qargs[0]] = node0
                mapping[binop.qargs[1]] = node1
            else:
                # no more matchings
                break

        return mapping
