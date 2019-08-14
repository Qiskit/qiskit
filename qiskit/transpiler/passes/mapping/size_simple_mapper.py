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

"""The size-optimizing simple mapper"""

from typing import Mapping, List, Tuple

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.passes.mapping.size import SizeMapper, Reg, ArchNode


class SimpleSizeMapper(SizeMapper[Reg, ArchNode]):
    """This mapper will place a single gate every iteration, if necessary"""
    def size_map(self,
                 circuit: DAGCircuit,  # pylint: disable=unused-argument
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """Perform a simple greedy mapping of the cheapest gate to the architecture."""

        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.

        def simple_saved_gates(place: Tuple[Placement[Reg, ArchNode], DAGNode]) -> int:
            """We have to repackage the second argument of place into an iterable."""
            return self.saved_gates((place[0], [place[1]]))

        return self._inner_simple(binops, current_mapping, self.arch_graph,
                                  simple_saved_gates)[0].mapped_to
