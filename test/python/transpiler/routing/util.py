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
"""Utility functions for the routing tests"""

from typing import List, TypeVar

import networkx as nx
from unittest2 import TestCase

from qiskit.transpiler.routing import Swap

_V = TypeVar('_V')


def valid_parallel_swaps(tester: TestCase, swaps: List[List[Swap[_V]]]) -> None:
    """Tests if the given sequence of swaps does not perform multiple swaps
    on the same node at the same time."""
    for step in swaps:
        used = {sw1 for sw1, _ in step} | {sw2 for _, sw2 in step}
        tester.assertEqual(len(used), 2 * len(step))


def valid_edge_swaps(tester: TestCase, swaps: List[List[Swap[_V]]],
                     valid_graph: nx.Graph) -> None:
    """Check if the swap is in the edge list."""

    for i in swaps:
        for j in i:
            tester.assertTrue(valid_graph.has_edge(*j), 'edge ' + str(j) + ' is not valid')
