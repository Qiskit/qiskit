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
Utility for speeding up a function to find ancestors in DAG.
"""

from functools import lru_cache

import networkx as nx

CACHE_SIZE = 2 ** 10


class Ancestors:
    """
    Utility class for speeding up a function to find ancestors in DAG.
    """

    def __init__(self, G: nx.DiGraph):
        self._graph = G.reverse()

    @lru_cache(maxsize=CACHE_SIZE)
    def ancestors(self, n: int) -> set:
        """
        Ancestor nodes of the node `n`
        Args:
            n: Index of the node in the `self._graph`
        Returns:
            Set of indices of the ancestor nodes.
        """
        ret = set()
        for n_succ in self._graph.successors(n):
            if n_succ in ret:
                continue
            ret.add(n_succ)
            ret.update(self.ancestors(n_succ))
        return ret
