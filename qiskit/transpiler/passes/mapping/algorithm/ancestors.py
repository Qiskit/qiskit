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

    def __init__(self, G: nx.DiGraph, max_depth: int = 5):
        self._graph = G.reverse()
        self._max_depth = max_depth
        self._depth = 0

    def ancestors(self, n: int) -> set:
        """
        Ancestor nodes of the node `n`
        Args:
            n: Index of the node in the `self._graph`
        Returns:
            Set of indices of the ancestor nodes.
        """
        self._depth = 0
        return self._ancestors_rec(n)

    @lru_cache(maxsize=CACHE_SIZE)
    def _ancestors_rec(self, n: int) -> set:
        if self._depth >= self._max_depth:
            return self._ancestors_loop(n)
        self._depth += 1
        ret = set()
        for n_succ in self._graph.successors(n):
            ret.add(n_succ)
            ret.update(self._ancestors_rec(n_succ))
        self._depth -= 1
        return ret

    @lru_cache(maxsize=CACHE_SIZE)
    def _ancestors_loop(self, n: int) -> set:
        ret = set()
        done = set()
        cands = [n]
        while cands:
            node = cands.pop()
            for v in self._graph.successors(node):
                if v not in done:
                    ret.add(v)
                    cands.append(v)
            done.add(node)
        return ret
