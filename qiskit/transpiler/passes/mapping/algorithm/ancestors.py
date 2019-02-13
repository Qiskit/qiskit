# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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

    def __init__(self, G: nx.DiGraph, max_depth=5):
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
    def _ancestors_rec(self, n) -> set:
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
    def _ancestors_loop(self, n) -> set:
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
