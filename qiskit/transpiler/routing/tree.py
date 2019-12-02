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

"""Implementations for routing with matchings permutation on a tree

Terminology used in comments:

Proper and Improper nodes
: A node is proper when it has reached its destination tree. If not, it is improper.

Pure tree
: A tree is pure when all nodes in that tree are proper.
"""

from typing import TypeVar, List, Iterator, Tuple, Dict, Set, Generic, Optional

import networkx as nx

from qiskit.transpiler.routing import Permutation, Swap, util

_V = TypeVar('_V')


class Pebble(Generic[_V]):
    """A pebble represents a the piece of data at a node that needs to be routed."""

    def __init__(self, destination: _V) -> None:
        self.destination = destination
        self._cached_tree = None  # type: Optional['Tree']
        self._cached_purity = False

    def is_proper(self, current_tree: 'Tree') -> bool:
        """Decide if the node's destination is inside the tree.

        Args:
          current_tree: The tree the pebble is currently located in.

        Returns:
          Whether the node is proper.

        """
        if self._cached_tree != current_tree:
            self._cached_tree = current_tree
            self._cached_purity = self.destination in current_tree.graph.nodes

        return self._cached_purity


class RootPebble(Pebble):
    """A pebble that was created when the root tree became pure.
    
    This pebble becomes impure when it has no impure children;
    it becomes pure when it is in the root tree.
    """

    def is_proper(self, current_tree: 'Tree') -> bool:
        """Decide if the node's destination is inside the tree.
        
         For the root destination pebble we always recompute the value,
         since it can change depending on other pebbles in the tree.
         And because there is only one RootPebble this stays cheap.

        Args:
          current_tree: The tree the pebble is currently located in.

        Returns:
          Whether the node is proper.

        """
        if self.destination in current_tree.graph.nodes:
            return True

        if all(child.is_proper(current_tree) for child in current_tree.children(self)):
            # If all its children are proper, then we can make this pebble improper
            # since it wont displace any impure pebble above it nor can it block an improper pebble
            # below it from moving up.
            # This also takes care of the situation where the tree is pure.
            return False

        return True


class Tree(Generic[_V]):
    """A tree graph wrapper that specifies the root node."""

    def __init__(self, root: _V, graph: nx.DiGraph, pebbles: Dict[_V, Pebble[_V]]) -> None:
        """Construct a tree object.

        Args:
            root: The root node of this tree.
            graph: A directed graph representing parent-to-children relations.
            pebbles: The pebbles located on this tree.
        """
        self.root = root
        self.graph = graph
        self.pebbles = pebbles
        self._has_changed = True

    def is_pure(self) -> bool:
        """Checks if all nodes in this tree are proper."""
        # IDEA: Cache purity.
        return all(pebble.is_proper(self) for pebble in self.pebbles.values())

    def is_proper(self, node: _V) -> bool:
        """Check if a node in this tree is proper."""
        return self.pebbles[node].is_proper(self)

    def children(self, pebble: Pebble[_V]) -> Iterator[Pebble[_V]]:
        """Find the child pebbles of the given node."""
        # Possibly slow inverse lookup.
        node = next(filter(lambda vertex: self.pebbles[vertex] == pebble, self.pebbles))
        return (self.pebbles[successor] for successor in self.graph.successors(node))

    def root_swap(self, other: 'Tree') -> None:
        """Swap the pebbles between this tree's root and the other's root."""
        self.pebbles[self.root], other.pebbles[other.root] = \
            other.pebbles[other.root], self.pebbles[self.root]
        self._has_changed = True
        other._has_changed = True

    def apply_internal_swaps(self, swaps: List[Swap[_V]]) -> None:
        """Take a list of swaps and apply the swaps that are internal to this tree."""
        for sw1, sw2 in swaps:
            if sw1 in self.pebbles and sw2 in self.pebbles:
                self.pebbles[sw2], self.pebbles[sw1] = self.pebbles[sw1], self.pebbles[sw2]
                self._has_changed = True

    def move_improper(self) -> Iterator[List[Swap[_V]]]:
        """Coroutine that lists of swaps that move improper vertices up towards the tree root.
        
        Every yield, takes in the current new permutation, a new ignore_root parameter,
        and a new tr_was_pure parameter.
        
        tr_was_pure indicates whether the tree Tr subgraph containing the root was pure.
        """

        # Make sets containing even and odd nodes
        even_nodes = {self.root}  # type: Set[_V]
        odd_nodes = set()  # type: Set[_V]
        for current, successors in nx.bfs_successors(self.graph, self.root):
            if current in odd_nodes:
                # All the following nodes to an odd node are even.
                even_nodes.update(successors)
            else:
                # All the following nodes to an even node are odd.
                odd_nodes.update(successors)

        def swap_up(nodes: Set[_V]) -> List[Swap[_V]]:
            """Compute a swap step for nodes with their parents if that improves the pebbles."""
            swaps = []
            for parent in nodes:
                if self.is_proper(parent):
                    try:
                        # Pick any improper child.
                        swapme = next(filter(lambda n: not self.is_proper(n),
                                             self.graph.successors(parent)))
                        swaps.append((parent, swapme))
                    except StopIteration:
                        # Do nothing, there are no children that are improper.
                        pass
            return swaps

        even_swaps = None
        odd_swaps = None
        while True:
            # Only compute the swaps when the tree has changed.
            # Otherwise swap_up will yield the empty list anyway.
            if self._has_changed:
                # If the tree has changed, delete the cached swaps.
                even_swaps = None
                odd_swaps = None
                self._has_changed = False

            # When both previous iterations were futile, skip running swap_up
            if even_swaps == [] and odd_swaps == []:
                yield []
            else:
                even_swaps = swap_up(even_nodes)
                yield even_swaps

            if self._has_changed:
                even_swaps = None
                odd_swaps = None
                self._has_changed = False

            if even_swaps == [] and odd_swaps == []:
                yield []
            else:
                odd_swaps = swap_up(odd_nodes)
                yield odd_swaps


def permute(graph: nx.Graph,
            perm: Permutation[_V]) -> Iterator[List[Swap[_V]]]:
    """List swaps that implement the given permutation on a graph repressenting a tree.
    
    The implementation is still fairly slow.
    
    SEE: Algorithm by Louxin Zhang: https://doi.org/10.1137/S0895480197323159
    """
    root = centroid(graph)  # type: _V
    tree_graph = nx.dfs_tree(graph, root)
    return permute_tree(Tree(root, tree_graph, {}), perm)


def permute_tree(tree: Tree, perm: Permutation[_V]) -> Iterator[List[Swap[_V]]]:
    """List swaps that implement the given permutation on a tree.
    
    The implementation is still fairly slow.
    
    SEE: Algorithm by Louxin Zhang: https://doi.org/10.1137/S0895480197323159
    """
    # Empty graph or single node (base case)
    if tree.graph.number_of_nodes() <= 1:
        return iter([])
    elif tree.graph.number_of_nodes() == 2:  # 2-node base case that needs to be handled.
        node1, node2 = tree.graph.nodes
        if perm[node1] == node2:
            return iter([[(node1, node2)]])
        return iter([])

    permutation = perm.copy()
    # We must regenerate the pebbles so that they are not influenced by previous iterations.
    partitioning, root_tree = partition(tree,
                                        pebbles={i: Pebble(perm[i]) for i in tree.graph.nodes})
    partitioning.append(root_tree)

    ###
    # Phase 1: Move improper pebbles up to the subtree roots
    ###
    tree_movers = {t: t.move_improper() for t in partitioning}
    all_swaps = []  # type: List[List[Swap[_V]]]
    phase1_length = max(sum(1 for pebble in i.pebbles.values() if pebble.is_proper(i))
                        for i in partitioning)
    for i in range(phase1_length):
        # Do an even and an odd step.
        for _ in range(2):
            # All tree swaps are disjoint so we can simply concatenate them.
            disjoint_swap_step = [(subtree, next(mover)) for subtree, mover in tree_movers.items()]
            # Update the trees.
            for subtree, swaps in disjoint_swap_step:
                subtree.apply_internal_swaps(swaps)
            # Flatten the swaps into one list.
            swap_step = [swap for subtree, swaps in disjoint_swap_step for swap in swaps]
            # Apply the swap step to the permutation
            util.swap_permutation([swap_step], permutation)
            all_swaps.append(swap_step)

    ###
    # Phase 2: Start swapping pebbles across the root to their destination trees.
    # But continue moving improper pebbles up the tree in parallel.
    ###
    while any(not tree.is_pure() for tree in partitioning):
        target_tree = None  # type: Optional[Tree]
        root_pebble = root_tree.pebbles[root_tree.root]
        if isinstance(root_pebble, RootPebble) or root_pebble.is_proper(root_tree):
            # The root node must be proper for the root tree to be pure and it is cheaper to check.
            if isinstance(root_tree.pebbles[root_tree.root], RootPebble) or root_tree.is_pure():
                # Tree Tr has become pure. Replace the pebble at the root with a special RootPebble.
                root_pebble = root_tree.pebbles[root_tree.root]
                root_tree.pebbles[root_tree.root] = RootPebble(root_pebble.destination)
                try:
                    # Try to pick a mixed tree with an improper root node. Pick that as target
                    # This is almost always the case.
                    target_tree = next(
                        filter(lambda t: not t.pebbles[t.root].is_proper(t), partitioning))
                except StopIteration:
                    # It can happen that there is no target tree with an improper root node,
                    # if the last two root swaps occured on even, then odd timesteps
                    # and there are only two impure trees left.
                    # Then we just wait one more timestep to allow the trees to move improper nodes.
                    target_tree = None
            else:
                # The root node is proper, but Tr is not pure.
                # So the root node must be moved internally.
                target_tree = None
        else:
            # Find the tree that the pebble at the root needs to go to.
            target_tree = next(filter(lambda t: root_pebble.destination in t.graph.nodes,
                                      partitioning))
            # But only swap the pebble there if the target tree's root is impure.
            subroot_pebble = target_tree.pebbles[target_tree.root]
            if subroot_pebble.is_proper(target_tree):
                target_tree = None

        # Compute the swaps that move nodes upwards in subtrees
        # Note this will be exactly one time step.
        move_up_swaps = [(subtree, next(mover))
                         for subtree, mover in tree_movers.items()]
        time_step = []  # type: List[Tuple[Tree, List[Swap[_V]]]]
        root_swap = None  # type: Optional[Swap[_V]]
        if target_tree is not None:
            # Remove any swaps interfering with the root swap because it is occupied
            time_step = [(subtree, [swap for swap in swaps
                                    if swap[0] != target_tree.root
                                    or swap[1] != target_tree.root
                                    or swap[0] != root_tree.root
                                    or swap[1] != root_tree.root])
                         for subtree, swaps in move_up_swaps]
            root_swap = (root_tree.root, target_tree.root)
            root_tree.root_swap(target_tree)
        else:
            # There was no root swap; just move pebbles upwards.
            root_swap = None
            time_step = move_up_swaps

        # Apply swaps and store
        for subtree, swaps in time_step:
            subtree.apply_internal_swaps(swaps)
        # Extract the swaps from time_stap and flatten the list.
        time_step_swaps = [swap for _, swaps in time_step for swap in swaps]
        if root_swap is not None:
            time_step_swaps.append(root_swap)

        util.swap_permutation([time_step_swaps], permutation)
        all_swaps.append(time_step_swaps)

    ###
    # Phase 3: Route each subtree in parallel.
    ###
    all_swaps.extend(util.flatten_swaps(permute_tree(t, permutation) for t in partitioning))

    return util.optimize_swaps(all_swaps)


def partition(tree: Tree,
              pebbles: Dict[_V, Pebble[_V]]) -> Tuple[List[Tree[_V]], Tree[_V]]:
    """Partition a tree graph into s+1 similarly-sized trees.

    Args:
      tree: The input tree to partition into subtrees according to a centroid.
      pebbles: The pebbles to place on the nodes in the subtrees.

    Returns:
      A tuple containing the first s trees and the special s+1'th tree,
      which contains the centroid node of the input graph.

    """
    subtrees = []  # type: List[Tree]
    for adj_root in tree.graph.adj[tree.root]:
        # Create a subgraph consisting of the nodes reachable from adj_root.
        subtree = tree.graph.subgraph(nx.dfs_preorder_nodes(tree.graph, adj_root))
        sub_pebbles = {node: pebbles[node] for node in subtree.nodes}
        subtrees.append(Tree(adj_root, subtree, sub_pebbles))

    # Find an s such that1+Σ_{j=s+1}^d |T_j| ≤ |T_1| < 1 + Σ_{j=s}^d |T_j|
    # Straightforward O(d) linear search.
    # IDEA: Binary search for s.
    current_sum = 0
    root_subtrees = len(subtrees)  # The parameter s.
    while current_sum + 1 <= subtrees[0].graph.number_of_nodes():
        root_subtrees -= 1
        current_sum += subtrees[root_subtrees].graph.number_of_nodes()
    # Constuct subtree Tr, which consists of the subtree
    # induced by nodes in {T_{s+1}, ..., T_d, root}
    tr_nodes = {tree.root} | {node
                              for subtree in subtrees[root_subtrees + 1:]
                              for node in subtree.graph.nodes}
    tr_tree = tree.graph.subgraph(tr_nodes)
    # We now have partitioning = {T_1, T_2, ... , T_s, T_r} where |T_i| ≤ |T_{i+1}| for 1≤i≤s-1.
    root_tree = Tree(root=tree.root, graph=tr_tree,
                     pebbles={i: p for i, p in pebbles.items() if i in tr_nodes})
    return subtrees[0:root_subtrees + 1], root_tree


def centroid(tree: nx.Graph) -> _V:
    """Find the centroid of a tree graph."""
    subtree_sizes = dict()  # type: Dict[_V, int]
    current_centroid, *_ = tree.nodes

    # Compute sizes of subtrees.
    visited = {current_centroid}  # type: Set[_V]

    def compute_subtree_size(root: _V) -> int:
        """Recursively add size of subtree rooted at root to subtree_sizes and return its size."""
        visited.add(root)
        # Sum default to 0 for empty sequences (leaves)
        subtree_size = sum((compute_subtree_size(adjacent_node)
                            for adjacent_node in tree.adj[root]
                            if adjacent_node not in visited)) + 1
        subtree_sizes[root] = subtree_size
        return subtree_size

    subtree_sizes[current_centroid] = compute_subtree_size(current_centroid)

    # Find the centroid using the subtree sizes.
    # We keep moving the centroid if it does not at least divide the graph in half (<n/2)
    visited = {current_centroid}

    def is_centroid(node: _V) -> bool:
        """Check if the given node v is a centroid or not."""
        other_subtree_sizes = (subtree_sizes[adjacent_node] for adjacent_node in tree.adj[node]
                               if adjacent_node not in visited)
        # ceiling integer division of tr.nodes (⌈n/2⌉)
        return max(other_subtree_sizes) <= tree.number_of_nodes() / 2

    while not is_centroid(current_centroid):
        visited.add(current_centroid)
        candidates = (adjacent_node for adjacent_node in tree.adj[current_centroid]
                      if adjacent_node not in visited)
        # Go to the candidate with the largest subtree.
        current_centroid = max(candidates, key=lambda c: subtree_sizes[c])
    return current_centroid
