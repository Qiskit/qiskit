// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use fixedbitset::FixedBitSet;
use ndarray::{Array2, ArrayViewMut1, Axis};
use rayon_cond::CondIterator;
use rustworkx_core::petgraph::visit::{IntoNeighbors, NodeCompactIndexable};

// The implementation of `distance_matrix` was forked from Rustworkx at its commit 30f29079eeae,
// from the file `src/shortest_path/distance_matrix.rs` (as `compute_distance_matrix`). Its licence
// terms are:
//
//      Licensed under the Apache License, Version 2.0 (the "License"); you may
//      not use this file except in compliance with the License. You may obtain
//      a copy of the License at
//
//          http://www.apache.org/licenses/LICENSE-2.0
//
//      Unless required by applicable law or agreed to in writing, software
//      distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//      WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
//      License for the specific language governing permissions and limitations
//      under the License.
//
// The implementation was modified to be generic over `petgraph::visit` traits, throw away the
// handling of `StableGraph`, minimise allocations in the `bfs_traversal` call, and use a bitmap
// data structure to track "seenness" rather than `HashSet`.

// This file may be obsoleted and/or upstreamed into Rustworkx in the future.

pub fn distance_matrix<G>(graph: G, parallel_threshold: usize, null_value: f64) -> Array2<f64>
where
    G: NodeCompactIndexable + IntoNeighbors,
{
    let n = graph.node_count();
    let neighbors = (0..n)
        .map(|index| {
            graph
                .neighbors(graph.from_index(index))
                .map(|neighbor| graph.to_index(neighbor))
                .collect::<FixedBitSet>()
        })
        .collect::<Vec<_>>();
    let bfs_traversal = |start: usize, mut row: ArrayViewMut1<f64>| {
        let mut distance = 0.0;
        let mut seen = FixedBitSet::with_capacity(n);
        let mut next = FixedBitSet::with_capacity(n);
        let mut cur = FixedBitSet::with_capacity(n);
        cur.put(start);
        while !cur.is_clear() {
            next.clear();
            for found in cur.ones() {
                row[[found]] = distance;
                next |= &neighbors[found];
            }
            seen.union_with(&cur);
            next.difference_with(&seen);
            distance += 1.0;
            ::std::mem::swap(&mut cur, &mut next);
        }
    };
    let mut out = Array2::from_elem((n, n), null_value);
    CondIterator::new(out.axis_iter_mut(Axis(0)), n >= parallel_threshold)
        .enumerate()
        .for_each(|(index, row)| bfs_traversal(index, row));
    out
}
