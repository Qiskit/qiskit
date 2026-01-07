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

use ahash::HashSet;
use indexmap::IndexMap;
use itertools::Itertools;
use qiskit_quantum_info::sparse_observable::SparseTermView;
use rustworkx_core::coloring::{ColoringStrategy, greedy_node_color_with_coloring_strategy};
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::{Graph, Undirected};
use std::convert::Infallible;
use std::usize;

pub fn evolution<'a, 'b>(order: u32, paulis: &'b [SparseTermView<'a>]) -> Vec<SparseTermView<'a>> {
    return match order {
        1 => paulis.to_vec(),
        2 => {
            let mut paulis = paulis.to_vec();

            let last = paulis.pop();

            paulis.iter_mut().for_each(|view| {
                view.coeff.re = view.coeff.re / 2.0;
            });

            let mut outer = paulis.clone();
            outer.reverse();

            if !last.is_none() {
                paulis.append(&mut vec![last.unwrap()]);
            }

            paulis.append(&mut outer);

            paulis
        }
        _ => {
            let reduction = 1.0 / (4.0 - 4_f64.powf(1.0 / (order as f64 - 1.0)));
            let mut paulis = paulis.to_vec();
            let mut inner = paulis.clone();

            paulis.iter_mut().for_each(|view| {
                view.coeff.re = view.coeff.re * reduction as f64;
            });
            let mut paulis = evolution(order - 2, &paulis);
            paulis = paulis
                .iter()
                .cycle()
                .take(paulis.len() * 2)
                .cloned()
                .collect();
            let mut outer = paulis.clone();

            inner.iter_mut().for_each(|view| {
                view.coeff.re = view.coeff.re * (1.0 - 4.0 * reduction) as f64;
            });
            let mut inner = evolution(order - 2, &inner);

            paulis.append(&mut inner);
            paulis.append(&mut outer);

            paulis
        }
    };
}

pub fn reorder_terms<'a>(
    terms: impl Iterator<Item = SparseTermView<'a>>,
) -> Result<Vec<SparseTermView<'a>>, &'a str> {
    let sorted: Vec<SparseTermView<'a>> = terms
        .sorted_by_key(|view| (view.indices, view.bit_terms))
        .collect();
    let edges: Vec<(usize, usize)> = (0..sorted.len())
        .combinations(2 as usize)
        .map(|combination| (combination[0], combination[1]))
        .filter(|(index1, index2)| {
            let (indices1, indices2) = (
                HashSet::from_iter(sorted[*index1].indices),
                HashSet::from_iter(sorted[*index2].indices),
            );
            indices1.intersection(&indices2).count() > 0
        })
        .collect();

    let mut graph: Graph<usize, Option<u8>, Undirected> =
        Graph::with_capacity(sorted.len(), edges.len());

    sorted.iter().enumerate().for_each(|(i, _)| {
        graph.add_node(i);
    });
    edges.iter().for_each(|(index1, index2)| {
        graph.add_edge((*index1 as u32).into(), (*index2 as u32).into(), None);
    });

    let callback = |_: NodeIndex| -> Result<Option<usize>, Infallible> { Ok(None) };
    match greedy_node_color_with_coloring_strategy(&graph, callback, ColoringStrategy::Saturation) {
        Ok(colors) => {
            let mut colors_map: IndexMap<usize, Vec<&NodeIndex>> = IndexMap::new();

            for (node_index, color) in colors.iter().sorted() {
                if !colors_map.contains_key(color) {
                    colors_map.insert(*color, Vec::new());
                }
                colors_map.get_mut(color).unwrap().push(node_index);
            }

            Ok(colors_map
                .iter()
                .map(|(_, node_index)| node_index)
                .flatten()
                .map(|index| sorted[graph[**index]])
                .collect())
        }

        Err(_) => Err("An error ocurred while coloring Pauli sparse terms"),
    }
}
