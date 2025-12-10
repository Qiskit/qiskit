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
use num_complex::Complex64;
use qiskit_quantum_info::sparse_observable::BitTerm;
use rustworkx_core::coloring::{ColoringStrategy, greedy_node_color_with_coloring_strategy};
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::{Graph, Undirected};
use std::convert::Infallible;
use std::usize;

// Helper struct for pauli evolution ease of use
#[derive(Clone, Debug)]
pub struct StrSparseTerm<'a> {
    pub terms: String,
    pub coeff: Complex64,
    pub indices: &'a [u32],
}

pub fn bit_term_as_char(bit_term: &BitTerm) -> char {
    match bit_term {
        BitTerm::X => 'X',
        BitTerm::Plus => '+',
        BitTerm::Minus => '-',
        BitTerm::Y => 'Y',
        BitTerm::Right => 'r',
        BitTerm::Left => 'l',
        BitTerm::Z => 'Z',
        BitTerm::Zero => '0',
        BitTerm::One => '1',
    }
}

pub fn evolution(order: u32, mut paulis: Vec<StrSparseTerm>) -> Vec<StrSparseTerm> {
    return match order {
        1 => paulis,
        2 => {
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
            let mut inner = paulis.clone();

            paulis.iter_mut().for_each(|view| {
                view.coeff.re = view.coeff.re * reduction as f64;
            });
            paulis = evolution(order - 2, paulis);
            paulis = paulis
                .iter()
                .cycle()
                .take(paulis.len() * 2)
                .map(|v| v.clone())
                .collect();
            let mut outer = paulis.clone();

            inner.iter_mut().for_each(|view| {
                view.coeff.re = view.coeff.re * (1.0 - 4.0 * reduction) as f64;
            });
            inner = evolution(order - 2, inner);

            paulis.append(&mut inner);
            paulis.append(&mut outer);

            paulis
        }
    };
}

pub fn reorder_terms<'a>(terms: &'a Vec<StrSparseTerm<'a>>) -> Vec<StrSparseTerm<'a>> {
    let sorted: Vec<&StrSparseTerm> = terms
        .iter()
        .sorted_by_key(|view| (view.indices, view.terms.clone()))
        .collect();

    let mut graph: Graph<&StrSparseTerm, Option<u8>, Undirected> = Graph::new_undirected();

    for term in sorted {
        graph.add_node(term);
    }

    for combination in (0..graph.node_count()).combinations(2 as usize) {
        let (indices1, indices2) = combination
            .iter()
            .map(|i| {
                let index = *i as u32;
                (
                    index,
                    graph
                        .node_weight(index.into())
                        .unwrap()
                        .indices
                        .iter()
                        .collect::<HashSet<&u32>>(),
                )
            })
            .collect_tuple()
            .expect("Expected a combination of two values");

        if indices1.1.intersection(&indices2.1).count() > 0 {
            graph.add_edge(indices1.0.into(), indices2.0.into(), None);
        }
    }

    let callback = |_: NodeIndex| -> Result<Option<usize>, Infallible> { Ok(None) };
    let colors =
        greedy_node_color_with_coloring_strategy(&graph, callback, ColoringStrategy::Saturation)
            .expect("An error ocurred while coloring Pauli sparse terms");
    let mut colors_map: IndexMap<usize, Vec<&NodeIndex>> = IndexMap::new();

    for (node_index, color) in colors.iter().sorted() {
        if !colors_map.contains_key(color) {
            colors_map.insert(*color, Vec::new());
        }
        colors_map.get_mut(color).unwrap().push(node_index);
    }

    colors_map
        .iter()
        .map(|(_, node_index)| node_index)
        .flatten()
        .map(|index| *graph.node_weight(**index).unwrap())
        .map(|term| term.clone())
        .collect()
}
