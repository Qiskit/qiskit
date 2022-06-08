// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use pyo3::prelude::*;

fn marginalize<T: std::ops::AddAssign + Copy>(
    counts: HashMap<String, T>,
    indices: Option<Vec<usize>>,
) -> HashMap<String, T> {
    let mut out_counts: HashMap<String, T> = HashMap::with_capacity(counts.len());
    let clbit_size = counts
        .keys()
        .next()
        .unwrap()
        .replace(|c| c == '_' || c == ' ', "")
        .len();
    let all_indices: Vec<usize> = (0..clbit_size).collect();
    counts
        .iter()
        .map(|(k, v)| (k.replace(|c| c == '_' || c == ' ', ""), *v))
        .for_each(|(k, v)| match &indices {
            Some(indices) => {
                if all_indices == *indices {
                    out_counts.insert(k, v);
                } else {
                    let key_arr = k.as_bytes();
                    let new_key: String = indices
                        .iter()
                        .map(|bit| {
                            let index = clbit_size - *bit - 1;
                            match key_arr.get(index) {
                                Some(bit) => *bit as char,
                                None => '0',
                            }
                        })
                        .rev()
                        .collect();
                    out_counts
                        .entry(new_key)
                        .and_modify(|e| *e += v)
                        .or_insert(v);
                }
            }
            None => {
                out_counts.insert(k, v);
            }
        });
    out_counts
}

#[pyfunction]
pub fn marginal_counts(
    counts: HashMap<String, u64>,
    indices: Option<Vec<usize>>,
) -> HashMap<String, u64> {
    marginalize(counts, indices)
}

#[pyfunction]
pub fn marginal_distribution(
    counts: HashMap<String, f64>,
    indices: Option<Vec<usize>>,
) -> HashMap<String, f64> {
    marginalize(counts, indices)
}
