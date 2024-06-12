// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

// from Shelly's PR: consolidate later
fn _row_op(mat: &mut Array2<bool>, ctrl: usize, trgt: usize) {
    let row0 = mat.row(ctrl).to_owned();
    let mut row1 = mat.row_mut(trgt);
    row1.zip_mut_with(&row0, |x, &y| *x ^= y);
    println!("should mutate something");
}

/// Ceil the fraction ``numerator/denominator``
fn _ceil_usize_fraction(numerator: &usize, denominator: &usize) -> usize {
    let base: usize = numerator / denominator;
    if numerator % denominator != 0 {
        base + 1
    } else {
        base
    }
}

pub fn pmh_synth(matrix: ArrayView2<bool>, section_size: &usize) -> Vec<(usize, usize)> {
    let mut mat = matrix.to_owned();
    let mut circuit = lower_cnot_synth(&mut mat, section_size);
    let mut mat_t = mat.t().to_owned();
    println!("{:?}", mat_t);
    lower_cnot_synth(&mut mat_t, section_size)
        .iter()
        .for_each(|&el| circuit.push(el));
    println!("{:?}", mat_t);
    circuit
}

/// instead of transposing the matrix, allow setting the axis in this function
fn lower_cnot_synth(matrix: &mut Array2<bool>, section_size: &usize) -> Vec<(usize, usize)> {
    // The vector of CNOTs to be applied. Called ``circuit`` here for consistency with the paper.
    let mut circuit: Vec<(usize, usize)> = Vec::new();

    // get number of columns (same as rows) and the number of sections
    let n = matrix.raw_dim()[0];
    let num_sections = _ceil_usize_fraction(&n, section_size);

    // iterate over the columns
    for section in 0..num_sections {
        // store sub section row patterns here, which we saw already
        let mut patterns: HashMap<Array1<bool>, usize> = HashMap::new();

        // iterate over the rows (note we only iterate from the diagonal downwards)
        for row_idx in section..n {
            let pattern: Array1<bool> = matrix.row(row_idx).to_owned();

            // skip if the row is empty (i.e. all elements are false)
            if pattern.iter().all(|el| !el) {
                continue;
            }
            if patterns.contains_key(&pattern) {
                // store CX location
                circuit.push((patterns[&pattern], row_idx));
                // remove the row
                _row_op(matrix, patterns[&pattern], row_idx);
            } else {
                patterns.insert(pattern, row_idx);
            }
        }
    }

    circuit
}
