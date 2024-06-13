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

use hashbrown::HashMap;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::cmp;

// from Shelly's PR: consolidate later
fn _add(mat: &mut Array2<bool>, axis: Axis, ctrl: usize, trgt: usize) {
    let row0 = mat.index_axis(axis, ctrl).to_owned();
    let mut row1 = mat.index_axis_mut(axis, trgt);
    row1.zip_mut_with(&row0, |x, &y| *x ^= y);
}

fn _index(axis: Axis, i: &usize, j: &usize) -> (usize, usize) {
    if axis.index() == 0 {
        (*i, *j)
    } else {
        (*j, *i)
    }
}

pub fn pmh_synth(
    matrix: ArrayView2<bool>,
    section_size: &usize,
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut mat = matrix.to_owned();
    let lower_cnots = lower_cnot_synth(&mut mat, section_size, &0);
    // let mut mat_t = mat.t().to_owned();
    let upper_cnots = lower_cnot_synth(&mut mat, section_size, &1);
    (lower_cnots, upper_cnots)
}

/// instead of transposing the matrix, allow setting the axis in this function
fn lower_cnot_synth(
    matrix: &mut Array2<bool>,
    section_size: &usize,
    axis: &usize,
) -> Vec<(usize, usize)> {
    // The vector of CNOTs to be applied. Called ``circuit`` here for consistency with the paper.
    let mut circuit: Vec<(usize, usize)> = Vec::new();
    let cutoff = 1;

    // to apply to the transposed matrix, we can just set axis = 1
    let row_axis = Axis(*axis);
    // let col_axis = Axis((*axis + 1) % 2);

    // get number of columns (same as rows) and the number of sections
    let n = matrix.raw_dim()[0];
    let num_sections = n / section_size + 1;

    // iterate over the columns
    for section in 1..num_sections {
        // store sub section row patterns here, which we saw already
        let mut patterns: HashMap<Array1<bool>, usize> = HashMap::new();
        let section_slice = s![(section - 1) * section_size..cmp::min(section * section_size, n)];

        // iterate over the rows (note we only iterate from the diagonal downwards)
        for row_idx in (section - 1) * section_size..n {
            let pattern: Array1<bool> = matrix
                .index_axis(row_axis, row_idx)
                .slice(section_slice)
                .to_owned();
            // let pattern: Array1<bool> = matrix.row(row_idx).slice(section_slice).to_owned();

            // skip if the row is empty (i.e. all elements are false)
            if pattern.iter().any(|&el| el) {
                if patterns.contains_key(&pattern) {
                    // store CX location
                    circuit.push((patterns[&pattern], row_idx));
                    // remove the row
                    _add(matrix, row_axis, patterns[&pattern], row_idx);
                } else {
                    patterns.insert(pattern, row_idx);
                }
            }
        }

        // gaussian eliminate the rest
        for col_idx in (section - 1) * section_size..section * section_size {
            let mut diag_el = true;
            if !matrix[[col_idx, col_idx]] {
                diag_el = false; // TODO why not just diag_el = state[col, col]
            }

            for r in col_idx + 1..n {
                // if matrix[[r, col_idx]] {
                if matrix[_index(row_axis, &r, &col_idx)] {
                    if !diag_el {
                        _add(matrix, row_axis, r, col_idx); // remove row with index col_idx
                        circuit.push((r, col_idx));
                        diag_el = true
                    }
                    _add(matrix, row_axis, col_idx, r); // remove row with index r
                    circuit.push((col_idx, r));
                }
                // check if the logical and between the two target rows has more ``true`` elements
                // than ``cutoff``
                if matrix
                    .index_axis(row_axis, col_idx)
                    .iter()
                    .zip(matrix.index_axis(row_axis, r).iter())
                    .map(|(&i, &j)| i & j)
                    .filter(|&x| x)
                    .count()
                    > cutoff
                {
                    _add(matrix, row_axis, r, col_idx);
                    circuit.push((r, col_idx));
                }
            }
        }
    }
    circuit
}
