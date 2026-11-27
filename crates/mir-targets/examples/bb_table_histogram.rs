// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_mir_targets::bb::{self, measurement_table::PackedData, pauli::SinglePauli};

#[allow(clippy::print_stdout)]
pub fn main() {
    let table = bb::MeasurementTable::from_automorphisms(&bb::SHIFT_AUTOMORPHISMS_G);
    for pivot in SinglePauli::IN_ORDER {
        let mut histogram = [0; 5];
        for p in PackedData::all() {
            histogram[table.chain_len(p, pivot) as usize] += 1;
        }
        println!("Rotations for {}: {:?}", pivot, histogram);
    }
}
