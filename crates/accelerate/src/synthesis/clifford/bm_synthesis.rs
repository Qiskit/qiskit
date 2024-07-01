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

use ndarray::ArrayView2;
use smallvec::smallvec;
use qiskit_circuit::operations::StandardGate;
use crate::synthesis::clifford::utils::{Clifford, CliffordGatesVec};

pub struct BmCliffordSynthesis<'a> {
    /// The Clifford tableau to be synthesized.
    tableau: ArrayView2<'a, bool>,

    /// The total number of qubits.
    num_qubits: usize,
}


/// Decompose a single-qubit clifford.
fn decompose_clifford_1q(tableau: ArrayView2<bool>) -> Result<(usize, CliffordGatesVec), String> {
    let mut clifford_gates = CliffordGatesVec::new();
    let destab_phase = tableau[[0, 2]];
    let stab_phase = tableau[[1, 2]];

    if destab_phase && !stab_phase {
        clifford_gates.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(0)]));
    }
    else if !destab_phase && stab_phase {
        clifford_gates.push((StandardGate::XGate, smallvec![], smallvec![Qubit(0)]));
    }
    else if destab_phase && stab_phase {
        clifford_gates.push((StandardGate::YGate, smallvec![], smallvec![Qubit(0)]));
    }

    let destab_x = tableau[[0, 0]];
    let destab_z = tableau[[0, 1]];
    let stab_x = tableau[[1, 0]];
    let stab_z = tableau[[1, 1]];

    if stab_z && !stab_x {
        if destab_z {
            clifford_gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(0)]));
        }
    }
    else if !stab_z && stab_x {
        if destab_x {
            clifford_gates.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(0)]));
        }
        clifford_gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(0)]));
    }
    else {
        if !destab_z {
            clifford_gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(0)]));
        }
        clifford_gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(0)]));
        clifford_gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(0)]));
    }
    Ok((1, clifford_gates))
}

/// Two-qubit cost reduction step.
fn reduce_cost() {

}

/// Return rank of 2x2 boolean matrix.
fn rank2(a: bool, b: bool, c: bool, d: bool) -> usize {
    if (a & d) ^ (b & c) {
        2
    }
    else if a | b | c | d {
        1
    }
    else {
        0
    }
}

/// Return the number of CX gates required for Clifford decomposition.
fn cx_cost(clifford: &Clifford) -> usize {
    if clifford.num_qubits == 2 {
        cx_cost2(clifford)
    }
    else {
        cx_cost3(clifford)
    }
}

/// Return CX cost of a 2-qubit clifford.
fn cx_cost2(clifford: &Clifford) -> usize {
    let r00 = rank2(clifford.tableau[[0, 0]], clifford.tableau[[0, 2]], clifford.tableau[[2, 0]], clifford.tableau[[2, 2]]);
    let r01 = rank2(clifford.tableau[[0, 1]], clifford.tableau[[0, 3]], clifford.tableau[[2, 1]], clifford.tableau[[2, 3]]);
    if r00 == 2 {
        r01
    }
    else {
        r01 + 1 - r00
    }
}

/// Return CX cost of a 3-qubit clifford.
fn cx_cost3(clifford: &Clifford) -> usize {
    let n = 3;

    /// create information transfer matrices R1, R2


    0 /// todo!
}

fn synth_clifford_bm_inner() {

}

